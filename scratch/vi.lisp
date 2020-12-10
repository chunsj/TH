;; https://www.ritchievink.com/blog/2019/09/16/variational-inference-from-scratch/

(defpackage :vi-learn
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers))

(in-package :vi-learn)

(defun generate-dataset (&optional (n 150))
  (let ((xmin -20)
        (xmax 60)
        (w0 0.125)
        (b0 5))
    (labels ((s (x)
               (let ((g ($div ($sub x xmin) (- xmax xmin))))
                 ($mul 3 ($add 0.25 ($square g))))))
      (let* ((x ($add ($mul (sample/uniform 0 1 n) (- xmax xmin))
                      xmin))
             (eps ($mul (sample/normal 0 1 n) (s x)))
             (y ($add eps ($add ($* w0 x ($add 1 ($sin x))) b0)))
             (y ($div ($sub y ($mean y)) ($sd y)))
             (indices (sort (loop :for i :from 0 :below n :collect i)
                            (lambda (a b) (< ($ x a) ($ x b))))))
        (list (tensor (loop :for i :in indices
                            :collect ($ x i)))
              (tensor (loop :for i :in indices
                            :collect ($ y i))))))))

(defparameter *dataset* (generate-dataset))
(defparameter *x* ($reshape ($0 *dataset*) 150 1))
(defparameter *y* ($reshape ($1 *dataset*) 150 1))

;; maximum likelihood estimation
(defparameter *mle* (sequential-layer
                     (affine-layer 1 20 :activation :relu)
                     (affine-layer 20 1 :activation :nil)))

;; what we get here is the best model parameter of *mle* assuming gaussian likelihood(mse).
;; y ~ N(g_theta(x), sigma^2)
;; theta_mle = argmax_theta PI P(y_i | theta)
;; g_theta is *mle*.
(loop :repeat 200
      :for ypred = ($execute *mle* *x*)
      :for loss = ($mse ypred *y*)
      :do ($amgd! *mle*))


(defparameter *qmu* (sequential-layer
                     (affine-layer 1 20 :activation :relu)
                     (affine-layer 20 10 :activation :relu)
                     (affine-layer 10 1 :activation :nil)))

(defparameter *qlv* (sequential-layer
                     (affine-layer 1 20 :activation :relu)
                     (affine-layer 20 10 :activation :relu)
                     (affine-layer 10 1 :activation :nil)))

(defparameter *vi* (list *qmu* *qlv*))

(defun reparameterize (mu lv)
  (let ((s ($add ($exp ($mul 0.5 lv)) 1E-5)))
    ($add mu ($mul s (apply #'$reshape (sample/normal 0 1($count s)) ($size s))))))

(defun vi (x &key (trainp T))
  (let ((mu ($execute *qmu* x :trainp trainp))
        (lv ($execute *qlv* x :trainp trainp)))
    (list (reparameterize mu lv)
          mu
          lv)))

(defun ll-gaussian (y mu lv)
  (let ((s ($exp ($mul 0.5 lv))))
    ($sub ($mul -0.5 ($log ($* 2 pi ($square s))))
          ($mul ($div 1 ($mul 2 ($square s)))
                ($square ($sub y mu))))))

(defun elbo (ypred y mu lv)
  (let ((ll (ll-gaussian y mu lv))
        (lp (ll-gaussian ypred ($zero mu) ($log ($one lv))))
        (lpq (ll-gaussian ypred mu lv)))
    ($mean ($+ ll lp ($neg lpq)))))

(defun loss-analytic (ypred y mu lv)
  (let ((reconstruction-error ($sum ($mul 0.5 ($square ($sub y ypred)))))
        (kld ($mul -0.5 ($sum ($+ 1 lv ($neg ($square mu)) ($neg ($exp lv)))))))
    ($sum ($add reconstruction-error kld))))

($cg! *vi*)
(loop :repeat 1500
      :for (ypred mu lv) = (vi *x*)
      :for loss = ($neg (elbo ypred *y* mu lv))
      :do ($amgd! *vi*))

($cg! *vi*)
(loop :repeat 1500
      :for (ypred mu lv) = (vi *x*)
      :for loss = (loss-analytic ypred *y* mu lv)
      :do ($amgd! *vi*))

(let* ((ys (-> (loop :repeat 1000
                     :for (y m lv) = (vi *x* :trainp nil)
                     :collect y)
               ($catn 1)
               ($sort 1 nil)
               (car)))
       (nr ($size ys 0))
       (nc ($size ys 1))
       (i/5 (ceiling (* 0.05 nc)))
       (i/50 (ceiling (* 0.5 nc)))
       (i/95 (min nc (ceiling (* 0.95 nc))))
       (q1 (tensor nr))
       (mu (tensor nr))
       (q2 (tensor nr)))
  (loop :for i :from 0 :below nr
        :do (setf ($ q1 i) ($ ys i i/5)
                  ($ mu i) ($ ys i i/50)
                  ($ q2 i) ($ ys i i/95)))
  (prn ($mean ($square ($sub *y* mu)))))

(defclass variational-affine-layer (layer)
  ((wmu :initform nil)
   (wp :initform nil)
   (bmu :initform nil)
   (bp :initform nil)
   (a :initform nil)
   (kld :initform nil)))

(defun variational-affine-layer (input-size output-size &key (activation :sigmoid) (biasp t))
  (let ((n (make-instance 'variational-affine-layer)))
    (with-slots (wmu wp bmu bp a) n
      (setf a (th.layers::afn activation))
      (when biasp
        (setf bmu ($parameter (zeros output-size))
              bp ($parameter (zeros output-size))))
      (setf wmu ($parameter ($normal (tensor input-size output-size) 0 0.001))
            wp ($parameter ($normal (tensor input-size output-size) -2.5 0.001))))
    n))

(defmethod $train-parameters ((l variational-affine-layer))
  (with-slots (wmu wp bmu bp) l
    (if bmu
        (list wmu wp bmu bp)
        (list wmu wp))))

(defmethod $parameters ((l variational-affine-layer))
  ($train-parameters l))

(defun variational-affine-layer-reparameterize (mu p &key (trainp t))
  (if trainp
      (let ((s ($log ($add 1 ($exp p))))
            (eps ($normal ($data p) 0 1)))
        ($add mu ($mul eps s)))
      (let ((s ($log ($add 1 ($exp ($data p)))))
            (eps ($normal ($data p) 0 1)))
        ($add ($data mu) ($mul eps s)))))

(defun variational-affine-layer-w (l &key (trainp t))
  (with-slots (wmu wp) l
    (variational-affine-layer-reparameterize wmu wp :trainp trainp)))

(defun variational-affine-layer-b (l &key (trainp t))
  (with-slots (bmu bp) l
    (when bmu
      (variational-affine-layer-reparameterize bmu bp :trainp trainp))))

(defgeneric layer-kl-divergence (layer))

(defmethod layer-kl-divergence ((l layer)) 0)

(defun variational-affine-layer-kl-divergence (z mu p &optional (ps 1))
  (labels ((ll-normal (y mu p)
             (let ((s ($log ($add 1 ($exp p)))))
               ($sub ($mul -0.5 ($log ($* 2 pi ($square s))))
                     ($mul ($div 1 ($mul 2 ($square s)))
                           ($square ($sub y mu)))))))
    (let ((log-prior (score/normal z 0 ps))
          (log-pq ($sum (ll-normal z mu p))))
      ($sub log-pq log-prior))))

(defmethod $execute ((l variational-affine-layer) x &key (trainp t))
  (with-slots (wmu wp bmu bp a kld) l
    (let ((w (variational-affine-layer-w l :trainp trainp))
          (b (variational-affine-layer-b l :trainp trainp)))
      (when trainp
        (if b
            (setf kld ($div ($add (variational-affine-layer-kl-divergence w
                                                                          wmu
                                                                          wp)
                                  (variational-affine-layer-kl-divergence b
                                                                          bmu
                                                                          bp))
                            ($size x 0)))
            (setf kld ($div (variational-affine-layer-kl-divergence w
                                                                    wmu
                                                                    wp)
                            ($size x 0)))))
      (if a
          (funcall a ($affine x w b))
          ($affine x w b)))))

(defmethod layer-kl-divergence ((l variational-affine-layer))
  (with-slots (kld) l
    kld))

(defmethod layer-kl-divergence ((ls list))
  (let ((kld 0))
    (loop :for l :in ls :do (setf kld ($add kld (layer-kl-divergence l))))
    kld))

(defparameter *bayesnn* (sequential-layer
                         (variational-affine-layer 1 20 :activation :relu)
                         (variational-affine-layer 20 20 :activation :relu)
                         (variational-affine-layer 20 1 :activation :nil)))

(defun bnn-reconstruction-error (ypred y)
  (let ((s ($mul 0.1 ($one y))))
    ($sum ($neg ($sub ($mul -0.5 ($log ($* 2 pi ($square s))))
                      ($mul ($div 1 ($mul 2 ($square s)))
                            ($square ($sub y ypred))))))))

(defun bnn-loss (nn y ypred)
  (let ((reconstruction-error (bnn-reconstruction-error ypred y))
        (kld (layer-kl-divergence nn)))
    ($add reconstruction-error kld)))

($cg! *bayesnn*)
(loop :repeat 20000
      :for i :from 1
      :for ypred = ($execute *bayesnn* *x*)
      :for loss = (bnn-loss *bayesnn* *y* ypred)
      :do (progn
            (when (zerop (rem i 1000)) (prn loss))
            ($amgd! *bayesnn*)))

(let* ((ys (-> (loop :repeat 1000
                     :for y = ($evaluate *bayesnn* *x*)
                     :collect y)
               ($catn 1)
               ($sort 1 nil)
               (car)))
       (nr ($size ys 0))
       (nc ($size ys 1))
       (i/5 (ceiling (* 0.05 nc)))
       (i/50 (ceiling (* 0.5 nc)))
       (i/95 (min nc (ceiling (* 0.95 nc))))
       (q1 (tensor nr))
       (mu (tensor nr))
       (q2 (tensor nr)))
  (loop :for i :from 0 :below nr
        :do (setf ($ q1 i) ($ ys i i/5)
                  ($ mu i) ($ ys i i/50)
                  ($ q2 i) ($ ys i i/95)))
  (prn ($mean ($square ($sub *y* q1)))
       ($mean ($square ($sub *y* mu)))
       ($mean ($square ($sub *y* q2)))))
