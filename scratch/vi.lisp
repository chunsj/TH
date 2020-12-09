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

($cg! *vi*)
(loop :repeat 1500
      :for (ypred mu lv) = (vi *x*)
      :for loss = ($neg (elbo ypred *y* mu lv))
      :do ($amgd! *vi*))

(defun quantiles (ys)
  (let* ((nr ($size ys 0))
         (nc ($size ys 1))
         (i/5 (ceiling (* 0.05 nc)))
         (i/50 (ceiling (* 0.5 nc)))
         (i/95 (min nc (ceiling (* 0.95 nc)))))
    (loop :for i :from 0 :below nr
          :for y = ($list ($ ys i))
          :for vs = (sort y #'<)
          :for k = (prn ($min y) ($max y))
          :collect (list ($ vs i/5)
                         ($ vs i/50)
                         ($ vs i/95)))))

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
  (prn ($dot *y* mu)))
