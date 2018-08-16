(ql:quickload :opticl)

(defpackage :rbm
  (:use #:common-lisp
        #:mu
        #:th
        #:th.db.mnist))

(in-package :rbm)

;; load mnist data, takes ~22 secs in macbook 2017
(defparameter *mnist* (read-mnist-data))

;; mnist data has following dataset
;; train-images, train-labels and test-images, test-labels
(print *mnist*)

(defparameter *batch-size* 60)
(defparameter *batch-count* (/ 60000 *batch-size*))

(defparameter *mnist-train-image-batches*
  (loop :for i :from 0 :below *batch-count*
        :for range = (loop :for k :from (* i *batch-size*) :below (* (1+ i) *batch-size*)
                           :collect k)
        :collect ($contiguous! ($index ($ *mnist* :train-images) 0 range))))

(defparameter *n-vis* 784)
(defparameter *n-hin* 500)

(defparameter *rbm* (parameters))

(defparameter *w* ($parameter *rbm* ($* 1E-2 (rndn *n-vis* *n-hin*))))
(defparameter *vb* ($parameter *rbm* (zeros 1 *n-vis*)))
(defparameter *hb* ($parameter *rbm* (zeros 1 *n-hin*)))
(defparameter *k* 1)

(defun sample-from-p (p)
  ($relu ($sign ($- p ($constant (apply #'rnd ($size p)))))))

(defun v-to-h (v)
  (let* ((nrow ($size v 0))
         (os ($constant (ones nrow 1)))
         (ah ($+ ($@ v *w*) ($@ os *hb*)))
         (ph ($sigmoid ah))
         (sh (sample-from-p ph)))
    (cons ph sh)))

(defun h-to-v (h)
  (let* ((nrow ($size h 0))
         (os ($constant (ones nrow 1)))
         (av ($+ ($@ h ($transpose *w*)) ($@ os *vb*)))
         (pv ($sigmoid av))
         (sv (sample-from-p pv)))
    (cons pv sv)))

(defun run (v)
  (let* ((rvh (v-to-h v))
         (h (cdr rvh))
         (vr nil))
    (loop :for i :from 0 :below *k*
          :for rhv = (h-to-v h)
          :for pv = (car rhv)
          :for cv = (cdr rhv)
          :for rvh = (v-to-h cv)
          :for ch = (cdr rvh)
          :do (setf h ch
                    vr cv))
    (cons v vr)))

(defun free-energy (v)
  (let* ((vbias ($@ v ($transpose *vb*)))
         (nrow ($size v 0))
         (os ($constant (ones nrow 1)))
         (wxb ($+ ($@ v *w*) ($@ os *hb*)))
         (hidden ($sum ($log ($+ ($exp wxb) 1)) 1)))
    ($mean ($- ($neg hidden) vbias))))

(defun opt! () ($gd! *rbm* 0.1))

(defun mean (vs) (* 1D0 (/ (reduce #'+ vs) ($count vs))))

(defparameter *epoch* 10)

($cg! *rbm*)

(loop :for epoch :from 1 :to (min 1 *epoch*)
      :for loss = nil
      :do (progn
            ($cg! *rbm*)
            (loop :for input :in (subseq *mnist-train-image-batches* 0 1)
                  :for sample = ($bernoulli input input)
                  :for res = (run ($constant sample))
                  :for v = (car res)
                  :for v1 = (cdr res)
                  :for l = ($- (free-energy v) (free-energy v1))
                  :do (progn
                        (push ($data l) loss)
                        (opt!)))
            (prn (mean loss))))

(defparameter *output* (format nil "~A/Desktop" (user-homedir-pathname)))

(defun outpng (data fname &optional (w 28) (h 28))
  (let ((img (opticl:make-8-bit-gray-image w h))
        (d ($reshape data w h)))
    (loop :for i :from 0 :below h
          :do (loop :for j :from 0 :below w
                    :do (progn
                          (setf (aref img i j) (round (* 255 ($ d i j)))))))
    (opticl:write-png-file (format nil "~A/~A" *output* fname) img)))

(let* ((input (car *mnist-train-image-batches*))
       (sample ($bernoulli input input))
       (res (run ($constant sample)))
       (v ($data (car res)))
       (v1 ($data (cdr res))))
  (outpng ($index v 0 0) "outv.png")
  (outpng ($index v1 0 0) "outv1.png"))
