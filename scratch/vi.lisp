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
