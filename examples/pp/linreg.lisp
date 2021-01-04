(defpackage :linear-regression
  (:use #:common-lisp
        #:mu
        #:th
        #:th.pp))

(in-package :linear-regression)

(defparameter *xs* (tensor (loop :for i :from 1 :to 200 :collect i)))
(defparameter *ys* ($+ ($normal (tensor ($count *xs*)) 0 1) ($* 2 *xs*)))

(defun lr-posterior (b0 b1 s)
  (let ((prior-b0 (score/normal b0 0 1))
        (prior-b1 (score/normal b1 1 1))
        (prior-s (score/normal s 0 1)))
    (when (and prior-b0 prior-b1 prior-s)
      (let ((ll (score/gaussian *ys* ($add b0 ($mul b1 *xs*)) ($exp s))))
        (when ll
          ($+ ($+ prior-b0 prior-b1 prior-s ll)))))))

(let ((traces (mcmc/mh '(0.0 1.0 0.0) #'lr-posterior)))
  (prn traces))
