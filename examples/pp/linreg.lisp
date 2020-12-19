(defpackage :linear-regression
  (:use #:common-lisp
        #:mu
        #:th
        #:th.pp))

(in-package :linear-regression)

(defparameter *xs* (tensor (loop :for i :from 1 :to 200 :collect i)))
(defparameter *ys* ($+ ($normal (tensor ($count *xs*)) 0 1) ($* 2 *xs*)))

(defun lr-posterior (b0 b1 s)
  (let ((prior-b0 (score/normal b0 1 1))
        (prior-b1 (score/normal b1 1 1))
        (prior-s (score/uniform s 0.001 5)))
    (when (and prior-b0 prior-b1 prior-s)
      (let ((m ($* s (eye ($count *xs*)))))
        (let ((likelihood-ys (score/mvn *ys* ($+ ($* b1 *xs*) b0) m)))
          (when likelihood-ys
            ($+ prior-b0 prior-b1 prior-s likelihood-ys)))))))

;; MH - WORKS
(let ((traces (mcmc/mh (list (r/variable 0) (r/variable 0) (r/variable 0.5))
                       #'lr-posterior)))
  (prn traces))

;; HMC
(let ((traces (mcmc/hmc (list (r/variable 0) (r/variable 0) (r/variable 1))
                        #'lr-posterior)))
  (prn traces))

;; NUTS
(let ((traces (mcmc/nuts (list (r/variable 0) (r/variable 0) (r/variable 0.5))
                         #'lr-posterior)))
  (prn traces))
