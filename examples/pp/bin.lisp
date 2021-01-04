(defpackage :infer-binomial
  (:use #:common-lisp
        #:mu
        #:th
        #:th.pp))

(in-package :infer-binomial)

(defvar *flips* (tensor '(1 1 1 1 1 1 1 1 1 1 1 0 0 0)))

(defun binomial-posterior (theta)
  (let ((prior-theta (score/beta theta 1 1)))
    (when prior-theta
      (let ((likelihood-flips (score/bernoulli *flips* theta)))
        (when likelihood-flips
          ($+ prior-theta likelihood-flips))))))

(let ((traces (mcmc/mh '(0.5) #'binomial-posterior)))
  (prn traces))
