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

(let ((r/theta (r/variable 0.5)))
  (let ((traces (mcmc/mh (list r/theta) #'binomial-posterior)))
    (prn traces)))

(let ((r/theta (r/variable 0.5)))
  (let ((traces (mcmc/hmc (list r/theta) #'binomial-posterior)))
    (prn traces)))

(let ((r/theta (r/variable 0.5)))
  (let ((traces (mcmc/nuts (list r/theta) #'binomial-posterior)))
    (prn traces)))
