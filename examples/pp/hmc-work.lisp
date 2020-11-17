(defpackage :hmc-work
  (:use #:common-lisp
        #:mu
        #:th
        #:th.distributions
        #:th.mcmc))

(in-package :hmc-work)

;; potential is negative-log-likelihood function
;; position is the parameter we want to compute

(defparameter *data* ($sample (distribution/normal 2.5) 1000))

(defun potential (position)
  ($- ($ll (distribution/normal position) *data*)))

(let ((samples (hmc 300 0 #'potential :step-size 0.05)))
  ;; data mean vs computed parameter which is the mean of the normal distribution
  (list ($mean *data*) ($mean samples)))
