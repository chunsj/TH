(defpackage :hmc-work
  (:use #:common-lisp
        #:mu
        #:th
        #:th.pp))

(in-package :hmc-work)

;; potential is negative-log-likelihood function
;; position is the parameter we want to compute

(defparameter *data* (th.pp:sample/gaussian 2.5 1 1000))

(defun likelihood (position)
  (score/gaussian *data* position 1))

(let ((traces (mcmc/hmc (list (r/variable 0)) #'likelihood)))
  (prn "TRACES:" traces)
  (prn "DMEAN:" ($mean *data*)))
