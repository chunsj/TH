(defpackage :hmc-work
  (:use #:common-lisp
        #:mu
        #:th
        #:th.pp))

(in-package :hmc-work)

(defparameter *data* (th.pp:sample/gaussian 2.5 1 1000))

(defun likelihood (mu)
  (score/gaussian *data* mu 1))

(let ((traces (mcmc/hmc (list (r/variable 0)) #'likelihood)))
  (prn "TRACE:" traces)
  (prn "DMEAN:" ($mean *data*)))
