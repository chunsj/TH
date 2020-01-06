(defpackage :mem-check
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :mem-check)

(defparameter *a* (rndn 1000 1000))
(prn *a*)
(setf *a* nil)
