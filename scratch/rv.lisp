(defpackage :random-variable
  (:use #:common-lisp
        #:mu
        #:th
        #:th.distributions
        #:th.mcmc))

(in-package :random-variable)

(defgeneric $val (rv))
(defgeneric $logp (rv))

(defclass random-variable ()
  ((g :initform nil)
   (v :initform nil)))

(defmethod print-object ((rv random-variable) stream)
  (format stream "~A" ($val rv)))

(defun rv (generator)
  (let ((n (make-instance 'random-variable)))
    (with-slots (g v) n
      (setf g generator)
      (setf v ($sample g 1)))
    n))

(defmethod $val ((rv random-variable))
  (with-slots (g v) rv
    (unless v
      (setf v ($sample g 1)))
    v))

(defmethod $reset! ((rv random-variable))
  (with-slots (g v) rv
    (setf v ($sample g 1))
    v))

(defmethod $logp ((rv random-variable))
  (let ((v ($val rv)))
    (with-slots (g) rv
      ($ll g v))))

(rv (distribution/normal))
($val (rv (distribution/normal)))
($reset! (rv (distribution/normal)))
($logp (rv (distribution/normal)))
