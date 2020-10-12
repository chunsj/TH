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

(defparameter *population* ($sample (distribution/normal 2.5) 5000))
(defparameter *observation* (tensor (loop :repeat 1000
                                          :collect ($ *population* (random 5000)))))

($mean *population*)
($mean *observation*)

(defun potential (position)
  ($- ($ll (distribution/normal position) *observation*)))

(let ((samples (hmc 1000 0 #'potential :step-size 0.05)))
  ;; data mean vs computed parameter which is the mean of the normal distribution
  (list ($mean *population*) ($mean *observation*) ($mean samples)))

;; model
;;
;; parameters
;; distributions - parameters
;; observations - distributions - parameters
;; likelihood - observations - distributions - parameters

;; XXX
;; bare functional log-likelihood and other functions should be in
;; the th.distributions.
;; current class based implementations should depends on them.


(defclass mymodel ()
  ((mu :initform 0)
   (n :initform (distribution/normal 0))))

(defmethod $ll ((m mymodel) data)
  (with-slots (mu n) m
    (setf ($ n :mu) mu)
    ($ll n data)))

(defun $nll (m mu data)
  (let ((muin mu))
    (with-slots (mu) m
      (setf mu muin))
    ($- ($ll m data))))

(defparameter *model* (make-instance 'mymodel))

(defun potential (position) ($nll *model* position *observation*))

(let ((samples (hmc 1000 0 #'potential :step-size 0.05)))
  ;; data mean vs computed parameter which is the mean of the normal distribution
  (list ($mean *population*) ($mean *observation*) ($mean samples)))
