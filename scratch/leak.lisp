(defpackage :leak-finding
  (:use #:common-lisp
        #:mu
        #:th
        #:th.ex.data))

(in-package :leak-finding)

(defparameter *w1* (rndn 1024 512))
(defparameter *w2* (rndn 1024 512))
(defparameter *x1* (rndn 1024 512))
(defparameter *x2* (rndn 1024 512))

(defun addm2 (x1 w1 x2 w2) ($addm2 x1 w1 x2 w2))

(defun addm2 (x1 w1 x2 w2) ($+ ($* x1 w1) ($* x2 w2)))

(time
 (progn
   (gcf)
   (loop :for ii :from 0 :below 1000
         :do (loop :for i :from 0 :below 100
                   :do (addm2 *x1* *w1* *x2* *w2*)))
   (gcf)))

(gcf)
