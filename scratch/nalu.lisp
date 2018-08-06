;; from
;; https://github.com/grananqvist/NALU-tf/blob/master/nalu.py
;;
;; Neural Arithmetic Logic Units
;; Refer - https://arxiv.org/abs/1808.00508

(defpackage :nalu-work
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :nalu-work)

(defparameter *batch-size* 10)
(defparameter *shape* (list 2 1))

(defparameter *dataset* nil)
(defparameter *target* nil)

(defparameter *operation* #'*)

;; generate
(loop :for n :from 0 :below 100
      :do (let ((args (tensor *batch-size* 2))
                (vals (tensor *batch-size* 1)))
            (loop :for i :from 0 :below 10
                  :for n1 = (random 10)
                  :for n2 = (random 10)
                  :for r = (funcall *operation* n1 n2)
                  :do (progn
                        (setf ($ args i 0) n1)
                        (setf ($ args i 1) n2)
                        (setf ($ vals i 0) r)))
            (push args *dataset*)
            (push vals *target*)))

(defparameter *nalu* (parameters))

(defparameter *w-hat* ($parameter *nalu* (vrnt *shape* 0 0.02)))
(defparameter *m-hat* ($parameter *nalu* (vrnt *shape* 0 0.02)))
(defparameter *g* ($parameter *nalu* (vrnt *shape* 0 0.02)))

(defparameter *epochs* 200)

($cg! *nalu*)

(loop :for epoch :from 1 :to *epochs*
      :for iter = 1
      :do (loop :for input :in *dataset*
                :for target :in *target*
                :for x = ($constant input)
                :for w = ($* ($tanh *w-hat*) ($sigmoid *m-hat*))
                :for m = ($exp ($@ ($log ($+ ($abs x) ($constant 1E-7))) w))
                :for g = ($sigmoid ($@ x *g*))
                :for a = ($@ x w)
                :for y* = ($+ ($* g a) ($* ($- ($constant 1) g) m))
                :for y = ($constant target)
                :for d = ($- y* y)
                :for l = ($/ ($dot d d) ($constant *batch-size*))
                :do (progn
                      ($adgd! *nalu*)
                      (when (zerop (rem iter 100))
                        (prn "LOSS:" iter epoch ($data l))
                        (prn ($sum ($- ($round ($data y*)) ($round target)))))
                      (incf iter))))

;; check training accuracy
(loop :for input :in *dataset*
      :for target :in *target*
      :for x = ($constant input)
      :for w = ($* ($tanh *w-hat*) ($sigmoid *m-hat*))
      :for m = ($exp ($@ ($log ($+ ($abs x) ($constant 1E-7))) w))
      :for g = ($sigmoid ($@ x *g*))
      :for a = ($@ x w)
      :for y* = ($+ ($* g a) ($* ($- ($constant 1) g) m))
      :for y = ($constant target)
      :for d = ($- y* y)
      :for l = ($/ ($dot d d) ($constant *batch-size*))
      :do (progn
            ($cg! *nalu*)
            (when (> ($sum ($- ($round ($data y*)) ($data y))) 1E-4)
              (prn "Y*" y*)
              (prn "Y" y))))

;; check test accuracy - generate new data
(defparameter *dataset* nil)
(defparameter *target* nil)
(loop :for n :from 0 :below 10
      :do (let ((args (tensor *batch-size* 2))
                (vals (tensor *batch-size* 1)))
            (loop :for i :from 0 :below 10
                  :for n1 = (random 10)
                  :for n2 = (random 10)
                  :for r = (funcall *operation* n1 n2)
                  :do (progn
                        (setf ($ args i 0) n1)
                        (setf ($ args i 1) n2)
                        (setf ($ vals i 0) r)))
            (push args *dataset*)
            (push vals *target*)))

;; okay, check with new data -
(loop :for input :in *dataset*
      :for target :in *target*
      :for x = ($constant input)
      :for w = ($* ($tanh *w-hat*) ($sigmoid *m-hat*))
      :for m = ($exp ($@ ($log ($+ ($abs x) ($constant 1E-7))) w))
      :for g = ($sigmoid ($@ x *g*))
      :for a = ($@ x w)
      :for y* = ($+ ($* g a) ($* ($- ($constant 1) g) m))
      :for y = ($constant target)
      :for d = ($- y* y)
      :for l = ($/ ($dot d d) ($constant *batch-size*))
      :do (progn
            ($cg! *nalu*)
            (when (> ($sum ($- ($round ($data y*)) ($data y))) 1E-4)
              (prn "**DIFFERENT**")
              (prn "X" x)
              (prn "Y*" y*)
              (prn "Y" y))))
