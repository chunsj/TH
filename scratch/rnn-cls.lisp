;; from
;; https://github.com/pangolulu/rnn-from-scratch
;; but this is no better than task's.

(defpackage :rnn-test2
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :rnn-test2)

(defgeneric forward (object &rest arguments))
(defgeneric backward (object &rest arguments))

(defclass multiply-gate () ())

(defmethod forward ((self multiply-gate) &rest arguments)
  (let ((w ($0 arguments))
        (x ($1 arguments)))
    ($mm w x)))

(defmethod backward ((self multiply-gate) &rest arguments)
  (let ((w ($0 arguments))
        (x ($1 arguments))
        (dz ($2 arguments)))
    (list ($mm ($transpose dz) x) ($mm ($transpose w) dz))))

(defclass add-gate () ())

(defmethod forward ((self add-gate) &rest arguments)
  (let ((x1 ($0 arguments))
        (x2 ($1 arguments)))
    ($add x1 x2)))

(defmethod backward ((self add-gate) &rest arguments)
  (let ((x1 ($0 arguments))
        (x2 ($1 arguments))
        (dz ($2 arguments)))
    (list ($mul dz ($one x1)) ($mul dz ($one x2)))))

(defclass sigmoid-gate () ())

(defmethod forward ((self sigmoid-gate) &rest arguments)
  (let ((x ($0 arguments)))
    ($sigmoid x)))

(defmethod backward ((self sigmoid-gate) &rest arguments)
  (let ((x ($0 arguments))
        (d ($1 arguments)))
    (let ((output (forward self x)))
      (list ($mul ($mul ($- 1 output) output) d)))))

(defclass tanh-gate () ())

(defmethod forward ((self tanh-gate) &rest arguments)
  (let ((x ($0 arguments)))
    ($tanh x)))

(defmethod backward ((self tanh-gate) &rest arguments)
  (let ((x ($0 arguments))
        (d ($1 arguments)))
    (let ((output (forward self x)))
      (list ($mul ($- 1 ($mul output output)) d)))))

(defparameter mulgate (make-instance 'multiply-gate))
(defparameter addgate (make-instance 'add-gate))
(defparameter activation (make-instance 'tanh-gate))

(defclass rnn-layer ()
  ((mulu :initform nil :accessor mulu)
   (mulw :initform nil :accessor mulw)
   (add :initform nil :accessor adduw)
   (s :initform nil :accessor act)
   (mulv :initform nil :accessor mulv)))

(defmethod forward ((self rnn-layer) &rest arguments)
  (let ((x ($0 arguments))
        (sp ($1 arguments))
        (u ($2 arguments))
        (w ($3 arguments))
        (v ($4 arguments)))
    (setf (mulu self) (forward mulgate u x))
    (setf (mulw self) (forward mulgate w sp))
    (setf (adduw self) (forward addgate (mulu self) (mulw self)))
    (setf (act self) (forward activation (adduw self)))
    (setf (mulv self) (forward mulgate v (act self)))))

(defmethod backward ((self rnn-layer) &rest arguments)
  (let ((x ($0 arguments))
        (sp ($1 arguments))
        (u ($2 arguments))
        (w ($3 arguments))
        (v ($4 arguments))
        (ds ($5 arguments))
        (dmulv ($6 arguments)))
    (forward self x sp u w v)
    (let* ((mbr (backward mulgate v (act self) dmulv))
           (dv ($0 mbr))
           (dsv ($1 mbr))
           (dadd nil))
      (setf ds ($+ dsv ds))
      (setf dadd (backward activation (adduw self) ds))
      (let* ((abr (backward addgate (mulw self) (mulu self) dadd))
             (dmulw ($0 abr))
             (dmulu ($1 abr)))
        (let* ((mbr2 (backward mulgate w sp dmulw))
               (dw ($0 mbr2))
               (dsp ($1 mbr2)))
          (let* ((mbr3 (backward mulgate u x dmulu))
                 (du ($0 mbr3)))
            (list dsp du dw dv)))))))
