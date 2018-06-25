;; XXX this is just a scratching, please ignore this
;; what i'd like to do is creating something like $bptt!

(defpackage :rnn-test
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :rnn-test)

(defparameter *wx* ($variable (rnd 2 3)))
(defparameter *bx* ($variable (ones 3)))

;; plain affine transformation
(prn ($xwpb ($constant (tensor '((1 0)))) *wx* *bx*))

;; plain feed forward layer
(prn (-> ($constant (tensor '((1 0))))
         ($xwpb *wx* *bx*)
         ($sigmoid)))

(defparameter *wh* ($variable (rnd 3 3)))
(defparameter *bh* ($variable (ones 3)))

(defparameter *w2* ($variable (rnd 3 1)))
(defparameter *b2* ($variable (ones 1)))

(defun recurrent (x xp wx bx wh bh)
  (if xp
      ($add ($xwpb x wx bx) ($xwpb xp wh bh))
      ($xwpb x wx bx)))

(defun predict (x xp)
  (let ((h1 (recurrent x xp *wx* *bx* *wh* *bh*)))
    (list (-> h1
              ($sigmoid)
              ($xwpb *w2* *b2*)
              ($sigmoid))
          h1)))

(let ((input (tensor '((0 0)
                       (1 0)
                       (0 1)
                       (1 1))))
      (output ($reshape (tensor '(0 1 1 0)) 4 1))
      (xp nil))
  (loop :for iter :from 1 :to 50
        :do (loop :for i :from 0 :below ($size input 0)
                  :for x = ($constant ($index input 0 (list i)))
                  :for y = ($constant ($index output 0 (list i)))
                  :for o = (predict x xp)
                  :for y* = (car o)
                  :for a = (cadr o)
                  :do (let* ((d ($sub y* y))
                             (l ($dot d d)))
                        ($bp! l)
                        ($gd! l 0.01)
                        (setf xp a))))
  (loop :for i :from 0 :below ($size input 0)
        :for x = ($constant ($index input 0 (list i)))
        :for y = ($constant ($index output 0 (list i)))
        :for o = (predict x xp)
        :for y* = (car o)
        :for a = (cadr o)
        :do (progn
              (prn x ($ ($data y*) 0 0) ($ ($data y) 0 0))
              (setf xp a))))

(prn *wx*)
(prn *wh*)
