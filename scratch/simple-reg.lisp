(defpackage :simple-reg
  (:use #:common-lisp
        #:mu
        #:th
        #:mplot))

(in-package :simple-reg)

(defparameter *x-train* ($unsqueeze (arange 0 1 0.05) 1))
(defparameter *y-train* ($+ ($fmap (lambda (x) (sin (* 2 pi x))) *x-train*)
                            ($rn! (tensor ($count *x-train*)) 0 0.1)))

(plot-points (loop :for i :from 0 :below ($count *x-train*)
                   :collect (list ($ *x-train* i 0)
                                  ($ *y-train* i 0))))

(defparameter *w* ($parameter ($reshape! (tensor (list pi)) 1 1)))

($cg! *w*)
(loop :for i :from 0 :below 100
      :do (let* ((d ($- ($sin ($@ *x-train* *w*)) *y-train*))
                 (l ($dot d d)))
            (prn ($data l))
            ($gd! *w*)))

(prn *w*)
(prn (* 2 pi))
