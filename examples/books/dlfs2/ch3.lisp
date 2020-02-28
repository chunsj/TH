(defpackage :dlfs-ch3
  (:use #:common-lisp
        #:mu
        #:th
        #:th.ex.data
        #:th.text))

(in-package :dlfs-ch3)

(let* ((c (tensor '((1 0 0 0 0 0 1))))
       (c2 ($select ($nonzero c) 1 1))
       (w (rndn 7 3)))
  (prn ($@ c w))
  (prn ($wimb c2 w))
  (prn ($wemb c w)))
