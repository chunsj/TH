(defpackage :etc-examples
  (:use #:common-lisp
        #:mu
        #:th))

;; some examples from numcl project
;; https://github.com/numcl/numcl/blob/master/example.lisp

(in-package :etc-examples)

;; creation
(prn (arange 0 10))

;; reshaping
(prn ($reshape (arange 0 10) 2 5))

;; arange with negative step
(prn (arange 10 -10 -3))

;; concatenation
(prn ($cat (zeros 10) (ones 10)))
(prn ($cat ($reshape (zeros 10) 2 5)
           ($reshape (ones 10) 2 5)))
(prn ($cat ($reshape (zeros 10) 2 5)
           ($reshape (ones 10) 2 5)
           1))
