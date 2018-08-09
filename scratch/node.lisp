(defpackage :node-scratch
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :node-scratch)

(prn ($add ($variable '(1 2 3)) 3))
(prn ($add 3 ($variable '(1 2 3))))
(prn ($add ($variable '(1 2 3)) ($constant 3)))
(prn ($add ($variable '(1 2 3)) (tensor '(1 2 3))))
(prn ($add ($variable '(1 2 3)) ($constant '(1 2 3))))
(prn ($+ ($variable '(1 2 3)) 3))
(prn ($+ 3 ($variable '(1 2 3))))
(prn ($+ ($variable '(1 2 3)) ($constant 3)))
(prn ($+ ($variable '(1 2 3)) (tensor '(1 2 3))))
(prn ($+ ($variable '(1 2 3)) ($constant '(1 2 3))))

(prn ($sub ($variable '(1 2 3)) 3))
(prn ($sub 3 ($variable '(1 2 3))))
(prn ($sub ($variable '(1 2 3)) ($constant 3)))
(prn ($sub ($variable '(1 2 3)) (tensor '(1 2 3))))
(prn ($sub ($variable '(1 2 3)) ($constant '(1 2 3))))
(prn ($- ($variable '(1 2 3)) 3))
(prn ($- 3 ($variable '(1 2 3))))
(prn ($- ($variable '(1 2 3)) ($constant 3)))
(prn ($- ($variable '(1 2 3)) (tensor '(1 2 3))))
(prn ($- ($variable '(1 2 3)) ($constant '(1 2 3))))

(prn ($* ($variable '(1 2 3)) 3))
(prn ($* 3 ($variable '(1 2 3))))

(prn ($/ ($variable '(1 2 3)) 3))
(prn ($/ 3 ($variable '(1 2 3))))

(prn ($@ (rnd 2 2) ($variable (rnd 2 2))))
;; do i need special broadcasted add operator?
(prn ($+ ($@ (rnd 2 2) ($variable (rnd 2 2))) ($@ (ones 2 1) (tensor '((2 2))))))
