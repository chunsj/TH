(defpackage :chainer-comp
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :chainer-comp)

(let* ((x ($variable '(5)))
       (y ($+ ($expt x ($constant 2)) ($* ($constant -2) x) ($constant 1)))
       (y1 ($expt x ($constant 2)))
       (y2 ($* ($constant -2) x))
       (y3 ($+ y1 y2 ($constant 1))))
  (prn y)
  ($bp! y)
  (prn "DY/DX" ($gradient x))
  (prn y3)
  ($bp! y3)
  (prn "DY3/DX" ($gradient x)))
