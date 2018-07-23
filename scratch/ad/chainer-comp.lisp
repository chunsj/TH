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

(let* ((x ($variable '(5)))
       (z ($* ($constant -2) x))
       (y ($+ ($expt x ($constant 2)) z ($constant 1))))
  (prn y)
  ($bp! y)
  (prn "DY/DX" ($gradient x)))

(let* ((x ($variable '((1 2 3) (4 5 6))))
       (y ($+ ($expt x ($constant 2)) ($* ($constant -2) x) ($constant 1))))
  (prn y)
  ($bp! y)
  (prn "DY/DX" ($gradient x)))

(defun muladd (x y z) ($+ ($* x y) z))

(let* ((x ($variable ($- ($* 2 (rnd 3 2)) 1)))
       (y ($variable ($- ($* 2 (rnd 3 2)) 1)))
       (z ($variable ($- ($* 2 (rnd 3 2)) 1)))
       (r (muladd x y z)))
  (prn r)
  ($bp! r)
  (prn "X" x)
  (prn "Y" y)
  (prn "DR/DX=Y"($gradient x))
  (prn "DR/DY=X"($gradient y))
  (prn "DR/DZ=1"($gradient z)))
