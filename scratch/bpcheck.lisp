(defpackage :backprop-check
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :backprop-check)

;; sum
(let* ((x ($parameter (tensor '((1 2 3) (4 5 6)))))
       (ex ($exp x))
       (s ($sum ex))
       (v ($/ ex s))
       (y ($* ($- v 1) ($- v 1)))
       (z ($sum y)))
  (prn "Z:" z)
  (prn ($gradient x)))

(let ((x ($parameter (tensor '((1 2 3) (4 5 6))))))
  ($sum ($+ x x) 1)
  ($gradient x))

($gradient ($sum ($parameter (tensor '((1 2 3) (4 5 6))))))

(let* ((x ($parameter (tensor '((1 2 3) (4 5 6)))))
       (y ($exp ($* 2 x)))
       (z ($sum y)))
  z
  ($gradient x))

(let* ((x ($parameter (tensor '((1 2 3) (4 5 6)))))
       (ex ($exp x))
       (s ($sum ex))
       (v ($/ ex s))
       (y ($* ($- v 1) ($- v 1)))
       (z ($sum y)))
  (prn z)
  (prn ($gradient x)))

(let* ((x ($parameter (tensor '((1 2 3) (4 5 6)))))
       (ex ($exp x))
       (s ($sum ex))
       (v ($/ ex s))
       (w ($sum v)))
  (prn w)
  (prn ($gradient x)))

(let* ((x ($parameter (tensor '((1 2 3) (4 5 6)))))
       (s ($sum x))
       (v ($- x s)))
  (prn v)
  (prn ($gradient x)))

(let* ((x1 ($parameter 1))
       (x2 ($parameter 2))
       (x3 ($parameter 3))
       (s ($+ x1 x2 x3))
       (v1 ($/ x1 s))
       (v2 ($/ x2 s))
       (v3 ($/ x3 s))
       (w ($+ v1 v2 v3)))
  (prn w)
  (prn ($gradient s))
  (prn ($gradient v1))
  (prn ($gradient x3)))

(let* ((x ($parameter (tensor '((1 2 3) (4 5 6)))))
       (s ($/ 10 ($sum x))))
  (prn s)
  (prn ($gradient x)))

;; $div has a problem?

(let* ((x ($parameter (tensor '((1 2 3) (4 5 6)))))
       (s ($sum x))
       (z ($/ x s)))
  (prn z)
  (prn ($gradient s))
  (prn ($gradient x))
  (prn ($- ($/ 1 s) ($* ($/ s x) ($/ x ($* s s))))))

(let* ((x ($parameter (tensor '((1 2 3)))))
       (s ($sum x))
       (y ($* x s)))
  (prn y)
  ($gs! y (ones 1 3))
  (prn ($gradient x)))

(let* ((x1 ($parameter (tensor '((1)))))
       (x2 ($parameter (tensor '((2)))))
       (x3 ($parameter (tensor '((3)))))
       (s ($+ x1 x2 x3))
       (y1 ($* x1 s))
       (y2 ($* x2 s))
       (y3 ($* x3 s)))
  (prn y1 y2 y3)
  ($gs! y1 (ones 1 1))
  ($gs! y2 (ones 1 1))
  ($gs! y3 (ones 1 1))
  (prn ($gradient x1) ($gradient x2) ($gradient x3)))

;; XXX sum with mul and div has problems
;; fix following
(let* ((x ($parameter (tensor '((1 2 3)))))
       (s ($sum x))
       (y ($* x s)))
  (prn y)
  (prn ($gradient x))
  (prn ($gradient s)))

(let* ((x1 ($parameter (tensor '((1)))))
       (x2 ($parameter (tensor '((2)))))
       (x3 ($parameter (tensor '((3)))))
       (s ($+ x1 x2 x3))
       (y1 ($* x1 s))
       (y2 ($* x2 s))
       (y3 ($* x3 s))
       )
  (prn y1 y2 y3)
  (prn ($gradient x1) ($gradient x2) ($gradient x3))
  (prn ($gradient s)))
