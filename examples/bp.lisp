(defpackage th.bp-example
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :th.bp-example)

;; add
(let* ((a ($parameter '(1 1 1)))
       (b ($parameter '(1 1 1)))
       (out ($add a b)))
  ($bp! out (tensor '(1 2 3)))
  (prn "[1 2 3]")
  (prn ($gradient a))
  (prn ($gradient b)))

;; sub
(let* ((x ($parameter '(1 2 3)))
       (y ($parameter '(3 2 1)))
       (out ($sub x y)))
  ($bp! out (tensor '(1 1 1)))
  (prn "[1 1 1] and [-1 -1 -1]")
  (prn ($gradient x))
  (prn ($gradient y)))

;; dot
(let* ((x ($parameter '(1 2 3)))
       (y (tensor '(2 2 2)))
       (out ($dot x y)))
  (prn "RESULT:" out)
  ($bp! out)
  (prn "[2 2 2]")
  (prn ($gradient x)))

(let* ((a (tensor '(1 1 1)))
       (b ($parameter '(1 2 3)))
       (out ($@ a b)))
  (prn "RESULT:" out)
  ($bp! out)
  (prn "[1 1 1]")
  (prn ($gradient b)))

;; mv
(let* ((X (tensor '((1) (3))))
       (b ($parameter '(10)))
       (out ($mv X b)))
  (prn "RESULT:" out)
  ($bp! out)
  (prn "4")
  (prn ($gradient b)))

(let* ((m ($parameter '((2 0) (0 2))))
       (v (tensor '(2 3)))
       (out ($mv m v)))
  (prn "RESULT:" out)
  ($bp! out)
  (prn "[[2 3] [2 3]]")
  (prn ($gradient m)))

;; mm
(let* ((a ($parameter '((1 1 1) (1 1 1))))
       (b ($parameter '((0.1) (0.1) (0.1))))
       (out ($mm a b)))
  (prn out)
  ($bp! out)
  (prn "[[0.1 0.1 0.1] [0.1 0.1 0.1]]")
  (prn ($gradient a))
  (prn "[[2] [2] [2]]")
  (prn ($gradient b)))

;; mean
(let* ((x ($parameter '((1 2 3) (4 5 6) (7 8 9))))
       (out ($mean x)))
  (prn out)
  ($bp! out)
  (prn "[0.111...]")
  (prn ($gradient x)))
