(defpackage :dlfs-01
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :dlfs-01)

;; creating tensor
(let ((x (tensor '(1 2 3))))
  (prn x)
  (prn (type-of x)))

;; generic operations
(let ((x (tensor '(1 2 3)))
      (y (tensor '(2 4 6))))
  (prn ($+ x y))
  (prn ($- x y))
  (prn ($* x y))
  (prn ($/ x y)))

;; constant broadcasting - different from numpy
(let ((x (tensor '(1 2 3)))
      (c 2))
  (prn ($/ x ($broadcast c x)))
  (prn ($/ x c)))

;; n-dimensional
(let ((a (tensor '((1 2) (3 4)))))
  (prn a)
  (prn ($size a))
  (prn ($type a)))

;; operations on matrices
(let ((a (tensor '((1 2) (3 4))))
      (b (tensor '((3 0) (0 6)))))
  (prn ($+ a b))
  (prn ($* a b)))

;; broadcasting again
(let ((a (tensor '((1 2) (3 4))))
      (c 10))
  (prn ($* a ($broadcast c a)))
  (prn ($* a 10)))

;; however, different from numpy style, th needs explicit shape adjustments
(let ((a (tensor '((1 2) (3 4))))
      (b (tensor '(10 20))))
  (prn ($* a ($vv (ones ($size a 0)) b)))
  (prn ($vv (ones ($size a 0)) b))
  (prn ($vv b (ones ($size a 1)))))

;; with support functions
(let ((a (tensor '((1 2) (3 4))))
      (b (tensor '(10 20))))
  (prn ($* a ($krows b ($size a 0))))
  (prn ($* a ($kcols b ($size a 1))))
  (prn ($* a ($broadcast b a))))

;; accessing elements
(let ((x (tensor '((51 55) (14 19) (0 4)))))
  (prn x)
  (prn ($ x 0))
  (prn ($ x 0 1))
  (loop :for i :from 0 :below ($size x 0)
        :do (prn ($ x i)))
  (let ((x ($reshape x ($count x))))
    (prn x)
    (prn ($gather x 0 '(0 2 4)))))
