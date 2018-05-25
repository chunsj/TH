(defpackage :dlfs-01
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :dlfs-01)

;; creating tensor
(let ((x (tensor '(1 2 3))))
  (print x)
  (print (type-of x)))

;; generic operations
(let ((x (tensor '(1 2 3)))
      (y (tensor '(2 4 6))))
  (print ($+ x y))
  (print ($- x y))
  (print ($* x y))
  (print ($/ x y)))

;; constant broadcasting - different from numpy
(let ((x (tensor '(1 2 3)))
      (c 2))
  (print ($/ x ($broadcast c x)))
  (print ($/ x c)))

;; n-dimensional
(let ((a (tensor '((1 2) (3 4)))))
  (print a)
  (print ($size a))
  (print ($type a)))


;; operations on matrices
(let ((a (tensor '((1 2) (3 4))))
      (b (tensor '((3 0) (0 6)))))
  (print ($+ a b))
  (print ($* a b)))

;; broadcasting again
(let ((a (tensor '((1 2) (3 4))))
      (c 10))
  (print ($* a ($broadcast c a)))
  (print ($* a 10)))

;; however, different from numpy style, th needs explicit shape adjust here
(let ((a (tensor '((1 2) (3 4))))
      (b (tensor '(10 20))))
  (print ($* a ($vv (ones ($size a 0)) b)))
  (print ($vv (ones ($size a 0)) b))
  (print ($vv b (ones ($size a 1)))))

;; with support functions
(let ((a (tensor '((1 2) (3 4))))
      (b (tensor '(10 20))))
  (print ($* a ($krows b ($size a 0))))
  (print ($* a ($kcols b ($size a 1))))
  (print ($* a ($broadcast b a))))

;; accessing elements
(let ((x (tensor '((51 55) (14 19) (0 4)))))
  (print x)
  (print ($ x 0))
  (print ($ x 0 1))
  (loop :for i :from 0 :below ($size x 0)
        :do (print ($ x i)))
  (let ((x ($reshape x ($count x))))
    (print x)
    (print ($gather x 0 '(0 2 4)))))
