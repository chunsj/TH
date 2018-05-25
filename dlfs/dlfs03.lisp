(defpackage :dlfs-03
  (:use #:common-lisp
        #:mu
        #:th
        #:th.db.mnist))

(in-package :dlfs-03)

;; sigmoid function
(print ($sigmoid 0))

;; step function
(defun step-function (x)
  (if ($tensorp x)
      (tensor ($gt x 0))
      ($gt x 0)))

;; testing step-function
(let ((x (tensor '(-1 1 2))))
  (print x)
  (print ($gt x 0))
  (print (step-function x)))

;; testing sigmoid
(let ((x (tensor '(-1 1 2))))
  (print ($sigmoid x)))

;; relu function
(print ($relu 1))
(print ($relu (tensor '(-2 1 2))))

;; multidimensional array
(let ((a (tensor '(1 2 3 4))))
  (print a)
  (print ($size a))
  (print ($size a 0)))

(let ((b (tensor '((1 2) (3 4) (5 6)))))
  (print b)
  (print ($ndim b))
  (print ($size b)))

;; matrix product
(let ((a (tensor '((1 2) (3 4))))
      (b (tensor '((5 6) (7 8)))))
  (print ($size a))
  (print ($size b))
  (print ($@ a b)))

(let ((a (tensor '((1 2 3) (4 5 6))))
      (b (tensor '((1 2) (3 4) (5 6)))))
  (print ($size a))
  (print ($size b))
  (print ($@ a b)))

;; mv
(let ((a (tensor '((1 2) (3 4) (5 6))))
      (b (tensor '(7 8))))
  (print ($mv a b))
  (print ($@ a b)))

;; neural network - note that size of x is different from the book
(let ((x (tensor '((1 2))))
      (w (tensor '((1 3 5) (2 4 6)))))
  (print w)
  (print ($size w))
  (print ($@ x w)))

(let ((x (tensor '((1.0 0.5))))
      (w1 (tensor '((0.1 0.3 0.5) (0.2 0.4 0.6))))
      (b1 (tensor '((0.1 0.2 0.3)))))
  (print ($size w1))
  (print ($size x))
  (print ($size b1))
  (let* ((a1 ($+ ($@ x w1) b1))
         (z1 ($sigmoid a1)))
    (print a1)
    (print z1)
    (let ((w2 (tensor '((0.1 0.4) (0.2 0.5) (0.3 0.6))))
          (b2 (tensor '((0.1 0.2)))))
      (print ($size z1))
      (print ($size w2))
      (print ($size b2))
      (let* ((a2 ($+ ($@ z1 w2) b2))
             (z2 ($sigmoid a2)))
        (print a2)
        (print z2)))))

;; softmax
(let* ((a (tensor '(0.3 2.9 4.0)))
       (exp-a ($exp a))
       (sum-exp-a ($sum exp-a))
       (y ($/ exp-a sum-exp-a)))
  (print exp-a)
  (print sum-exp-a)
  (print y))

(let ((a (tensor '(1010 1000 990))))
  (print ($softmax a)))

(let* ((a (tensor '(0.3 2.9 4.0)))
       (y ($softmax a)))
  (print y)
  (print ($sum y)))

;; mnist data loading - takes time, so load and set
(defparameter *mnist* (read-mnist-data))
(print *mnist*)

;; network parameters
(defparameter *w1* ($variable (rndn 784 50)))
(defparameter *b1* ($variable (zeros 50)))
(defparameter *w2* ($variable (rndn 50 100)))
(defparameter *b2* ($variable (zeros 100)))
(defparameter *w3* ($variable (rndn 100 10)))
(defparameter *b3* ($variable (zeros 10)))

(defun mnist-predict (x)
  (-> x
      ($xwpb *w1* *b1*)
      ($sigmoid)
      ($xwpb *w2* *b2*)
      ($sigmoid)
      ($xwpb *w3* *b3*)
      ($softmax)))

;; train data
(print ($ *mnist* :train-images))

;; run prediction - test
(print (-> *mnist*
           ($ :train-images)
           ($index 0 '(0 1 2 3 4))
           ($constant)
           (mnist-predict)))

(defun mnist-loss (prediction trueth) ($bce prediction trueth))
