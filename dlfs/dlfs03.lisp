(defpackage :dlfs-03
  (:use #:common-lisp
        #:mu
        #:th
        #:th.db.mnist))

(in-package :dlfs-03)

;; sigmoid function
(prn ($sigmoid 0))

;; step function
(defun step-function (x)
  (if ($tensorp x)
      (tensor ($gt x 0))
      ($gt x 0)))

;; testing step-function
(let ((x (tensor '(-1 1 2))))
  (prn x)
  (prn ($gt x 0))
  (prn (step-function x)))

;; testing sigmoid
(let ((x (tensor '(-1 1 2))))
  (prn ($sigmoid x)))

;; relu function
(prn ($relu 1))
(prn ($relu (tensor '(-2 1 2))))

;; multidimensional array
(let ((a (tensor '(1 2 3 4))))
  (prn a)
  (prn ($size a))
  (prn ($size a 0)))

(let ((b (tensor '((1 2) (3 4) (5 6)))))
  (prn b)
  (prn ($ndim b))
  (prn ($size b)))

;; matrix product
(let ((a (tensor '((1 2) (3 4))))
      (b (tensor '((5 6) (7 8)))))
  (prn ($size a))
  (prn ($size b))
  (prn ($@ a b)))

(let ((a (tensor '((1 2 3) (4 5 6))))
      (b (tensor '((1 2) (3 4) (5 6)))))
  (prn ($size a))
  (prn ($size b))
  (prn ($@ a b)))

;; mv
(let ((a (tensor '((1 2) (3 4) (5 6))))
      (b (tensor '(7 8))))
  (prn ($mv a b))
  (prn ($@ a b)))

;; neural network - note that size of x is different from the book
(let ((x (tensor '((1 2))))
      (w (tensor '((1 3 5) (2 4 6)))))
  (prn w)
  (prn ($size w))
  (prn ($@ x w)))

(let ((x (tensor '((1.0 0.5))))
      (w1 (tensor '((0.1 0.3 0.5) (0.2 0.4 0.6))))
      (b1 (tensor '((0.1 0.2 0.3)))))
  (prn ($size w1))
  (prn ($size x))
  (prn ($size b1))
  (let* ((a1 ($+ ($@ x w1) b1))
         (z1 ($sigmoid a1)))
    (prn a1)
    (prn z1)
    (let ((w2 (tensor '((0.1 0.4) (0.2 0.5) (0.3 0.6))))
          (b2 (tensor '((0.1 0.2)))))
      (prn ($size z1))
      (prn ($size w2))
      (prn ($size b2))
      (let* ((a2 ($+ ($@ z1 w2) b2))
             (z2 ($sigmoid a2)))
        (prn a2)
        (prn z2)))))

;; softmax
(let* ((a (tensor '(0.3 2.9 4.0)))
       (y ($softmax a)))
  (prn y)
  (prn ($sum y)))

(let ((a (tensor '(1010 1000 990))))
  (prn ($softmax a)))

;; mnist data loading - takes time, so load and set
(defparameter *mnist* (read-mnist-data))
(prn *mnist*)

;; network parameters
(defparameter *w1* ($parameter (rndn 784 50)))
(defparameter *b1* ($parameter (zeros 50)))
(defparameter *w2* ($parameter (rndn 50 100)))
(defparameter *b2* ($parameter (zeros 100)))
(defparameter *w3* ($parameter (rndn 100 10)))
(defparameter *b3* ($parameter (zeros 10)))

(defun mnist-predict (x)
  (-> x
      ($xwpb *w1* *b1*)
      ($sigmoid)
      ($xwpb *w2* *b2*)
      ($sigmoid)
      ($xwpb *w3* *b3*)
      ($softmax)))

(defun mnist-loss (prediction trueth) ($cee prediction trueth))

;; write to file
(let ((f (file.disk "dlfs/mnist-w1.dat" "w")))
  ($fwrite ($data *w1*) f)
  ($fclose f))
(let ((f (file.disk "dlfs/mnist-b1.dat" "w")))
  ($fwrite ($data *b1*) f)
  ($fclose f))
(let ((f (file.disk "dlfs/mnist-w2.dat" "w")))
  ($fwrite ($data *w2*) f)
  ($fclose f))
(let ((f (file.disk "dlfs/mnist-b2.dat" "w")))
  ($fwrite ($data *b2*) f)
  ($fclose f))
(let ((f (file.disk "dlfs/mnist-w3.dat" "w")))
  ($fwrite ($data *w3*) f)
  ($fclose f))
(let ((f (file.disk "dlfs/mnist-b3.dat" "w")))
  ($fwrite ($data *b3*) f)
  ($fclose f))

;; read from file
(let ((f (file.disk "dlfs/mnist-w1.dat" "r")))
  ($fread ($data *w1*) f)
  ($fclose f))
(let ((f (file.disk "dlfs/mnist-b1.dat" "r")))
  ($fread ($data *b1*) f)
  ($fclose f))
(let ((f (file.disk "dlfs/mnist-w2.dat" "r")))
  ($fread ($data *w2*) f)
  ($fclose f))
(let ((f (file.disk "dlfs/mnist-b2.dat" "r")))
  ($fread ($data *b2*) f)
  ($fclose f))
(let ((f (file.disk "dlfs/mnist-w3.dat" "r")))
  ($fread ($data *w3*) f)
  ($fclose f))
(let ((f (file.disk "dlfs/mnist-b3.dat" "r")))
  ($fread ($data *b3*) f)
  ($fclose f))

;; train data
(prn ($ *mnist* :train-images))

;; run prediction - test
(prn (-> *mnist*
         ($ :train-images)
         ($index 0 '(0 1 2 3 4))
         (mnist-predict)))

(let ((y (-> *mnist*
             ($ :train-images)
             ($index 0 '(0 1 2 3 4))
             (mnist-predict)) )
      (r (-> *mnist*
             ($ :train-labels)
             ($index 0 '(0 1 2 3 4)))))
  (prn y)
  (prn r)
  (prn ($cee y r)))
