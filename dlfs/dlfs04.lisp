(defpackage :dlfs-04
  (:use #:common-lisp
        #:mu
        #:th
        #:th.db.mnist))

(in-package :dlfs-04)

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

;; backprop testing
(with-foreign-memory-limit
    (let* ((sels '(0 1 2 3 4))
           (x (-> *mnist*
                  ($ :train-images)
                  ($index 0 sels)))
           (y (-> *mnist*
                  ($ :train-labels)
                  ($index 0 sels)))
           (lr 0.01))
      (loop :for i :from 1 :below 100
            :do (let* ((y* (mnist-predict x))
                       (loss (mnist-loss y* y)))
                  (prn loss)
                  ($gs! loss)
                  ($gd! (list *w1* *b1* *w2* *b2* *w3* *b3*) lr)))
      (prn y)
      (prn ($round ($data (mnist-predict x))))))

(let* ((sels '(0 1 2 3 4))
       (x (-> *mnist*
              ($ :train-images)
              ($index 0 sels)))
       (y (-> *mnist*
              ($ :train-labels)
              ($index 0 sels)))
       (lr 0.01))
  (let* ((y* (mnist-predict x))
         (loss (mnist-loss y* y)))
    ($gs! loss)
    (prn loss)
    (prn (th::$fns *w1*))
    ($gd! *w1* lr)))
