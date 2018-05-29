(defpackage :dlfs-04
  (:use #:common-lisp
        #:mu
        #:th
        #:th.db.mnist))

(in-package :dlfs-04)

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

(defun mnist-loss (prediction trueth) ($cee prediction trueth))

;; write to file
(let ((f (file.disk "mnist-w1.dat" "w")))
  ($fwrite ($data *w1*) f)
  ($fclose f))
(let ((f (file.disk "mnist-b1.dat" "w")))
  ($fwrite ($data *b1*) f)
  ($fclose f))
(let ((f (file.disk "mnist-w2.dat" "w")))
  ($fwrite ($data *w2*) f)
  ($fclose f))
(let ((f (file.disk "mnist-b2.dat" "w")))
  ($fwrite ($data *b2*) f)
  ($fclose f))
(let ((f (file.disk "mnist-w3.dat" "w")))
  ($fwrite ($data *w3*) f)
  ($fclose f))
(let ((f (file.disk "mnist-b3.dat" "w")))
  ($fwrite ($data *b3*) f)
  ($fclose f))

;; read from file
(let ((f (file.disk "mnist-w1.dat" "r")))
  ($fread ($data *w1*) f)
  ($fclose f))
(let ((f (file.disk "mnist-b1.dat" "r")))
  ($fread ($data *b1*) f)
  ($fclose f))
(let ((f (file.disk "mnist-w2.dat" "r")))
  ($fread ($data *w2*) f)
  ($fclose f))
(let ((f (file.disk "mnist-b2.dat" "r")))
  ($fread ($data *b2*) f)
  ($fclose f))
(let ((f (file.disk "mnist-w3.dat" "r")))
  ($fread ($data *w3*) f)
  ($fclose f))
(let ((f (file.disk "mnist-b3.dat" "r")))
  ($fread ($data *b3*) f)
  ($fclose f))

;; backprop testing
(let* ((sels '(0 1 2 3 4))
       (x (-> *mnist*
              ($ :train-images)
              ($index 0 sels)
              ($constant)))
       (y (-> *mnist*
              ($ :train-labels)
              ($index 0 sels)
              ($constant)))
       (lr 0.01))
  (loop :for i :from 1 :below 100
        :do (let* ((y* (mnist-predict x))
                   (loss (mnist-loss y* y)))
              (print loss)
              ($bp! loss)
              ($gd! loss lr)))
  (print y)
  (print ($round ($data (mnist-predict x)))))
