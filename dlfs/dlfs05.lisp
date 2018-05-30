(defpackage :dlfs-05
  (:use #:common-lisp
        #:mu
        #:th
        #:th.db.mnist))

(in-package :dlfs-05)

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

;; running loaded model with test data
(let ((xt ($ *mnist* :test-images))
      (yt ($ *mnist* :test-labels)))
  (print ($count (loop :for i :from 0 :below ($size xt 0)
                       :for xi = ($index xt 0 (list i))
                       :for yi = ($index yt 0 (list i))
                       :for yi* = ($data (mnist-predict ($constant xi)))
                       :for err = ($sum ($abs ($sub ($round yi*) yi)))
                       :when (> err 0)
                         :collect i))))

;; full training
(let* ((x (-> *mnist*
              ($ :train-images)
              ($constant)))
       (y (-> *mnist*
              ($ :train-labels)
              ($constant)))
       (lr 1.3))
  (loop :for i :from 1 :to 100
        :for y* = (mnist-predict x)
        :for loss = (mnist-loss y* y)
        :do (progn
              (when (zerop (mod i 5))
                (print (list i loss))
                (finish-output)
                (sb-ext::gc :full t))
              ($bp! loss)
              ($gd! loss lr)))
  (let ((xt ($ *mnist* :test-images))
        (yt ($ *mnist* :test-labels)))
    (print ($count (loop :for i :from 0 :below ($size xt 0)
                         :for xi = ($index xt 0 (list i))
                         :for yi = ($index yt 0 (list i))
                         :for yi* = ($data (mnist-predict ($constant xi)))
                         :for err = ($sum ($abs ($sub ($round yi*) yi)))
                         :when (> err 0)
                           :collect i)))))
