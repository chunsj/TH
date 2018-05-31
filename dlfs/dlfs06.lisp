(defpackage :dlfs-06
  (:use #:common-lisp
        #:mu
        #:th
        #:th.db.mnist))

(in-package :dlfs-06)

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

(defun mnist-reset-parameters ()
  (setf *w1* ($variable ($div (rndn 784 50) ($sqrt 784))))
  (setf *b1* ($variable (zeros 50)))
  (setf *w2* ($variable ($div (rndn 50 100) ($sqrt 50))))
  (setf *b2* ($variable (zeros 100)))
  (setf *w3* ($variable ($div (rndn 100 10) ($sqrt 100))))
  (setf *b3* ($variable (zeros 10))))

(defun mnist-predict (x)
  (-> x
      ($xwpb *w1* *b1*)
      ($sigmoid)
      ($xwpb *w2* *b2*)
      ($sigmoid)
      ($xwpb *w3* *b3*)
      ($softmax)))

(defun mnist-loss (prediction trueth) ($cee prediction trueth))

(defun mnist-write-weight-to (w fname)
  (let ((f (file.disk fname "w")))
    ($fwrite ($data w) f)
    ($fclose f)))

(defun mnist-write-weights ()
  (mnist-write-weight-to *w1* "dlfs/mnist-w1.dat")
  (mnist-write-weight-to *b1* "dlfs/mnist-b1.dat")
  (mnist-write-weight-to *w2* "dlfs/mnist-w2.dat")
  (mnist-write-weight-to *b2* "dlfs/mnist-b2.dat")
  (mnist-write-weight-to *w3* "dlfs/mnist-w3.dat")
  (mnist-write-weight-to *b3* "dlfs/mnist-b3.dat"))

(defun mnist-read-weight-from (w fname)
  (let ((f (file.disk fname "r")))
    ($fread ($data w) f)
    ($fclose f)))

(defun mnist-read-weights ()
  (mnist-read-weight-from *w1* "dlfs/mnist-w1.dat")
  (mnist-read-weight-from *b1* "dlfs/mnist-b1.dat")
  (mnist-read-weight-from *w2* "dlfs/mnist-w2.dat")
  (mnist-read-weight-from *b2* "dlfs/mnist-b2.dat")
  (mnist-read-weight-from *w3* "dlfs/mnist-w3.dat")
  (mnist-read-weight-from *b3* "dlfs/mnist-b3.dat"))

;; write to file
(mnist-write-weights)

;; read from file
(mnist-read-weights)

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

(defun mnist-test-stat ()
  (let ((xt ($ *mnist* :test-images))
        (yt ($ *mnist* :test-labels)))
    ($count (loop :for i :from 0 :below ($size xt 0)
                  :for xi = ($index xt 0 (list i))
                  :for yi = ($index yt 0 (list i))
                  :for yi* = ($data (mnist-predict ($constant xi)))
                  :for err = ($sum ($abs ($sub ($round yi*) yi)))
                  :when (> err 0)
                    :collect i))))

;; compare sgd vs others
(let* ((x (-> *mnist*
              ($ :train-images)
              ($constant)))
       (y (-> *mnist*
              ($ :train-labels)
              ($constant)))
       (lr 0.01))
  (mnist-reset-parameters)
  (loop :for i :from 1 :to 10
        :for y* = (mnist-predict x)
        :for loss = (mnist-loss y* y)
        :do (progn
              (print (list i ($data loss)))
              (finish-output)
              ($bp! loss)
              ($gd! loss lr)
              (gcf)))
  (gcf))

(let* ((x (-> *mnist*
              ($ :train-images)
              ($constant)))
       (y (-> *mnist*
              ($ :train-labels)
              ($constant)))
       (lr 0.01)
       (a 0.9))
  (mnist-reset-parameters)
  (loop :for i :from 1 :to 10
        :for y* = (mnist-predict x)
        :for loss = (mnist-loss y* y)
        :do (progn
              (print (list i ($data loss)))
              (finish-output)
              ($bp! loss)
              ($mgd! loss lr a)
              (gcf)))
  (gcf))

(let* ((x (-> *mnist*
              ($ :train-images)
              ($constant)))
       (y (-> *mnist*
              ($ :train-labels)
              ($constant)))
       (lr 0.01))
  (mnist-reset-parameters)
  (loop :for i :from 1 :to 10
        :for y* = (mnist-predict x)
        :for loss = (mnist-loss y* y)
        :do (progn
              (print (list i ($data loss)))
              (finish-output)
              ($bp! loss)
              ($agd! loss lr)
              (gcf)))
  (gcf))
