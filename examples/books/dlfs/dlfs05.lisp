(defpackage :dlfs-05
  (:use #:common-lisp
        #:mu
        #:th
        #:th.db.mnist))

(in-package :dlfs-05)

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
  (prn ($count (loop :for i :from 0 :below ($size xt 0)
                     :for xi = ($index xt 0 (list i))
                     :for yi = ($index yt 0 (list i))
                     :for yi* = ($data (mnist-predict xi))
                     :for err = ($sum ($abs ($sub ($round yi*) yi)))
                     :when (> err 0)
                       :collect i))))

(defun mnist-test-stat ()
  (let ((xt ($ *mnist* :test-images))
        (yt ($ *mnist* :test-labels)))
    ($count (loop :for i :from 0 :below ($size xt 0)
                  :for xi = ($index xt 0 (list i))
                  :for yi = ($index yt 0 (list i))
                  :for yi* = ($data (mnist-predict xi))
                  :for err = ($sum ($abs ($sub ($round yi*) yi)))
                  :when (> err 0)
                    :collect i))))

;; full training
(let* ((x (-> *mnist*
              ($ :train-images)))
       (y (-> *mnist*
              ($ :train-labels)))
       (lr 1.4)
       (pwrcnt 526))
  (loop :for i :from 1 :to 1000
        :for y* = (mnist-predict x)
        :for loss = (mnist-loss y* y)
        :do (progn
              (when (zerop (mod i 5))
                (prn (list i ($data loss)))
                (finish-output))
              ($gs! loss)
              ($gd! (list *w1* *b1* *w2* *b2* *w3* *b3*) lr)
              (when (zerop (mod i 50))
                (let ((wrcnt (mnist-test-stat)))
                  (prn (list i wrcnt 10000))
                  (when (< wrcnt pwrcnt)
                    (setf pwrcnt wrcnt)
                    (prn "Saving weights...")
                    (mnist-write-weights)
                    (prn "Done saving."))))))
  (prn (mnist-test-stat)))
