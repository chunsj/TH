(defpackage :dlfs-06
  (:use #:common-lisp
        #:mu
        #:th
        #:th.db.mnist))

(in-package :dlfs-06)

;; mnist data loading - takes time, so load and set
(defparameter *mnist* (read-mnist-data))
(prn *mnist*)

(defparameter *mnist-train-image-batches*
  (loop :for i :from 0 :below 6
        :for rng = (loop :for k :from (* i 10000) :below (* (1+ i) 10000)
                         :collect k)
        :collect ($contiguous! ($index ($ *mnist* :train-images) 0 rng))))

(defparameter *mnist-train-label-batches*
  (loop :for i :from 0 :below 6
        :for rng = (loop :for k :from (* i 10000) :below (* (1+ i) 10000)
                         :collect k)
        :collect ($contiguous! ($index ($ *mnist* :train-labels) 0 rng))))

;; network parameters
(defparameter *w1* ($parameter (rndn 784 50)))
(defparameter *b1* ($parameter (zeros 50)))
(defparameter *w2* ($parameter (rndn 50 100)))
(defparameter *b2* ($parameter (zeros 100)))
(defparameter *w3* ($parameter (rndn 100 10)))
(defparameter *b3* ($parameter (zeros 10)))

(defun mnist-reset-parameters-xavier ()
  (setf *w1* ($parameter ($div (rnd 784 50) ($sqrt 784))))
  (setf *b1* ($parameter (zeros 50)))
  (setf *w2* ($parameter ($div (rnd 50 100) ($sqrt 50))))
  (setf *b2* ($parameter (zeros 100)))
  (setf *w3* ($parameter ($div (rnd 100 10) ($sqrt 100))))
  (setf *b3* ($parameter (zeros 10))))

(defun mnist-reset-parameters-he ()
  (setf *w1* ($parameter ($div (rnd 784 50) ($sqrt (/ 784 2)))))
  (setf *b1* ($parameter (zeros 50)))
  (setf *w2* ($parameter ($div (rnd 50 100) ($sqrt (/ 50 2)))))
  (setf *b2* ($parameter (zeros 100)))
  (setf *w3* ($parameter ($div (rnd 100 10) ($sqrt (/ 100 2)))))
  (setf *b3* ($parameter (zeros 10))))

(defun mnist-predict (x)
  (-> x
      ($xwpb *w1* *b1*)
      ($sigmoid)
      ($xwpb *w2* *b2*)
      ($sigmoid)
      ($xwpb *w3* *b3*)
      ($softmax)))

(defun mnist-predict-relu (x)
  (-> x
      ($xwpb *w1* *b1*)
      ($relu)
      ($xwpb *w2* *b2*)
      ($relu)
      ($xwpb *w3* *b3*)
      ($softmax)))

(defun mnist-predict-relu-do (x &optional (trainp t) (p 0.1))
  (-> x
      ($xwpb *w1* *b1*)
      ($relu)
      ($dropout trainp p)
      ($xwpb *w2* *b2*)
      ($relu)
      ($dropout trainp p)
      ($xwpb *w3* *b3*)
      ($softmax)))

(defparameter *g1* ($parameter (ones 50)))
(defparameter *e1* ($parameter (zeros 50)))
(defparameter *m1* (zeros 50))
(defparameter *v1* (ones 50))
(defparameter *g2* ($parameter (ones 100)))
(defparameter *e2* ($parameter (zeros 100)))
(defparameter *m2* (zeros 100))
(defparameter *v2* (ones 100))

(defun mnist-reset-parameters-bn ()
  (setf *w1* ($parameter ($div (rnd 784 50) ($sqrt (/ 784 2)))))
  (setf *b1* ($parameter (zeros 50)))
  (setf *w2* ($parameter ($div (rnd 50 100) ($sqrt (/ 50 2)))))
  (setf *b2* ($parameter (zeros 100)))
  (setf *w3* ($parameter ($div (rnd 100 10) ($sqrt (/ 100 2)))))
  (setf *b3* ($parameter (zeros 10)))
  (setf *g1* ($parameter (ones 50)))
  (setf *e1* ($parameter (zeros 50)))
  (setf *m1* (zeros 50))
  (setf *v1* (ones 50))
  (setf *g2* ($parameter (ones 100)))
  (setf *e2* ($parameter (zeros 100)))
  (setf *m2* (zeros 100))
  (setf *v2* (ones 100)))

(defun prn-and-pass (x)
  (prn x)
  x)

(defun mnist-predict-bn (x &optional (trainp t))
  (-> x
      ($xwpb *w1* *b1*)
      ($bnorm *g1* *e1* *m1* *v1* trainp)
      ($relu)
      ($xwpb *w2* *b2*)
      ($bnorm *g2* *e2* *m2* *v2* trainp)
      ($relu)
      ($xwpb *w3* *b3*)
      ($softmax)))

(defun mnist-predict-bns (x &optional (trainp t))
  (-> x
      ($xwpb *w1* *b1*)
      ($bnorm nil nil *m1* *v1* trainp)
      ($relu)
      ($xwpb *w2* *b2*)
      ($bnorm nil nil *m2* *v2* trainp)
      ($relu)
      ($xwpb *w3* *b3*)
      ($softmax)))

(defun mnist-loss (prediction truth) ($cee prediction truth))

(defun mnist-loss-wr (prediction truth &optional (l 0.1))
  ($+ (mnist-loss prediction truth)
      ($* ($+ ($sum ($* *w1* *w1*))
              ($sum ($* *b1* *b1*))
              ($sum ($* *w2* *w2*))
              ($sum ($* *b2* *b2*))
              ($sum ($* *w3* *w3*))
              ($sum ($* *b3* *b3*)))
          (/ l (+ ($count *w1*)
                  ($count *b1*)
                  ($count *w2*)
                  ($count *b2*)
                  ($count *w3*)
                  ($count *b3*))))))

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

;; code test
(let ((xt ($ *mnist* :test-images))
      (yt ($ *mnist* :test-labels)))
  (mnist-reset-parameters-xavier)
  (prn ($count (loop :for i :from 0 :below ($size xt 0)
                     :for xi = ($index xt 0 (list i))
                     :for yi = ($index yt 0 (list i))
                     :for yi* = ($data (mnist-predict xi))
                     :for err = ($sum ($abs ($sub ($round yi*) yi)))
                     :when (> err 0)
                       :collect i))))

(let ((xt ($ *mnist* :test-images))
      (yt ($ *mnist* :test-labels)))
  (mnist-reset-parameters-he)
  (prn ($count (loop :for i :from 0 :below ($size xt 0)
                     :for xi = ($index xt 0 (list i))
                     :for yi = ($index yt 0 (list i))
                     :for yi* = ($data (mnist-predict-relu xi))
                     :for err = ($sum ($abs ($sub ($round yi*) yi)))
                     :when (> err 0)
                       :collect i))))

(let ((xt ($ *mnist* :test-images))
      (yt ($ *mnist* :test-labels)))
  (mnist-reset-parameters-bn)
  (prn ($count (loop :for i :from 0 :below ($size xt 0)
                     :for xi = ($index xt 0 (list i))
                     :for yi = ($index yt 0 (list i))
                     :for yi* = ($data (mnist-predict-bn xi))
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

(defun mnist-test-stat-relu ()
  (let ((xt ($ *mnist* :test-images))
        (yt ($ *mnist* :test-labels)))
    ($count (loop :for i :from 0 :below ($size xt 0)
                  :for xi = ($index xt 0 (list i))
                  :for yi = ($index yt 0 (list i))
                  :for yi* = ($data (mnist-predict-relu xi))
                  :for err = ($sum ($abs ($sub ($round yi*) yi)))
                  :when (> err 0)
                    :collect i))))

;; compare sgd vs others
(let* ((x (-> *mnist*
              ($ :train-images)))
       (y (-> *mnist*
              ($ :train-labels)))
       (lr 0.01))
  (mnist-reset-parameters-xavier)
  (loop :for i :from 1 :to 10
        :for y* = (mnist-predict x)
        :for loss = (mnist-loss y* y)
        :do (progn
              (prn (list i ($data loss)))
              ($gs! loss)
              ($gd! (list *w1* *b1* *w2* *b2* *w3* *b3*) lr))))

(let* ((x (-> *mnist*
              ($ :train-images)))
       (y (-> *mnist*
              ($ :train-labels)))
       (lr 0.01)
       (a 0.9))
  (mnist-reset-parameters-xavier)
  (loop :for i :from 1 :to 10
        :for y* = (mnist-predict x)
        :for loss = (mnist-loss y* y)
        :do (progn
              (prn (list i ($data loss)))
              ($gs! loss)
              ($mgd! (list *w1* *b1* *w2* *b2* *w3* *b3*) lr a))))

(let* ((x (-> *mnist*
              ($ :train-images)))
       (y (-> *mnist*
              ($ :train-labels)))
       (lr 0.01))
  (mnist-reset-parameters-xavier)
  (loop :for i :from 1 :to 10
        :for y* = (mnist-predict x)
        :for loss = (mnist-loss y* y)
        :do (progn
              (prn (list i ($data loss)))
              ($gs! loss)
              ($agd! (list *w1* *b1* *w2* *b2* *w3* *b3*) lr))))

;; relu model comparison
(let* ((x (-> *mnist*
              ($ :train-images)))
       (y (-> *mnist*
              ($ :train-labels)))
       (lr 0.01))
  (mnist-reset-parameters-he)
  (loop :for i :from 1 :to 10
        :for y* = (mnist-predict-relu x)
        :for loss = (mnist-loss y* y)
        :do (progn
              (prn (list i ($data loss)))
              ($gs! loss)
              ($gd! (list *w1* *b1* *w2* *b2* *w3* *b3*) lr))))

(let* ((x (-> *mnist*
              ($ :train-images)))
       (y (-> *mnist*
              ($ :train-labels)))
       (lr 0.01)
       (a 0.9))
  (mnist-reset-parameters-he)
  (loop :for i :from 1 :to 10
        :for y* = (mnist-predict-relu x)
        :for loss = (mnist-loss y* y)
        :do (progn
              (prn (list i ($data loss)))
              ($gs! loss)
              ($mgd! (list *w1* *b1* *w2* *b2* *w3* *b3*) lr a))))

(let* ((x (-> *mnist*
              ($ :train-images)))
       (y (-> *mnist*
              ($ :train-labels)))
       (lr 0.01))
  (mnist-reset-parameters-he)
  (loop :for i :from 1 :to 10
        :for y* = (mnist-predict-relu x)
        :for loss = (mnist-loss y* y)
        :do (progn
              (prn (list i ($data loss)))
              ($gs! loss)
              ($agd! (list *w1* *b1* *w2* *b2* *w3* *b3*) lr))))

;; dropout
(let* ((x (-> *mnist*
              ($ :train-images)))
       (y (-> *mnist*
              ($ :train-labels)))
       (lr 0.01)
       (p 0.4))
  (mnist-reset-parameters-he)
  (loop :for i :from 1 :to 10
        :for y* = (mnist-predict-relu-do x t p)
        :for loss = (mnist-loss y* y)
        :do (progn
              (prn (list i ($data loss)))
              ($gs! loss)
              ($agd! (list *w1* *b1* *w2* *b2* *w3* *b3*) lr))))

(let ((xt ($ *mnist* :test-images))
      (yt ($ *mnist* :test-labels))
      (p 0.4))
  (prn ($count (loop :for i :from 0 :below ($size xt 0)
                     :for xi = ($index xt 0 (list i))
                     :for yi = ($index yt 0 (list i))
                     :for yi* = ($data (mnist-predict-relu-do xi nil p))
                     :for err = ($sum ($abs ($sub ($round yi*) yi)))
                     :when (> err 0)
                       :collect i))))

(let ((xt ($ *mnist* :test-images))
      (yt ($ *mnist* :test-labels)))
  (prn ($count (loop :for i :from 0 :below ($size xt 0)
                     :for xi = ($index xt 0 (list i))
                     :for yi = ($index yt 0 (list i))
                     :for yi* = ($data (mnist-predict-relu xi))
                     :for err = ($sum ($abs ($sub ($round yi*) yi)))
                     :when (> err 0)
                       :collect i))))

;; without weight regularization
(let* ((x (-> *mnist*
              ($ :train-images)))
       (y (-> *mnist*
              ($ :train-labels)))
       (lr 0.01))
  (mnist-reset-parameters-he)
  (loop :for i :from 1 :to 10
        :for y* = (mnist-predict-relu x)
        :for loss = (mnist-loss y* y)
        :do (progn
              (prn (list i ($data loss)))
              ($gs! loss)
              ($agd! (list *w1* *b1* *w2* *b2* *w3* *b3*) lr))))

;; with weight regularization - does not work in 50 step
(let* ((x (-> *mnist*
              ($ :train-images)))
       (y (-> *mnist*
              ($ :train-labels)))
       (lr 0.01)
       (l 0.1))
  (mnist-reset-parameters-he)
  (loop :for i :from 1 :to 100
        :for y* = (mnist-predict-relu x)
        :for loss = (mnist-loss-wr y* y l)
        :do (progn
              (prn (list i ($data loss)))
              ($gs! loss)
              ($agd! (list *w1* *b1* *w2* *b2* *w3* *b3*) lr))))

;; test result
(let ((xt ($ *mnist* :test-images))
      (yt ($ *mnist* :test-labels)))
  (prn ($count (loop :for i :from 0 :below ($size xt 0)
                     :for xi = ($index xt 0 (list i))
                     :for yi = ($index yt 0 (list i))
                     :for yi* = ($data (mnist-predict-relu ($constant xi)))
                     :for err = ($sum ($abs ($sub ($round yi*) yi)))
                     :when (> err 0)
                       :collect i))))

;; without batch normalization
(let* ((x (-> *mnist*
              ($ :train-images)))
       (y (-> *mnist*
              ($ :train-labels)))
       (lr 0.01))
  (mnist-reset-parameters-he)
  (loop :for i :from 1 :to 50
        :for y* = (mnist-predict-relu x)
        :for loss = (mnist-loss y* y)
        :do (progn
              (prn (list i ($data loss)))
              ($gs! loss)
              ($agd! (list *w1* *b1* *w2* *b2* *w3* *b3*) lr))))

;; test result
(let ((xt ($ *mnist* :test-images))
      (yt ($ *mnist* :test-labels)))
  (prn ($count (loop :for i :from 0 :below ($size xt 0)
                     :for xi = ($index xt 0 (list i))
                     :for yi = ($index yt 0 (list i))
                     :for yi* = ($data (mnist-predict-relu xi))
                     :for err = ($sum ($abs ($sub ($round yi*) yi)))
                     :when (> err 0)
                       :collect i))))

;; batch normalization - does not converge
(let* ((x (-> *mnist*
              ($ :train-images)))
       (y (-> *mnist*
              ($ :train-labels)))
       (lr 0.01))
  (mnist-reset-parameters-bn)
  (loop :for i :from 1 :to 40
        :for y* = (mnist-predict-bn x)
        :for loss = (mnist-loss y* y)
        :do (progn
              (prn (list i ($data loss)))
              ($gs! loss)
              ($agd! (list *w1* *b1* *w2* *b2* *w3* *b3* *g1* *e1* *g2* *m2*) lr))))

;; test result
(let ((xt ($ *mnist* :test-images))
      (yt ($ *mnist* :test-labels)))
  (prn ($count (loop :for i :from 0 :below ($size xt 0)
                     :for xi = ($index xt 0 (list i))
                     :for yi = ($index yt 0 (list i))
                     :for yi* = ($data (mnist-predict-bn xi nil))
                     :for err = ($sum ($abs ($sub ($round yi*) yi)))
                     :when (> err 0)
                       :collect i))))
