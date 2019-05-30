(defpackage :gdl-ch08
  (:use #:common-lisp
        #:mu
        #:th
        #:th.db.mnist))

(in-package :gdl-ch08)

;; use smaller dataset for speed
(defun smaller-mnist-data ()
  (let ((train-images (tensor))
        (train-labels (tensor))
        (test-images (tensor))
        (test-labels (tensor)))
    (let ((f (file.disk "./gdl/mnist/mnist-small-train-images.tensor" "r")))
      ($fread train-images f)
      ($fclose f))
    (let ((f (file.disk "./gdl/mnist/mnist-small-train-labels.tensor" "r")))
      ($fread train-labels f)
      ($fclose f))
    (let ((f (file.disk "./gdl/mnist/mnist-small-test-images.tensor" "r")))
      ($fread test-images f)
      ($fclose f))
    (let ((f (file.disk "./gdl/mnist/mnist-small-test-labels.tensor" "r")))
      ($fread test-labels f)
      ($fclose f))
    #{:train-images train-images
      :train-labels train-labels
      :test-images test-images
      :test-labels test-labels}))

(defparameter *mnist* (smaller-mnist-data))

(defparameter *pixels-per-image* 784)
(defparameter *hidden-size* 40)
(defparameter *num-labels* 10)

(defparameter *alpha* 0.005)
(defparameter *iterations* 300)

(defparameter *w01* ($parameter ($- ($* 0.2 (rnd *pixels-per-image* *hidden-size*)) 0.1)))
(defparameter *w12* ($parameter ($- ($* 0.2 (rnd *hidden-size* *num-labels*)) 0.1)))

(defun mnist-predict (x)
  (-> x
      ($@ *w01*)
      ($relu)
      ($@ *w12*)
      ($softmax)))

(defun mnist-loss (y* y) ($cee y* y))

;; test functions
(prn (mnist-predict ($index ($ *mnist* :train-images) 0 '(0))))
(prn (mnist-loss (mnist-predict ($index ($ *mnist* :train-images) 0 '(0)))
                 ($index ($ *mnist* :train-labels) 0 '(0))))

(defparameter *mnist-train-images* ($ *mnist* :train-images))
(defparameter *mnist-train-labels* ($ *mnist* :train-labels))

(defun amax (x &optional (dimension 0))
  (let ((vals (tensor))
        (indices (tensor.long)))
    ($max! vals indices x dimension)
    indices))

(loop :for n :from 1 :to 100
      :do (let ((ndata ($size *mnist-train-images* 0)))
            (loop :for i :from 0 :below ndata
                  :for x = ($index *mnist-train-images* 0 (list i))
                  :for y = ($index *mnist-train-labels* 0 (list i))
                  :for y* = (mnist-predict x)
                  :for l = (mnist-loss y* y)
                  :do (progn
                        ($gs! l)
                        ($gd! (list *w01* *w12*) *alpha*)))
            (when (zerop (rem n 1))
              (let* ((indices (loop :for k :from 0 :below ndata :collect k))
                     (predictions (mnist-predict ($index *mnist-train-images* 0 indices)))
                     (truevals ($index *mnist-train-labels* 0 indices)))
                (prn n ($data (mnist-loss predictions truevals)))
                (prn "missed:" ($sum ($ne (amax ($data predictions) 1) (amax truevals 1))))))))

(let* ((indices (loop :for k :from 0 :below 100 :collect k))
       (ps (mnist-predict ($index ($ *mnist* :test-images) 0 indices)))
       (cs ($index ($ *mnist* :test-labels) 0 indices)))
  (prn ($sum ($ne (amax ($data ps) 1) (amax cs 1)))))

;; with dropout
(defun mnist-predict-do (x &optional trainp)
  (-> x
      ($@ *w01*)
      ($relu)
      ($dropout trainp 0.2)
      ($@ *w12*)
      ($softmax)))

(prn (mnist-predict-do ($index ($ *mnist* :train-images) 0 '(0)) t))
(prn (mnist-loss (mnist-predict-do ($index ($ *mnist* :train-images) 0 '(0)) t)
                 ($index ($ *mnist* :train-labels) 0 '(0))))

(defparameter *w01* ($parameter ($- ($* 0.2 (rnd *pixels-per-image* *hidden-size*)) 0.1)))
(defparameter *w12* ($parameter ($- ($* 0.2 (rnd *hidden-size* *num-labels*)) 0.1)))

(loop :for n :from 1 :to 100
      :do (let ((ndata ($size *mnist-train-images* 0)))
            (loop :for i :from 0 :below ndata
                  :for x = ($index *mnist-train-images* 0 (list i))
                  :for y = ($index *mnist-train-labels* 0 (list i))
                  :for y* = (mnist-predict-do x t)
                  :for l = (mnist-loss y* y)
                  :do (progn
                        ($gs! l)
                        ($gd! (list *w01* *w12*) *alpha*)))
            (when (zerop (rem n 1))
              (let* ((indices (loop :for k :from 0 :below ndata :collect k))
                     (predictions (mnist-predict-do ($index *mnist-train-images* 0 indices)))
                     (truevals ($index *mnist-train-labels* 0 indices)))
                (prn n ($data (mnist-loss predictions truevals)))
                (prn "missed:" ($sum ($ne (amax ($data predictions) 1) (amax truevals 1))))))))

(let* ((indices (loop :for k :from 0 :below 100 :collect k))
       (ps (mnist-predict-do ($index ($ *mnist* :test-images) 0 indices)))
       (cs ($index ($ *mnist* :test-labels) 0 indices)))
  (prn ($sum ($ne (amax ($data ps) 1) (amax cs 1)))))
