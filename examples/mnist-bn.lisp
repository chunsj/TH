(defpackage :mnist-bn
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers
        #:th.db.mnist
        #:th.db.fashion))

(in-package :mnist-bn)

;; use one of following
(defparameter *mnist* (read-mnist-data))
(defparameter *mnist* (read-fashion-data))

(defparameter *data-size* 1000)

(defparameter *x-train* ($index ($ *mnist* :train-images) 0 (xrange 0 *data-size*)))
(defparameter *y-train* ($index ($ *mnist* :train-labels) 0 (xrange 0 *data-size*)))

(defparameter *batch-size* 100)
(defparameter *batch-count* (/ *data-size* *batch-size*))

(defparameter *x-batches*
  (loop :for i :from 0 :below *batch-count*
        :for rng = (loop :for k :from (* i *batch-size*) :below (* (1+ i) *batch-size*)
                         :collect k)
        :collect ($contiguous! ($index *x-train* 0 rng))))
(defparameter *y-batches*
  (loop :for i :from 0 :below *batch-count*
        :for rng = (loop :for k :from (* i *batch-size*) :below (* (1+ i) *batch-size*)
                         :collect k)
        :collect ($contiguous! ($index *y-train* 0 rng))))

(defparameter *input-size* 784)
(defparameter *weight-size* 100)
(defparameter *output-size* 10)

(defparameter *net01* (sequence-layer
                       (affine-layer *input-size* *weight-size*
                                     :activation :relu
                                     :weight-initializer :he-normal)
                       (affine-layer *weight-size* *weight-size*
                                     :activation :relu
                                     :weight-initializer :he-normal)
                       (affine-layer *weight-size* *output-size*
                                     :activation :softmax
                                     :weight-initializer :he-normal)))

(defparameter *net02* (sequence-layer
                       (affine-layer *input-size* *weight-size*
                                     :activation :relu
                                     :weight-initializer :he-normal
                                     :batch-normalization-p t)
                       (affine-layer *weight-size* *weight-size*
                                     :activation :relu
                                     :weight-initializer :he-normal
                                     :batch-normalization-p t)
                       (affine-layer *weight-size* *output-size*
                                     :activation :softmax
                                     :weight-initializer :he-normal)))

(defparameter *net03* (sequence-layer
                       (affine-layer *input-size* *weight-size*
                                     :activation :selu
                                     :weight-initializer :he-normal)
                       (affine-layer *weight-size* *weight-size*
                                     :activation :selu
                                     :weight-initializer :he-normal)
                       (affine-layer *weight-size* *output-size*
                                     :activation :softmax
                                     :weight-initializer :he-normal)))

(defparameter *net04* (sequence-layer
                       (affine-layer *input-size* *weight-size*
                                     :activation :selu
                                     :weight-initializer :selu-normal)
                       (affine-layer *weight-size* *weight-size*
                                     :activation :selu
                                     :weight-initializer :selu-normal)
                       (affine-layer *weight-size* *output-size*
                                     :activation :softmax
                                     :weight-initializer :he-normal)))

(defparameter *net05* (sequence-layer
                       (affine-layer *input-size* *weight-size*
                                     :activation :swish
                                     :weight-initializer :he-normal)
                       (affine-layer *weight-size* *weight-size*
                                     :activation :swish
                                     :weight-initializer :he-normal)
                       (affine-layer *weight-size* *output-size*
                                     :activation :softmax
                                     :weight-initializer :he-normal)))

(defparameter *net06* (sequence-layer
                       (affine-layer *input-size* *weight-size*
                                     :activation :mish
                                     :weight-initializer :he-normal)
                       (affine-layer *weight-size* *weight-size*
                                     :activation :mish
                                     :weight-initializer :he-normal)
                       (affine-layer *weight-size* *output-size*
                                     :activation :softmax
                                     :weight-initializer :he-normal)))

(defparameter *epochs* 500)

(defun train (net)
  (let ((losses nil))
    (loop :for epoch :from 1 :to *epochs*
          :do (loop :for xb :in *x-batches*
                    :for yb :in *y-batches*
                    :for i :from 0
                    :for y* = ($execute net xb)
                    :for l = ($cee y* yb)
                    :do (progn
                          ($adgd! net)
                          (when (and (zerop (rem epoch 50))
                                     (zerop i))
                            (let ((lv ($data l)))
                              (push lv losses)
                              (prn (format nil "[~A] ~2,5E" epoch lv)))))))
    losses))

(defparameter *losses01* (train *net01*))
(defparameter *losses02* (train *net02*))
(defparameter *losses03* (train *net03*))
(defparameter *losses04* (train *net04*))
(defparameter *losses05* (train *net05*))
(defparameter *losses06* (train *net06*))
