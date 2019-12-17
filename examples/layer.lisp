(defpackage :layer-example
  (:use #:common-lisp
        #:mu
        #:th
        #:th.db.mnist
        #:th.layers))

(in-package :layer-example)

;; this example will use layers to re-create the mnist classification
;; example using 2D convolutional network.
;; personally, i do not like this kind of approach/design of layers,
;; however, i cannot find better way yet.

;; load mnist data, takes ~22 secs in macbook 2017
(defparameter *mnist* (read-mnist-data))

;; mnist data has following dataset
;; train-images, train-labels and test-images, test-labels
(prn *mnist*)

;; prepare data
(defparameter *batch-size* 500)
(defparameter *batch-count* (/ ($size ($ *mnist* :train-images) 0) *batch-size*))
(defparameter *channel-number* 1)
(defparameter *image-width* 28)
(defparameter *image-height* 28)

;; training data - uses batches for performance
(defparameter *mnist-train-image-batches*
  (loop :for i :from 0 :below *batch-count*
        :for rng = (loop :for k :from (* i *batch-size*) :below (* (1+ i) *batch-size*)
                         :collect k)
        :collect (-> ($contiguous! ($index ($ *mnist* :train-images) 0 rng))
                     ($reshape *batch-size* *channel-number* *image-height* *image-width*))))

(defparameter *mnist-train-label-batches*
  (loop :for i :from 0 :below *batch-count*
        :for rng = (loop :for k :from (* i *batch-size*) :below (* (1+ i) *batch-size*)
                         :collect k)
        :collect ($contiguous! ($index ($ *mnist* :train-labels) 0 rng))))

(defparameter *mnist-test-images*
  (let ((xt ($ *mnist* :test-images)))
    ($reshape xt ($size xt 0) *channel-number* *image-width* *image-height*)))

(defparameter *mnist-test-labels* ($ *mnist* :test-labels))

;; data is set up so free the original data
(setf *mnist* nil)

;; network parameters - copied from mnist example
(defparameter *filter-number* 30)
(defparameter *filter-width* 5)
(defparameter *filter-height* 5)
(defparameter *pool-width* 2)
(defparameter *pool-height* 2)
(defparameter *pool-stride-width* 2)
(defparameter *pool-stride-height* 2)
(defparameter *pool-out-width* 12)
(defparameter *pool-out-height* 12)
(defparameter *l2-output* 100)
(defparameter *l3-output* 10)

(defparameter *network* (sequence-layer
                         (convolution-2d-layer *channel-number*
                                               *filter-number*
                                               *filter-width*
                                               *filter-height*
                                               :activation :relu
                                               :batch-normalization-p t)
                         (maxpool-2d-layer *pool-width* *pool-height*
                                           :stride-width *pool-stride-width*
                                           :stride-height *pool-stride-height*)
                         (flatten-layer)
                         (affine-layer (* *filter-number* *pool-out-width* *pool-out-height*)
                                       *l2-output*
                                       :activation :relu
                                       :batch-normalization-p t)
                         (affine-layer *l2-output* *l3-output*
                                       :activation :softmax)))

(defun mnist-predict (x &optional (trainp t)) ($execute *network* x :trainp trainp))

(defun mnist-test-stat (&optional verbose)
  (let* ((yt (-> *mnist-test-labels*
                 (tensor.byte)))
         (yt* (-> (mnist-predict *mnist-test-images* nil)
                  ($round)
                  (tensor.byte)))
         (errors ($ne ($sum ($eq yt* yt) 1)
                      (-> (tensor.byte ($size yt 0) 1)
                          ($fill! 10)))))
    (when verbose (loop :for i :from 0 :below ($size errors 0)
                        :do (when (eq 1 ($ errors i 0))
                              (prn i))))
    ($sum errors)))

($load-weights "./examples/weights/layers-mnist" *network*)
(with-foreign-memory-limit ()
  (mnist-test-stat))

;; if you want to train again, run following code
(defparameter *epoch* 30)

($reset! *network*)
(time
 (with-foreign-memory-limit ()
   (loop :for epoch :from 1 :to *epoch*
         :do (loop :for i :from 0 :below *batch-count*
                   :for x = ($ *mnist-train-image-batches* i)
                   :for y = ($ *mnist-train-label-batches* i)
                   :for y* = (mnist-predict x)
                   :for loss = ($cee y* y)
                   :do (progn
                         (when (zerop (rem i 10))
                           (prn (format nil "[~A|~A]: ~A" (1+ i) epoch ($data loss))))
                         ($adgd! *network*))))))

($save-weights "./examples/weights/layers-mnist" *network*)
