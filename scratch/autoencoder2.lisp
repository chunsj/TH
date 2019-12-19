(defpackage :autoencoder2
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers
        #:th.db.mnist))

(in-package :autoencoder2)

(defparameter *mnist* (read-mnist-data))

(defparameter *batch-size* 32)
(defparameter *batch-count* (/ ($size ($ *mnist* :train-images) 0) *batch-size*))

(defparameter *mnist-train-image-batches*
  (loop :for i :from 0 :below *batch-count*
        :for rng = (loop :for k :from (* i *batch-size*) :below (* (1+ i) *batch-size*)
                         :collect k)
        :for xs = ($index ($ *mnist* :train-images) 0 rng)
        :collect ($contiguous! ($reshape xs ($size xs 0) 1 28 28))))

(setf *mnist* nil)

;; define autoencoder = encoder + decoder
(defparameter *encoder* (sequence-layer
                         (convolution-2d-layer 1 32 3 3
                                               :padding-width 1 :padding-height 1
                                               :activation :lrelu)
                         (convolution-2d-layer 32 64 3 3
                                               :stride-width 2 :stride-height 2
                                               :padding-width 1 :padding-height 1
                                               :activation :lrelu)
                         (convolution-2d-layer 64 64 3 3
                                               :stride-width 2 :stride-height 2
                                               :padding-width 1 :padding-height 1
                                               :activation :lrelu)
                         (convolution-2d-layer 64 64 3 3
                                               :padding-width 1 :padding-height 1
                                               :activation :lrelu)
                         (flatten-layer)
                         (affine-layer 3136 2)))

(defparameter *decoder* (sequence-layer
                         (affine-layer 2 3136)
                         (reshape-layer 64 7 7)
                         (full-convolution-2d-layer 64 64 3 3
                                                    :padding-width 1 :padding-height 1
                                                    :activation :lrelu)
                         (full-convolution-2d-layer 64 64 3 3
                                                    :stride-width 2 :stride-height 2
                                                    :padding-width 1 :padding-height 1
                                                    :adjust-width 1 :adjust-height 1
                                                    :activation :lrelu)
                         (full-convolution-2d-layer 64 32 3 3
                                                    :stride-width 2 :stride-height 2
                                                    :padding-width 1 :padding-height 1
                                                    :adjust-width 1 :adjust-height 1
                                                    :activation :lrelu)
                         (full-convolution-2d-layer 32 1 3 3
                                                    :padding-width 1 :padding-height 1
                                                    :activation :sigmoid)))

(defparameter *model* (sequence-layer *encoder* *decoder*))

;; test model
($execute *model* (car *mnist-train-image-batches*) :trainp nil)

(defun loss (y x)
  (let ((d ($- y x)))
    ($/ ($dot d d) ($size y 0))))

(defparameter *epochs* 90)

($reset! *model*)
(time
 (with-foreign-memory-limit ()
   (loop :for epoch :from 0 :below *epochs*
         :do (loop :for xs :in *mnist-train-image-batches*
                   :for idx :from 1
                   :do (let* ((ys ($execute *model* xs))
                              (l (loss ys xs)))
                         (when (zerop (rem idx 20))
                           (prn "LOSS[" idx "/" epoch "]" ($data l)))
                         ($amgd! *model* 0.005))))))
