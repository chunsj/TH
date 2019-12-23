(defpackage :convolutional-autoencoder-example
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers
        #:th.db.mnist))

(in-package :convolutional-autoencoder-example)

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
(defparameter *encoder* (sequential-layer
                         (convolution-2d-layer 1 32 3 3
                                               :padding-width 1 :padding-height 1
                                               :batch-normalization-p t
                                               :activation :lrelu)
                         (convolution-2d-layer 32 64 3 3
                                               :stride-width 2 :stride-height 2
                                               :padding-width 1 :padding-height 1
                                               :batch-normalization-p t
                                               :activation :lrelu)
                         (convolution-2d-layer 64 64 3 3
                                               :stride-width 2 :stride-height 2
                                               :padding-width 1 :padding-height 1
                                               :batch-normalization-p t
                                               :activation :lrelu)
                         (convolution-2d-layer 64 64 3 3
                                               :padding-width 1 :padding-height 1
                                               :batch-normalization-p t
                                               :activation :lrelu)
                         (flatten-layer)
                         (affine-layer 3136 2 :activation :nil)))

(defparameter *decoder* (sequential-layer
                         (affine-layer 2 3136 :activation :nil)
                         (reshape-layer 64 7 7)
                         (full-convolution-2d-layer 64 64 3 3
                                                    :padding-width 1 :padding-height 1
                                                    :batch-normalization-p t
                                                    :activation :lrelu)
                         (full-convolution-2d-layer 64 64 3 3
                                                    :stride-width 2 :stride-height 2
                                                    :padding-width 1 :padding-height 1
                                                    :adjust-width 1 :adjust-height 1
                                                    :batch-normalization-p t
                                                    :activation :lrelu)
                         (full-convolution-2d-layer 64 32 3 3
                                                    :stride-width 2 :stride-height 2
                                                    :padding-width 1 :padding-height 1
                                                    :adjust-width 1 :adjust-height 1
                                                    :batch-normalization-p t
                                                    :activation :lrelu)
                         (full-convolution-2d-layer 32 1 3 3
                                                    :padding-width 1 :padding-height 1
                                                    :batch-normalization-p t
                                                    :activation :sigmoid)))

(defparameter *model* (sequential-layer *encoder* *decoder*))

;; test model
($execute *model* (car *mnist-train-image-batches*) :trainp nil)

(defun loss (y x) ($bce y x))

(defun train (model xs epoch idx)
  (let* ((pstep 50)
         (ys ($execute model xs))
         (l (loss ys xs)))
    (when (zerop (rem idx pstep))
      (prn (format nil "~5,D" idx) "/" (format nil "~5,D" epoch) ":" ($data l)))
    ;;($adgd! *model*)
    ($amgd! *model* 1E-3)))

(defparameter *epochs* 4)

($reset! *model*)
(time
 (with-foreign-memory-limit ()
   (loop :for epoch :from 0 :below *epochs*
         :do (loop :for xs :in *mnist-train-image-batches*
                   :for idx :from 1
                   :do (train *model* xs epoch idx)))))

;; XXX for analysis
(defun compare-xy (encoder decoder xs)
  (let* ((bn ($size xs 0))
         (es ($execute encoder xs :trainp nil))
         (ds ($execute decoder es :trainp nil))
         (ys ($reshape! ds bn 1 28 28))
         (idx (random bn))
         (x ($ xs idx))
         (y ($ ys idx))
         (inf "/Users/Sungjin/Desktop/input.png")
         (ouf "/Users/Sungjin/Desktop/output.png"))
    (prn "ENCODED:" es)
    (prn "INDEX:" idx)
    (th.image:write-tensor-png-file x inf)
    (th.image:write-tensor-png-file y ouf)))

(compare-xy *encoder* *decoder* ($0 *mnist-train-image-batches*))

($save-weights "examples/weights/cae" *model*)
($load-weights "examples/weights/cae" *model*)
