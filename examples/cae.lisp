(defpackage :cae-example
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers
        #:th.db.mnist))

(in-package :cae-example)

(defparameter *mnist* (read-mnist-data))

(defparameter *batch-size* 32)
(defparameter *max-batch-count* 100)
(defparameter *batch-count* (min *max-batch-count*
                                 (/ ($size ($ *mnist* :train-images) 0) *batch-size*)))

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
                                               :stride-width 2 :stride-height 2
                                               :activation :relu)
                         (convolution-2d-layer 32 64 3 3
                                               :padding-width 1 :padding-height 1
                                               :stride-width 2 :stride-height 2
                                               :activation :relu)
                         (flatten-layer)
                         (affine-layer 3136 2
                                       :batch-normalization-p t
                                       :activation :nil)))

(defparameter *decoder* (sequential-layer
                         (affine-layer 2 3136 :activation :relu)
                         (reshape-layer 64 7 7)
                         (full-convolution-2d-layer 64 64 3 3
                                                    :padding-width 1 :padding-height 1
                                                    :stride-width 2 :stride-height 2
                                                    :adjust-width 1 :adjust-height 1
                                                    :activation :relu)
                         (full-convolution-2d-layer 64 32 3 3
                                                    :stride-width 2 :stride-height 2
                                                    :padding-width 1 :padding-height 1
                                                    :adjust-width 1 :adjust-height 1
                                                    :activation :relu)
                         (full-convolution-2d-layer 32 1 3 3
                                                    :padding-width 1 :padding-height 1
                                                    :batch-normalization-p t
                                                    :activation :sigmoid)))

(defparameter *model* (sequential-layer *encoder* *decoder*))

(defun loss (y x) ($bce y x))

(defun update-params (model gd)
  (cond ((eq gd :adam) ($amgd! model 1E-3))
        ((eq gd :rmsprop) ($rmgd! model))
        (t ($adgd! model))))

(defun train (model xs epoch idx gd)
  (let* ((pstep 20)
         (ys ($execute model xs))
         (l (loss ys xs)))
    (when (zerop (rem idx pstep))
      (prn (format nil "~5,D" idx) "/" (format nil "~5,D" epoch) ":" ($data l)))
    (update-params model gd)))

(defparameter *epochs* 100)

($reset! *model*)

(time
 (with-foreign-memory-limit ()
   (loop :for epoch :from 1 :to *epochs*
         :do (loop :for xs :in *mnist-train-image-batches*
                   :for idx :from 1
                   :do (train *model* xs epoch idx :rmsprop)))))

;; check results
(defun compare-xy (encoder decoder bs)
  (let* ((xs ($ bs (random ($count bs))))
         (bn ($size xs 0))
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

(compare-xy *encoder* *decoder* *mnist-train-image-batches*)

;; test model
($execute *model* (car *mnist-train-image-batches*) :trainp nil)

;; save/load trained weights
($save-weights "examples/weights/cae" *model*)
($load-weights "examples/weights/cae" *model*)
