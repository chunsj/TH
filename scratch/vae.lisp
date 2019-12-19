(defpackage :vae-example
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers
        #:th.db.mnist))

(in-package :vae-example)

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

(defclass sampling-layer (th.layers::layer)
  ((mu-layer :initform nil)
   (log-var-layer :initform nil)))

(defun sampling-layer (input-size output-size)
  (let ((n (make-instance 'sampling-layer)))
    (with-slots (mu-layer log-var-layer) n
      (setf mu-layer (affine-layer input-size output-size :activation :nil)
            log-var-layer (affine-layer input-size output-size :activation nil)))
    n))

(defmethod th.layers::$train-parameters ((l sampling-layer))
  (with-slots (mu-layer log-var-layer) l
    (append (th.layers::$train-parameters mu-layer)
            (th.layers::$train-parameters log-var-layer))))

(defmethod th.layers::$parameters ((l sampling-layer))
  (with-slots (mu-layer log-var-layer) l
    (append (th.layers::$parameters mu-layer)
            (th.layers::$parameters log-var-layer))))

(defmethod th.layers::$execute ((l sampling-layer) x &key (trainp t))
  (with-slots (mu-layer log-var-layer) l
    (let* ((mu ($execute mu-layer x :trainp trainp))
           (log-var ($execute log-var-layer x :trainp trainp))
           (epsilon (apply #'rndn ($size mu))))
      ($+ mu ($* ($exp ($/ log-var 2)) epsilon)))))

;; define autoencoder = encoder + decoder
(defparameter *encoder* (sequence-layer
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
                         (sampling-layer 3136 2)))

(defparameter *decoder* (sequence-layer
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

(defparameter *model* (sequence-layer *encoder* *decoder*))

;; test model
($execute *model* (car *mnist-train-image-batches*) :trainp nil)

(defun loss (y x)
  (let ((d ($- y x)))
    ($/ ($dot d d) ($size y 0))))

(defparameter *epochs* 30)

($reset! *model*)
(time
 (with-foreign-memory-limit ()
   (loop :for epoch :from 0 :below *epochs*
         :do (loop :for xs :in *mnist-train-image-batches*
                   :for idx :from 1
                   :do (let* ((ys ($execute *model* xs))
                              (l (loss ys xs)))
                         (when (zerop (rem idx 20))
                           (prn idx "/" epoch "-" ($data l)))
                         ($adgd! *model*))))))

(setf *epochs* 1)
