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

(defun sample-function (mu log-var &key (trainp t))
  (declare (ignore trainp))
  (let ((epsilon (apply #'rndn ($size mu))))
    ($+ mu ($* ($exp ($/ log-var 2)) epsilon))))

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
                         (parallel-layer (affine-layer 3136 2 :activation :nil)
                                         (affine-layer 3136 2 :activation :nil))
                         (functional-layer #'sample-function)))

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

(defparameter *epochs* 30)

(defun vae-loss (model xs &optional (rf 1000))
  (let* ((ys ($execute model xs))
         (recon-loss ($bce ys xs))
         (args ($function-arguments ($ ($ model 0) 6)))
         (mu ($ args 0))
         (log-var ($ args 1))
         (kl ($* 0.5 ($sum ($+ ($exp log-var) ($* mu mu) -1 ($- log-var))))))
    ($+ ($* rf recon-loss) kl)))

($reset! (list *encoder* *decoder*))
(time
 (with-foreign-memory-limit ()
   (loop :for epoch :from 1 :to *epochs*
         :do (loop :for xs :in *mnist-train-image-batches*
                   :for idx :from 1
                   :do (let ((l (vae-loss *model* xs)))
                         (when (zerop (rem idx 10))
                           (prn idx "/" epoch ":" ($data l)))
                         ($adgd! (list *encoder* *decoder*)))))))

(setf *epochs* 3)

;; XXX for analysis
(prn ($execute *encoder* ($0 *mnist-train-image-batches*) :trainp nil))
(let ((res ($execute *decoder* ($execute *encoder* ($0 *mnist-train-image-batches*) :trainp nil)
                     :trainp nil)))
  (th.image:write-tensor-png-file ($reshape ($ res 2) 1 28 28) "/Users/Sungjin/Desktop/hello.png"))

;; XXX need to fix batch normalization input shape problem
(let ((res ($execute *decoder* (tensor '((0 0) (0 0)))
                     :trainp nil)))
  (th.image:write-tensor-png-file ($reshape ($ res 0) 1 28 28) "/Users/Sungjin/Desktop/hello.png"))
