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

(defparameter *epochs* 144000)

(defun vae-loss (model xs &optional (usekl t) (beta 1) (trainp t) verbose)
  (let* ((ys ($execute model xs :trainp trainp))
         (recon-loss ($bce ys xs)))
    (if usekl
        (let* ((args ($function-arguments ($ ($ model 0) 6)))
               (m ($size xs 0))
               (mu ($ args 0))
               (log-var ($ args 1))
               (kl ($* ($sum ($+ ($exp log-var) ($* mu mu) -1 ($- log-var)))
                       (/ 1 m)
                       beta
                       0.5)))
          (when verbose (prn "RECON/KL:" recon-loss kl))
          ($+ recon-loss kl))
        recon-loss)))

($reset! *model*)
(time
 (with-foreign-memory-limit ()
   (let ((pstep 1)
         (estep 10))
     (loop :for epoch :from 1 :to *epochs*
           :do (loop :for xs :in (subseq *mnist-train-image-batches* 0 1)
                     :for idx :from 1
                     :do (let ((l (vae-loss *model* xs t 1 t (zerop (rem epoch estep)))))
                           (when (zerop (rem idx pstep))
                             (prn idx "/" epoch ":" ($data l)))
                           ($amgd! *model* 1E-4)))))))

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

($save-weights "./scratch/vae" *model*)
($load-weights "./scratch/vae" *model*)
