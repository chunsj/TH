(defpackage :vae-example
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers
        #:th.db.mnist))

(in-package :vae-example)

(defparameter *mnist* (read-mnist-data))

(defparameter *batch-size* 32)
(defparameter *max-batch-count* 10)
(defparameter *batch-count* (min *max-batch-count*
                                 (/ ($size ($ *mnist* :train-images) 0) *batch-size*)))

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
                                               :stride-width 2 :stride-height 2
                                               :batch-normalization-p t
                                               :activation :relu)
                         (convolution-2d-layer 32 64 3 3
                                               :padding-width 1 :padding-height 1
                                               :stride-width 2 :stride-height 2
                                               :batch-normalization-p t
                                               :activation :relu)
                         (flatten-layer)
                         (affine-layer 3136 16
                                       :batch-normalization-p t
                                       :activation :nil)
                         (parallel-layer (affine-layer 16 2 :activation :nil)
                                         (affine-layer 16 2 :activation :nil))
                         (functional-layer #'sample-function)))

(defparameter *decoder* (sequential-layer
                         (affine-layer 2 3136 :activation :relu)
                         (reshape-layer 64 7 7)
                         (full-convolution-2d-layer 64 64 3 3
                                                    :padding-width 1 :padding-height 1
                                                    :stride-width 2 :stride-height 2
                                                    :adjust-width 1 :adjust-height 1
                                                    :batch-normalization-p t
                                                    :activation :relu)
                         (full-convolution-2d-layer 64 32 3 3
                                                    :stride-width 2 :stride-height 2
                                                    :padding-width 1 :padding-height 1
                                                    :adjust-width 1 :adjust-height 1
                                                    :batch-normalization-p t
                                                    :activation :relu)
                         (full-convolution-2d-layer 32 1 3 3
                                                    :padding-width 1 :padding-height 1
                                                    :batch-normalization-p t
                                                    :activation :sigmoid)))

(defparameter *model* (sequential-layer *encoder* *decoder*))

(defun vae-loss (model xs &optional (usekl t) (trainp t))
  (let* ((ys ($execute model xs :trainp trainp))
         (recon-loss ($bce ys xs)))
    (if usekl
        (let* ((args ($function-arguments ($ ($ model 0) 5)))
               (m ($size xs 0))
               (mu ($ args 0))
               (log-var ($ args 1))
               (kl ($* ($sum ($+ ($exp log-var) ($* mu mu) -1 ($- log-var)))
                       (/ 1 m)
                       0.5)))
          (list recon-loss kl))
        (list recon-loss 0))))

(defun update-params (model gd)
  (cond ((eq gd :adam) ($amgd! model 1E-3))
        ((eq gd :rmsprop) ($rmgd! model))
        (t ($adgd! model))))

(defun update-kl-params (model gd)
  (cond ((eq gd :adam) ($amgd! ($ model 0) 1E-3))
        ((eq gd :rmsprop) ($rmgd! ($ model 0)))
        (t ($adgd! ($ model 0))))
  ($cg! model))

(defun vae-train-step (model xs st gd)
  (let* ((ntr 10)
         (beta 0.01)
         (pstep 10))
    (loop :for i :from 0 :to ntr
          :do (progn
                (vae-loss model xs nil)
                (update-params model gd)))
    (let* ((losses (vae-loss model xs t))
           (lr (car losses))
           (lkl (cadr losses))
           (l ($+ lr ($* beta lkl))))
      (when (zerop (rem st pstep))
        (prn st ":"
             (format nil "~,4E" (if ($parameterp l) ($data l) l))
             (format nil "~,4E" (if ($parameterp lr) ($data lr) lr))
             (format nil "~,4E" (if ($parameterp lkl) ($data lkl) lkl))))
      (update-params model gd))))

(defun vae-train-step-2 (model xs st gd)
  (let* ((ntr 10)
         (beta 0.05)
         (pstep 10))
    (loop :for i :from 0 :to ntr
          :do (progn
                (vae-loss model xs nil)
                (update-params model gd)))
    (let* ((losses (vae-loss model xs t))
           (lr (car losses))
           (lkl (cadr losses))
           (l ($+ lr ($* beta lkl))))
      (when (zerop (rem st pstep))
        (prn st ":"
             (format nil "~,4E" (if ($parameterp l) ($data l) l))
             (format nil "~,4E" (if ($parameterp lr) ($data lr) lr))
             (format nil "~,4E" (if ($parameterp lkl) ($data lkl) lkl))))
      (update-kl-params model gd))))

(defun vae-train (epochs model batches)
  (let ((nbs ($count batches)))
    (loop :for epoch :from 1 :to epochs
          :do (loop :for xs :in batches
                    :for idx :from 1
                    :do (vae-train-step-2 model xs (+ idx (* nbs (1- epoch))) :rmsprop)))))

(defparameter *epochs* 1000)

($reset! *model*)

;; train
(time
 (with-foreign-memory-limit ()
   (vae-train *epochs* *model* *mnist-train-image-batches*)))

;; test model
($execute *model* (car *mnist-train-image-batches*) :trainp nil)

;; trained weights
($load-weights "./examples/weights/vae" *model*)
;; ($save-weights "./examples/weights/vae" *model*)

;; check results
(defun compare-xy (encoder decoder bs)
  (let* ((nb ($count bs))
         (bidx (random nb))
         (xs ($ bs bidx))
         (bn ($size xs 0))
         (es ($execute encoder xs :trainp nil))
         (ds ($execute decoder es :trainp nil))
         (ys ($reshape! ds bn 1 28 28))
         (idx (random bn))
         (x ($ xs idx))
         (y ($ ys idx))
         (inf ($concat (namestring (user-homedir-pathname)) "Desktop/input.png"))
         (ouf ($concat (namestring (user-homedir-pathname)) "Desktop/output.png")))
    (prn "BIDX:" bidx)
    (prn "ENCODED:" es)
    (prn "INDEX:" idx)
    (th.image:write-tensor-png-file x inf)
    (th.image:write-tensor-png-file y ouf)))

(compare-xy *encoder* *decoder* *mnist-train-image-batches*)

;; generate images
(defun genimg (decoder)
  (let* ((bn *batch-size*)
         (xs (rndn bn 2))
         (mn ($mean xs 0))
         (ds ($execute decoder xs :trainp nil))
         (ys ($reshape! ds bn 1 28 28))
         (fs ($concat (namestring (user-homedir-pathname)) "Desktop/gen~A.png")))
    (prn "XS:" ($ mn 0 0) ($exp ($ mn 0 1)))
    (loop :for i :from 0 :below (min 10 bn)
          :for filename = (format nil fs (1+ i))
          :do (th.image:write-tensor-png-file ($ ys i) filename))))

(genimg *decoder*)

(defun patchimg (&optional (n 21))
  (let* ((minv -1E0)
         (maxv 1E0)
         (sv (/ (- maxv minv) n))
         (xs (tensor (1+ n) (1+ n) 2)))
    (loop :for i :from 0 :to n
          :for vi = (- maxv (* i sv))
          :do (loop :for j :from 0 :to n
                    :for vj = (+ minv (* j sv))
                    :do (setf ($ xs i j 0) vj
                              ($ xs i j 1) vi)))
    (let* ((xs ($reshape! xs (* (1+ n) (1+ n)) 2))
           (mn ($mean xs 0))
           (ds ($execute *decoder* xs :trainp nil))
           (ys ($reshape! ds (1+ n) (1+ n) 1 28 28))
           (img (opticl:make-8-bit-gray-image (* (1+ n) 28) (* (1+ n) 28)))
           (fs ($concat (namestring (user-homedir-pathname)) "Desktop/patch.png")))
      (prn xs)
      (prn "MN:" ($ mn 0 0) ($exp ($ mn 0 1)))
      (loop :for ti :from 0 :to n
            :for sy = (* ti 28)
            :do (loop :for tj :from 0 :to n
                      :for tx = ($ ys ti tj)
                      :for sx = (* tj 28)
                      :do (loop :for ii :from 0 :below 28
                                :do (loop :for ij :from 0 :below 28
                                          :do (setf (aref img (+ sy ii) (+ sx ij))
                                                    (round (* 255 ($ tx 0 ii ij))))))))
      (prn ys)
      (opticl:write-png-file fs img))))

(defun showimg (xs)
  (let* ((bn ($size xs 0))
         (fs ($concat (namestring (user-homedir-pathname)) "Desktop/in~A.png")))
    (loop :for i :from 1 :to (min bn 40)
          :for filename = (format nil fs i)
          :do (th.image:write-tensor-png-file ($ xs (1- i)) filename))))

(showimg ($ *mnist-train-image-batches* 0))
