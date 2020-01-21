(defpackage :ragan
  (:use #:common-lisp
        #:mu
        #:th
        #:th.image
        #:th.layers
        #:th.db.mnist))

(in-package :ragan)

(defun build-batches (batch-size batch-count)
  (let ((mnist (read-mnist-data)))
    (loop :for i :from 0 :below batch-count
          :for s = (* i batch-size)
          :for e = (* (1+ i) batch-size)
          :for r = (loop :for k :from s :below e :collect k)
          :collect ($contiguous! ($reshape! ($- ($* 2 ($index ($ mnist :train-images) 0 r)) 1)
                                            batch-size 1 28 28)))))

(defparameter *batch-size* 120)
(defparameter *batch-count* (/ 60000 *batch-size*))

(defparameter *mnist-batches* (build-batches *batch-size* *batch-count*))

(defparameter *latent-dim* 100)

(defparameter *generator* (sequential-layer
                           (reshape-layer *latent-dim* 1 1)
                           (full-convolution-2d-layer *latent-dim* 128 3 3
                                                      :weight-initializer :random-normal
                                                      :weight-initialization '(0 0.02)
                                                      :activation :selu)
                           (full-convolution-2d-layer 128 64 3 3
                                                      :stride-width 2 :stride-height 2
                                                      :weight-initializer :random-normal
                                                      :weight-initialization '(0 0.02)
                                                      :activation :selu)
                           (full-convolution-2d-layer 64 32 2 2
                                                      :stride-width 2 :stride-height 2
                                                      :weight-initializer :random-normal
                                                      :weight-initialization '(0 0.02)
                                                      :activation :selu)
                           (full-convolution-2d-layer 32 1 2 2
                                                      :stride-width 2 :stride-height 2
                                                      :weight-initializer :random-normal
                                                      :weight-initialization '(0 0.02)
                                                      :activation :tanh)))

(defparameter *discriminator* (sequential-layer
                               (convolution-2d-layer 1 64 4 4
                                                     :stride-width 2 :stride-height 2
                                                     :padding-width 1 :padding-height 1
                                                     :weight-initializer :random-normal
                                                     :weight-initialization '(0 0.02)
                                                     :activation :selu)
                               (convolution-2d-layer 64 32 4 4
                                                     :stride-width 2 :stride-height 2
                                                     :padding-width 1 :padding-height 1
                                                     :weight-initializer :random-normal
                                                     :weight-initialization '(0 0.02)
                                                     :activation :selu)
                               (convolution-2d-layer 32 16 4 4
                                                     :stride-width 2 :stride-height 2
                                                     :padding-width 1 :padding-height 1
                                                     :weight-initializer :random-normal
                                                     :weight-initialization '(0 0.02)
                                                     :activation :selu)
                               (convolution-2d-layer 16 1 3 3
                                                     :weight-initializer :random-normal
                                                     :weight-initialization '(0 0.02)
                                                     :activation :nil)
                               (reshape-layer 1)))

(defparameter *lr* 4E-4)

(defun optim (model) ($amgd! model *lr* 0.5 0.999))

(defun generate (&key (trainp t))
  "generates fake images from random normal inputs"
  ($execute *generator* (rndn *batch-size* *latent-dim*) :trainp trainp))

(defun discriminate (xs &key (trainp t))
  "check whether inputs are real or fake"
  ($execute *discriminator* xs :trainp trainp))

(defun discriminator-loss (fake-scores real-scores)
  ($* 0.5 ($+ ($mean ($expt ($- real-scores 1) 2))
              ($mean ($expt fake-scores 2)))))

(defun train-discriminator (xs &optional verbose)
  "teaching discriminator how to discriminate reals from fakes"
  (let* ((fake-scores (discriminate (generate :trainp nil)))
         (real-scores (discriminate xs))
         (dloss (discriminator-loss fake-scores real-scores)))
    (when verbose (prn "  DL:" (if ($parameterp dloss) ($data dloss) dloss)))
    (optim *discriminator*)))

(defun generator-loss (fake-scores)
  ($* 0.5 ($mean ($expt ($- fake-scores 1) 2))))

(defun train-generator (&optional verbose)
  "teaching generator how to create more real fakes"
  (let* ((fake-scores (discriminate (generate) :trainp nil))
         (gloss (generator-loss fake-scores)))
    (when verbose (prn "  GL:" (if ($parameterp gloss) ($data gloss) gloss)))
    (optim *generator*)))

(defun train-batch (xs epoch idx)
  (let ((verbose (zerop (rem idx 100))))
    (when verbose (prn "EPOCH/IDX>" epoch ":" idx))
    (loop :for k :from 0 :below 1
          :do (train-discriminator xs verbose))
    (train-generator verbose)
    (when (zerop (rem idx 500))
      (let ((generated (generate :trainp nil))
            (fname (format nil "~A/Desktop/~A-~A.png" (namestring (user-homedir-pathname))
                           epoch idx)))
        (write-tensors-png-file generated fname)))
    (when verbose (th::report-foreign-memory-allocation))))

(defun train (epochs batches)
  (loop :for epoch :from 1 :to epochs
        :do (loop :for xs :in batches
                  :for idx :from 0
                  :do (train-batch xs epoch idx))))

(defparameter *epochs* 40)

($reset! (list *generator* *discriminator*))

(time
 (with-foreign-memory-limit ()
   (train *epochs* *mnist-batches*)))

(let ((generated (generate :trainp nil))
      (fname (format nil "~A/Desktop/images.png" (namestring (user-homedir-pathname)))))
  (write-tensors-png-file generated fname))

(let ((xs (car *mnist-batches*))
      (fname (format nil "~A/Desktop/ixs.png" (namestring (user-homedir-pathname)))))
  (write-tensors-png-file xs fname))

(gcf)
(setf *epochs* 1)
(th::report-foreign-memory-allocation)
