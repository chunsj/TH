(defpackage :gan-new
  (:use #:common-lisp
        #:mu
        #:th
        #:th.image
        #:th.layers
        #:th.db.mnist))

(in-package :gan-new)

(defun build-batches (batch-size batch-count)
  (let ((mnist (read-mnist-data)))
    (loop :for i :from 0 :below batch-count
          :for s = (* i batch-size)
          :for e = (* (1+ i) batch-size)
          :for r = (loop :for k :from s :below e :collect k)
          :collect ($contiguous! ($reshape! ($index ($ mnist :train-images) 0 r)
                                            batch-size 1 28 28)))))

(defparameter *batch-size* 60)
(defparameter *batch-count* (/ 60000 *batch-size*))

(defparameter *mnist-batches* (build-batches *batch-size* *batch-count*))
(defparameter *mnist-batches* (build-batches *batch-size* 10))

(defparameter *latent-dim* 100)

(defparameter *generator* (sequential-layer
                           (reshape-layer *latent-dim* 1 1)
                           (full-convolution-2d-layer *latent-dim* 512 4 4
                                                      :stride-width 2 :stride-height 2
                                                      :padding-width 1 :padding-height 1
                                                      :activation :relu
                                                      :batch-normalization-p t
                                                      :biasp nil)
                           (full-convolution-2d-layer 512 256 4 4
                                                      :stride-width 2 :stride-height 2
                                                      :padding-width 1 :padding-height 1
                                                      :activation :relu
                                                      :batch-normalization-p t
                                                      :biasp nil)
                           (full-convolution-2d-layer 256 128 4 4
                                                      :stride-width 2 :stride-height 2
                                                      :padding-width 1 :padding-height 1
                                                      :activation :relu
                                                      :batch-normalization-p t
                                                      :biasp nil)
                           (full-convolution-2d-layer 128 64 3 3
                                                      :stride-width 2 :stride-height 2
                                                      :padding-width 1 :padding-height 1
                                                      :activation :relu
                                                      :batch-normalization-p t
                                                      :biasp nil)
                           (full-convolution-2d-layer 64 1 2 2
                                                      :stride-width 2 :stride-height 2
                                                      :padding-width 1 :padding-height 1
                                                      :activation :sigmoid
                                                      :batch-normalization-p t
                                                      :biasp nil)))

(defparameter *discriminator* (sequential-layer
                               (convolution-2d-layer 1 64 4 4
                                                     :stride-width 2 :stride-height 2
                                                     :padding-width 1 :padding-height 1
                                                     :activation :lrelu
                                                     :biasp nil)
                               (convolution-2d-layer 64 128 4 4
                                                     :stride-width 2 :stride-height 2
                                                     :padding-width 1 :padding-height 1
                                                     :activation :lrelu
                                                     :batch-normalization-p t
                                                     :biasp nil)
                               (convolution-2d-layer 128 256 4 4
                                                     :stride-width 2 :stride-height 2
                                                     :padding-width 1 :padding-height 1
                                                     :activation :lrelu
                                                     :batch-normalization-p t
                                                     :biasp nil)
                               (convolution-2d-layer 256 512 4 4
                                                     :stride-width 2 :stride-height 2
                                                     :padding-width 1 :padding-height 1
                                                     :activation :lrelu
                                                     :batch-normalization-p t
                                                     :biasp nil)
                               (convolution-2d-layer 512 1 1 1
                                                     :stride-width 1 :stride-height 1
                                                     :activation :sigmoid)
                               (flatten-layer)))

(defparameter *lr* 0.0002)
(defparameter *real-labels* (ones *batch-size*))
(defparameter *fake-labels* (zeros *batch-size*))

(defun train-discriminator (xs &optional verbose)
  (let* ((z (rndn *batch-size* 100))
         (fake-images ($execute *generator* z))
         (real-images xs)
         (fake-scores ($execute *discriminator* fake-images))
         (real-scores ($execute *discriminator* real-images))
         (fake-loss ($bce fake-scores *fake-labels*))
         (real-loss ($bce real-scores *real-labels*))
         (dloss ($+ fake-loss real-loss)))
    (when verbose (prn "  DL:" (if ($parameterp dloss) ($data dloss) dloss)))
    ($amgd! *discriminator* *lr* 0.5 0.999)
    ($cg! *generator*)
    ($cg! *discriminator*)))

(defun train-generator (&optional verbose)
  (let* ((z (rndn *batch-size* 100))
         (fake-images ($execute *generator* z))
         (fake-scores ($execute *discriminator* fake-images))
         (gloss ($bce fake-scores *real-labels*)))
    (when verbose (prn "  GL:" (if ($parameterp gloss) ($data gloss) gloss)))
    ($amgd! *generator* *lr* 0.5 0.999)
    ($cg! *generator*)
    ($cg! *discriminator*)))

(defun train (xs epoch idx)
  (let ((verbose (zerop (rem idx 5))))
    (when verbose (prn epoch ":" idx))
    (train-discriminator xs verbose)
    (train-generator verbose)))

(defparameter *epochs* 100)

($reset! *generator*)
($reset! *discriminator*)

(time
 (with-foreign-memory-limit ()
   (loop :for epoch :from 1 :to *epochs*
         :do (loop :for xs :in *mnist-batches*
                   :for idx :from 0
                   :do (train xs epoch idx)))))

(defun outpngs (data81 fname &optional (w 28) (h 28))
  (let* ((n 9)
         (img (opticl:make-8-bit-gray-image (* n w) (* n h))))
    (loop :for i :from 0 :below n
          :do (loop :for j :from 0 :below n
                    :for sx = (* j w)
                    :for sy = (* i h)
                    :for d = ($ data81 (+ (* j n) i))
                    :do (loop :for i :from 0 :below h
                              :do (loop :for j :from 0 :below w
                                        :do (progn
                                              (setf (aref img (+ sx i) (+ sy j))
                                                    (round (* 255 ($ d 0 i j)))))))))
    (opticl:write-png-file fname img)))

(let ((generated ($execute *generator* (rndn 81 100) :trainp nil))
      (fname (format nil "~A/Desktop/81.png" (namestring (user-homedir-pathname)))))
  (outpngs generated fname))
