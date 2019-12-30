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

(defparameter *latent-dim* 100)
(defparameter *imgsz* (* 28 28))
(defparameter *hidden-size* 128)

(defparameter *generator* (sequential-layer
                           (affine-layer *latent-dim* *imgsz* :activation :nil)
                           (reshape-layer 16 7 7)
                           (full-convolution-2d-layer 16 32 4 4
                                                      :stride-width 2 :stride-height 2
                                                      :padding-width 1 :padding-height 1
                                                      :batch-normalization-p t
                                                      :biasp nil
                                                      :activation :lrelu)
                           (full-convolution-2d-layer 32 1 4 4
                                                      :stride-width 2 :stride-height 2
                                                      :padding-width 1 :padding-height 1
                                                      :batch-normalization-p t
                                                      :biasp nil
                                                      :activation :sigmoid)))

(defparameter *discriminator* (sequential-layer
                               (convolution-2d-layer 1 32 4 4
                                                     :stride-width 2 :stride-height 2
                                                     :padding-width 1 :padding-height 1
                                                     :activation :lrelu)
                               (convolution-2d-layer 32 16 4 4
                                                     :stride-width 2 :stride-height 2
                                                     :padding-width 1 :padding-height 1
                                                     :batch-normalization-p t
                                                     :biasp nil
                                                     :activation :lrelu)
                               (reshape-layer *imgsz*)
                               (affine-layer *imgsz* *hidden-size*
                                             :batch-normalization-p t
                                             :activation :lrelu)
                               (affine-layer *hidden-size* 1
                                             :batch-normalization-p t
                                             :activation :sigmoid)))

(defparameter *lr* 0.001)
(defparameter *real-labels* (ones *batch-size*))
(defparameter *fake-labels* (zeros *batch-size*))

(defun optim (model)
  ($amgd! model *lr* 0.5 0.999)
  ($cg! *generator*)
  ($cg! *discriminator*))

(defun train-discriminator (xs &optional verbose)
  (let* ((z (rndn *batch-size* 100))
         (fake-images ($execute *generator* z))
         (real-images xs)
         (fake-scores ($execute *discriminator* fake-images))
         (real-scores ($execute *discriminator* real-images))
         (fake-loss ($bce fake-scores *fake-labels*))
         (real-loss ($bce real-scores *real-labels*))
         (dloss ($* 0.5 ($+ fake-loss real-loss))))
    (when verbose (prn "  DL:" (if ($parameterp dloss) ($data dloss) dloss)))
    (optim *discriminator*)))

(defun train-generator (&optional verbose)
  (let* ((z (rndn *batch-size* 100))
         (gen-images ($execute *generator* z))
         (gen-scores ($execute *discriminator* gen-images))
         (gloss ($bce gen-scores *real-labels*)))
    (when verbose (prn "  GL:" (if ($parameterp gloss) ($data gloss) gloss)))
    (optim *generator*)))

(defun write-tensor-at (img x y tx)
  (let* ((h ($size tx 1))
         (w ($size tx 2))
         (sx (* x w))
         (sy (* y h)))
    (loop :for j :from 0 :below h
          :do (loop :for i :from 0 :below w
                    :for px = ($ tx 0 j i)
                    :do (setf (aref img (+ sy j) (+ sx i)) (round (* 255 px)))))))

(defun outpngs (data fname)
  (let* ((h ($size ($ data 0) 1))
         (w ($size ($ data 0) 2))
         (dn ($size data 0))
         (n (ceiling (sqrt dn)))
         (nc n)
         (nr (ceiling (/ dn n)))
         (img (opticl:make-8-bit-gray-image (* nr h) (* nc w))))
    (loop :for sy :from 0 :below nr
          :do (loop :for sx :from 0 :below nc
                    :for idx = (+ (* sy nr) sx)
                    :for tx = (when (< idx dn) ($ data idx))
                    :when tx
                      :do (write-tensor-at img sx sy tx)))
    (opticl:write-png-file fname img)))

(defun train (xs epoch idx)
  (let ((verbose (zerop (rem idx 50))))
    (when verbose (prn "EPOCH/IDX>" epoch ":" idx))
    (loop :for k :from 0 :below 1
          :do (train-discriminator xs verbose))
    (train-generator verbose)
    (when (zerop (rem idx 500))
      (let ((generated ($execute *generator* (rndn *batch-size* *latent-dim*) :trainp nil))
            (fname (format nil "~A/Desktop/~A-~A.png" (namestring (user-homedir-pathname))
                           epoch idx)))
        (outpngs generated fname)))
    (when verbose (th::report-foreign-memory-allocation))))

(defparameter *epochs* 10)

($reset! *generator*)
($reset! *discriminator*)

(time
 (with-foreign-memory-limit ()
   (loop :for epoch :from 1 :to *epochs*
         :do (loop :for xs :in *mnist-batches*
                   :for idx :from 0
                   :do (train xs epoch idx)))))

(let ((generated ($execute *generator* (rndn *batch-size* *latent-dim*) :trainp nil))
      (fname (format nil "~A/Desktop/images.png" (namestring (user-homedir-pathname)))))
  (outpngs generated fname))

(gcf)
