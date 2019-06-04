;; from
;; https://github.com/soumith/dcgan.torch
;; https://towardsdatascience.com.having-fun-with-deep-convolutional-gans-f4f8393686ed

(defpackage :dcgan
  (:use #:common-lisp
        #:mu
        #:th
        #:th.image
        #:th.db.mnist))

(in-package :dcgan)

;; load mnist data, takes ~22 secs in macbook 2017
(defparameter *mnist* (read-mnist-data))

;; mnist data has following dataset
;; train-images, train-labels and test-images, test-labels
(prn *mnist*)

;; png output directory
(defparameter *output* (format nil "~A/Desktop" (user-homedir-pathname)))

;; 7x7 png output function
(defun outpngs (data fname &optional (w 28) (h 28))
  (let* ((n 7)
         (img (opticl:make-8-bit-gray-image (* n w) (* n h)))
         (data (mapcar (lambda (data) ($reshape data w h)) data)))
    (loop :for i :from 0 :below n
          :do (loop :for j :from 0 :below n
                    :for sx = (* j w)
                    :for sy = (* i h)
                    :for d = ($ data (+ (* j n) i))
                    :do (loop :for i :from 0 :below h
                              :do (loop :for j :from 0 :below w
                                        :do (progn
                                              (setf (aref img (+ sx i) (+ sy j))
                                                    (round (* 255 (* 0.5 (+ 1 ($ d i j)))))))))))
    (opticl:write-png-file fname img)))

(defparameter *nz* 100)
(defparameter *imgw* 28)
(defparameter *imgh* 28)
(defparameter *nimg* (* *imgw* *imgh*))
(defparameter *hidden-size* 128)
(defparameter *batch-size* 120)
(defparameter *batch-count* (/ 60000 *batch-size*))

(defparameter *generator* (parameters))
(defparameter *gw1* ($push *generator* (vxavier (list *nz* *nimg*))))
(defparameter *gb1* ($push *generator* (zeros *nimg*)))
(defparameter *gk2* ($push *generator* ($* 0.01 (rndn 16 32 4 4))))
(defparameter *gb2* ($push *generator* ($* 0.01 (rndn 32))))
(defparameter *gk3* ($push *generator* ($* 0.04 (rndn 32 1 4 4))))
(defparameter *gb3* ($push *generator* ($* 0.04 (rndn 1))))

(defun generate (z)
  (let ((nbatch ($size z 0)))
    (-> z
        ($affine *gw1* *gb1*)
        ($reshape nbatch 16 7 7) ;; 16 plane, 7x7
        ($selu)
        ($dconv2d *gk2* *gb2* 2 2 1 1) ;; 32 plane, 14x14
        ($selu)
        ($dconv2d *gk3* *gb3* 2 2 1 1) ;; 1 plane, 28x28
        ($tanh))))

;; generator shape checking
(let* ((nbatch 10)
       (noise (rndn nbatch *nz*)))
  ($cg! *generator*)
  (prn noise)
  (prn (generate noise))
  ($cg! *generator*))

(defparameter *discriminator* (parameters))
(defparameter *dk1* ($push *discriminator* ($* 0.04 (rndn 32 1 4 4))))
(defparameter *db1* ($push *discriminator* ($* 0.04 (rndn 32))))
(defparameter *dk2* ($push *discriminator* ($* 0.01 (rndn 16 32 4 4))))
(defparameter *db2* ($push *discriminator* ($* 0.01 (rndn 16))))
(defparameter *dw3* ($push *discriminator* ($* 0.03 (rndn *nimg* *hidden-size*))))
(defparameter *db3* ($push *discriminator* (zeros *hidden-size*)))
(defparameter *dw4* ($push *discriminator* ($* 0.04 (rndn *hidden-size* 1))))
(defparameter *db4* ($push *discriminator* (zeros 1)))

(defun discriminate (x)
  (let ((nbatch ($size x 0)))
    (-> x
        ($conv2d *dk1* *db1* 2 2 1 1) ;; 32 plane, 14x14
        ($lrelu)
        ($conv2d *dk2* *db2* 2 2 1 1) ;; 16 plane, 7x7
        ($selu)
        ($reshape nbatch *nimg*) ;; 1x784, flatten
        ($affine *dw3* *db3*)
        ($selu)
        ($affine *dw4* *db4*) ;; 1x1
        ($sigmoid))))

;; discriminator shape checking
(let* ((nbatch 10)
       (x (rnd nbatch 1 *imgh* *imgw*)))
  ($cg! *discriminator*)
  (prn x)
  (prn (discriminate x))
  ($cg! *discriminator*))

(defun samplez () (rndn *batch-size* *nz*))

(defun bced (dr df) ($+ ($bce dr ($one dr)) ($bce df ($zero df))))
(defun bceg (df) ($bce df ($one df)))

(defun lossd (dr df) (bced dr df))
(defun lossg (df) (bceg df))

(defun optm (params) ($amgd! params 1E-3))

(defparameter *epoch* 20)
(defparameter *k* 1)

($cg! *generator*)
($cg! *discriminator*)

;; renormalize values between -1 and 1.
(defparameter *mnist-train-image-batches*
  (loop :for i :from 0 :below *batch-count*
        :for range = (loop :for k :from (* i *batch-size*) :below (* (1+ i) *batch-size*)
                           :collect k)
        :collect ($contiguous! ($- ($* 2 ($index ($ *mnist* :train-images) 0 range)) 1))))

(defparameter *train-data-batches* (subseq *mnist-train-image-batches* 0))
(defparameter *train-count* ($count *train-data-batches*))

(gcf)

(time
 (loop :for epoch :from 1 :to *epoch*
       :for dloss = 0
       :for gloss = 0
       :do (progn
             ($cg! *generator*)
             ($cg! *discriminator*)
             (prn "*****")
             (prn "EPOCH:" epoch)
             (loop :for data :in *train-data-batches*
                   :for bidx :from 0
                   :for x = ($reshape data *batch-size* 1 *imgh* *imgw*)
                   :for z = (samplez)
                   :do (let ((dlv nil)
                             (dgv nil))
                         ;; discriminator
                         (dotimes (k *k*)
                           (let* ((dr (discriminate x))
                                  (df (discriminate (generate z)))
                                  (l ($data (lossd dr df))))
                             (incf dloss l)
                             (setf dlv l)
                             (optm *discriminator*)
                             ($cg! *generator*)
                             ($cg! *discriminator*)))
                         ;; generator
                         (let* ((df (discriminate (generate z)))
                                (l ($data (lossg df))))
                           (incf gloss l)
                           (setf dgv l)
                           (optm *generator*)
                           ($cg! *generator*)
                           ($cg! *discriminator*))
                         (when (zerop (rem bidx 10))
                           (prn "  D/G:" bidx dlv dgv))))
             ;; output at every epoch
             (prn " LOSS:" epoch (/ dloss *train-count* *k*) (/ gloss *train-count*))
             (let ((generated (generate (samplez))))
               (outpngs (loop :for i :from 0 :below 49
                              :collect ($index ($data generated) 0 (random *batch-size*)))
                        (format nil "~A/samples-~A.png" *output* epoch))
               ($cg! *generator*)
               ($cg! *discriminator*)))))

;; generate samples
(let ((generated (generate (samplez))))
  (outpngs (loop :for i :from 0 :below 49
                 :collect ($index ($data generated) 0 (random *batch-size*)))
           (format nil "~A/samples.png" *output*))
  ($cg! *generator*)
  ($cg! *discriminator*))

;; check training data
(let ((x (car *train-data-batches*)))
  (outpngs (loop :for i :from 0 :below 49
                 :collect ($index x 0 i))
           (format nil "~A/images.png" *output*)))

(setf *mnist* nil
      *mnist-train-image-batches* nil
      *train-data-batches* nil)

(gcf)
