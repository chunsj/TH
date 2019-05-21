;; from
;; https://wiseodd.github.io/techblog/2017/01/20/gan-pytorch/
;; with tanh generator

(defpackage :gan2
  (:use #:common-lisp
        #:mu
        #:th
        #:th.db.mnist))

(in-package :gan2)

;; load mnist data, takes ~22 secs in macbook 2017
(defparameter *mnist* (read-mnist-data))

;; mnist data has following dataset
;; train-images, train-labels and test-images, test-labels
(prn *mnist*)

(defparameter *output* (format nil "~A/Desktop" (user-homedir-pathname)))

(defun bced (dr df) ($+ ($bce dr ($one dr)) ($bce df ($zero df))))
(defun bceg (df) ($bce df ($one df)))

(defun lossd (dr df) (bced dr df))
(defun lossg (df) (bceg df))

(defun optm (params) ($amgd! params 1E-3))

(defun outpng (data fname &optional (w 28) (h 28))
  (let ((img (opticl:make-8-bit-gray-image w h))
        (d ($reshape data w h)))
    (loop :for i :from 0 :below h
          :do (loop :for j :from 0 :below w
                    :do (progn
                          (setf (aref img i j) (round (* 255 (* 0.5 (+ 1 ($ d i j)))))))))
    (opticl:write-png-file fname img)))

;; training data - uses batches for performance, it affects quantity as well; 60 works well
(defparameter *batch-size* 50)
(defparameter *batch-count* (/ 60000 *batch-size*))

(defparameter *mnist-train-image-batches*
  (loop :for i :from 0 :below *batch-count*
        :for range = (loop :for k :from (* i *batch-size*) :below (* (1+ i) *batch-size*)
                           :collect k)
        :collect ($contiguous! ($- ($* 2 ($index ($ *mnist* :train-images) 0 range)) 1))))

;; check range of -1 to 1
(prn (car *mnist-train-image-batches*))
(prn ($max (car *mnist-train-image-batches*)))
(prn ($min (car *mnist-train-image-batches*)))

(defparameter *discriminator* (parameters))
(defparameter *generator* (parameters))

(defparameter *gen-size* 100)
(defparameter *hidden-size* 128)
(defparameter *img-size* (* 28 28))

(defun xinit (size) ($* (apply #'rndn size) (/ 1 (sqrt (/ ($ size 0) 2)))))

(defparameter *os* (ones *batch-size* 1))

;; generator network
(defparameter *gw1* ($push *generator* (xinit (list *gen-size* *hidden-size*))))
(defparameter *gb1* ($push *generator* (zeros 1 *hidden-size*)))
(defparameter *gw2* ($push *generator* (xinit (list *hidden-size* *img-size*))))
(defparameter *gb2* ($push *generator* (zeros 1 *img-size*)))

(defun generate (z)
  (let* ((h ($lrelu ($+ ($@ z *gw1*) ($@ *os* *gb1*)) 0.2))
         (x ($tanh ($+ ($@ h *gw2*) ($@ *os* *gb2*)))))
    x))

;; discriminator network
(defparameter *dw1* ($push *discriminator* (xinit (list *img-size* *hidden-size*))))
(defparameter *db1* ($push *discriminator* (zeros 1 *hidden-size*)))
(defparameter *dw2* ($push *discriminator* (xinit (list *hidden-size* 1))))
(defparameter *db2* ($push *discriminator* (zeros 1 1)))

(defun discriminate (x)
  (let* ((h ($lrelu ($+ ($@ x *dw1*) ($@ *os* *db1*)) 0.2))
         (y ($sigmoid ($+ ($@ h *dw2*) ($@ *os* *db2*)))))
    y))

(defun samplez () (rndn *batch-size* *gen-size*))

(defparameter *epoch* 100)
(defparameter *k* 1)

($cg! *discriminator*)
($cg! *generator*)

(defparameter *train-data-batches* (subseq *mnist-train-image-batches* 0))
(defparameter *train-count* ($count *train-data-batches*))

(gcf)

(with-foreign-memory-limit
    (loop :for epoch :from 1 :to *epoch*
          :for dloss = 0
          :for gloss = 0
          :do (progn
                ($cg! *discriminator*)
                ($cg! *generator*)
                (prn "*****")
                (prn "EPOCH:" epoch)
                (loop :for x :in *train-data-batches*
                      :for bidx :from 0
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
                                ($cg! *discriminator*)
                                ($cg! *generator*)))
                            ;; generator
                            (let* ((df (discriminate (generate z)))
                                   (l ($data (lossg df))))
                              (incf gloss l)
                              (setf dgv l)
                              (optm *generator*)
                              ($cg! *discriminator*)
                              ($cg! *generator*))
                            (when (zerop (rem bidx 100))
                              (prn "  D/L:" bidx dlv dgv))))
                (when (zerop (rem epoch 1))
                  (let ((g (generate (samplez))))
                    ($cg! *discriminator*)
                    ($cg! *generator*)
                    (loop :for i :from 1 :to 1
                          :for s = (random *batch-size*)
                          :for fname = (format nil "~A/i~A-~A.png" *output* epoch i)
                          :do (outpng ($index ($data g) 0 s) fname))))
                (prn " LOSS:" epoch (/ dloss *train-count*) (/ gloss *train-count*)))))

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

;; generate samples
(let ((generated (generate (samplez))))
  (outpngs (loop :for i :from 0 :below 49
                 :collect ($index ($data generated) 0 i))
           (format nil "~A/samples.png" *output*))
  ($cg! *discriminator*)
  ($cg! *generator*))
