;; from
;; https://wiseodd.github.io/techblog/2017/01/20/gan-pytorch/

(defpackage :gan
  (:use #:common-lisp
        #:mu
        #:th
        #:th.image
        #:th.db.mnist))

(in-package :gan)

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

;; XXX cannot figure out why adam works best (adadelta does not work well)
(defun optm (params) ($amgd! params 1E-3))

(defun outpng (data fname) (write-tensor-png-file ($reshape data 28 28) fname))

;; training data - uses batches for performance, 30, 60 works well
(defparameter *batch-size* 60)
(defparameter *batch-count* (/ 60000 *batch-size*))

(defparameter *mnist-train-image-batches*
  (loop :for i :from 0 :below *batch-count*
        :for range = (loop :for k :from (* i *batch-size*) :below (* (1+ i) *batch-size*)
                           :collect k)
        :collect ($contiguous! ($index ($ *mnist* :train-images) 0 range))))

(defparameter *discriminator* (parameters))
(defparameter *generator* (parameters))

(defparameter *gen-size* 100)
(defparameter *hidden-size* 128)
(defparameter *img-size* (* 28 28))

(defun xinit (size) ($variable ($* (apply #'rndn size) (/ 1 (sqrt (/ ($ size 0) 2))))))

(defparameter *os* (ones *batch-size* 1))

;; generator network
(defparameter *gw1* ($push *generator* (xinit (list *gen-size* *hidden-size*))))
(defparameter *gb1* ($push *generator* (zeros 1 *hidden-size*)))
(defparameter *gw2* ($push *generator* (xinit (list *hidden-size* *img-size*))))
(defparameter *gb2* ($push *generator* (zeros 1 *img-size*)))

;; you can apply leaky relu, $lrelu
(defun generate (z)
  (let* ((h ($relu ($+ ($@ z *gw1*) ($@ ($constant *os*) *gb1*))))
         (x ($sigmoid ($+ ($@ h *gw2*) ($@ ($constant *os*) *gb2*)))))
    x))

;; discriminator network
(defparameter *dw1* ($push *discriminator* (xinit (list *img-size* *hidden-size*))))
(defparameter *db1* ($push *discriminator* (zeros 1 *hidden-size*)))
(defparameter *dw2* ($push *discriminator* (xinit (list *hidden-size* 1))))
(defparameter *db2* ($push *discriminator* (zeros 1 1)))

;; you can apply leaky relu, $lrelu
(defun discriminate (x)
  (let* ((h ($relu ($+ ($@ x *dw1*) ($@ ($constant *os*) *db1*))))
         (y ($sigmoid ($+ ($@ h *dw2*) ($@ ($constant *os*) *db2*)))))
    y))

(defun samplez () ($constant (rndn *batch-size* *gen-size*)))

(defparameter *epoch* 100)
(defparameter *k* 1)

($cg! *discriminator*)
($cg! *generator*)

(defparameter *train-data-batches* (subseq *mnist-train-image-batches* 0))
(defparameter *train-count* ($count *train-data-batches*))

(gcf)

(loop :for epoch :from 1 :to *epoch*
      :for dloss = 0
      :for gloss = 0
      :do (progn
            ($cg! *discriminator*)
            ($cg! *generator*)
            (prn "*****")
            (prn "EPOCH:" epoch)
            (loop :for data :in *train-data-batches*
                  :for bidx :from 0
                  :for x = ($constant data)
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
            (prn " LOSS:" epoch (/ dloss *train-count*) (/ gloss *train-count*))
            (gcf)))

(defun outpngs49 (data81 fname &optional (w 28) (h 28))
  (let* ((n 7)
         (img (opticl:make-8-bit-gray-image (* n w) (* n h)))
         (datas (mapcar (lambda (data) ($reshape data w h)) data81)))
    (loop :for i :from 0 :below n
          :do (loop :for j :from 0 :below n
                    :for sx = (* j w)
                    :for sy = (* i h)
                    :for d = ($ datas (+ (* j n) i))
                    :do (loop :for i :from 0 :below h
                              :do (loop :for j :from 0 :below w
                                        :do (progn
                                              (setf (aref img (+ sx i) (+ sy j))
                                                    (round (* 255 ($ d i j)))))))))
    (opticl:write-png-file fname img)))

;; generate samples
(let ((generated (generate (samplez))))
  (outpngs49 (loop :for i :from 0 :below 49
                   :collect ($index ($data generated) 0 i))
             (format nil "~A/49.png" *output*))
  ($cg! *discriminator*)
  ($cg! *generator*))

(setf *mnist* nil
      *mnist-train-image-batches* nil
      *train-data-batches* nil)

(gcf)
