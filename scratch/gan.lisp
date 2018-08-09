;; from
;; https://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/

(ql:quickload :opticl)

(defpackage :gan-work
  (:use #:common-lisp
        #:mu
        #:th
        #:th.db.mnist))

(in-package :gan-work)

;; load mnist data, takes ~22 secs in macbook 2017
(defparameter *mnist* (read-mnist-data))

;; mnist data has following dataset
;; train-images, train-labels and test-images, test-labels
(print *mnist*)

;; training data - uses batches for performance
(defparameter *batch-size* 1000)

(defparameter *mnist-train-image-batches*
  (loop :for i :from 0 :below 60
        :for rng = (loop :for k :from (* i 1000) :below (* (1+ i) 1000)
                         :collect k)
        :collect ($contiguous! ($index ($ *mnist* :train-images) 0 rng))))

(defparameter *discriminator* (parameters))
(defparameter *generator* (parameters))

(defparameter *gen-size* 100)
(defparameter *hidden-size* 128)
(defparameter *img-size* (* 28 28))

;; discriminator network
(defparameter *dw1* ($parameter *discriminator* (vxavier (list *img-size* *hidden-size*))))
(defparameter *db1* ($parameter *discriminator* (zeros 1 *hidden-size*)))
(defparameter *dw2* ($parameter *discriminator* (vxavier (list *hidden-size* 1))))
(defparameter *db2* ($parameter *discriminator* (zeros 1 1)))

;; generator network
(defparameter *gw1* ($parameter *generator* (vxavier (list *gen-size* *hidden-size*))))
(defparameter *gb1* ($parameter *generator* (zeros 1 *hidden-size*)))
(defparameter *gw2* ($parameter *generator* (vxavier (list *hidden-size* *img-size*))))
(defparameter *gb2* ($parameter *generator* (zeros 1 *img-size*)))

(defparameter *os* (ones *batch-size* 1))

(defun generate (z)
  (let* ((gh1 ($relu ($+ ($@ z *gw1*) ($@ ($constant *os*) *gb1*))))
         (glogprob ($+ ($@ gh1 *gw2*) ($@ ($constant *os*) *gb2*))))
    ($sigmoid glogprob)))

(defun discriminate (x)
  (let* ((dh1 ($relu ($+ ($@ x *dw1*) ($@ ($constant *os*) *db1*))))
         (dlogit ($+ ($@ dh1 *dw2*) ($@ ($constant *os*) *db2*))))
    ($sigmoid dlogit)))

(defun samplez () ($constant ($ru! (tensor *batch-size* *gen-size*) -1 1)))

(defun outpng (data fname)
  (let ((img (opticl:make-8-bit-gray-image 28 28))
        (d ($reshape data 28 28)))
    (loop :for i :from 0 :below 28
          :do (loop :for j :from 0 :below 28
                    :do (progn
                          (setf (aref img i j) (round (* 255 ($ d i j)))))))
    (opticl:write-png-file fname img)))

($cg! *discriminator*)
($cg! *generator*)

(defparameter *epoch* 2)
(defparameter *k* 1)
(defparameter *j* 1)

(defparameter *eps* ($constant 1E-7))

(loop :for epoch :from 1 :to *epoch*
      :do (loop :for input :in *mnist-train-image-batches*
                :for bidx :from 0
                :for x = ($constant input)
                :for z = (samplez)
                :do (progn
                      (when (zerop (rem bidx 10))
                        (prn epoch "=>" bidx))
                      ($cg! *discriminator*)
                      ($cg! *generator*)
                      (loop :for k :from 1 :to *k*
                            :do (let* (;;(z (samplez))
                                       (g (generate z))
                                       (dr (discriminate x))
                                       (df (discriminate g))
                                       (ldr ($log ($+ dr *eps*)))
                                       (ldf ($log ($+ ($- ($constant 1) df) *eps*)))
                                       (l ($neg ($mean ($+ ldr ldf)))))
                                  ($adgd! *discriminator*)
                                  ($cg! *discriminator*)
                                  ($cg! *generator*)
                                  (when (and (zerop (rem bidx 10)) (eq k *k*))
                                    (prn "  LD:" ($data l))
                                    (prn "  PR:" ($mean ($data dr)))
                                    (prn "  PG:" ($mean ($data df)))
                                    (let ((fname (format nil
                                                         "/Users/Sungjin/Desktop/img~A.png"
                                                         bidx)))
                                      (outpng ($index ($data g) 0 0) fname)))))
                      (loop :for j :from 1 :to *j*
                            :do (let* (;;(z (samplez))
                                       (g (generate z))
                                       (gf (discriminate g))
                                       (l ($neg ($mean ($log ($+ gf *eps*))))))
                                  ($adgd! *generator*)
                                  ($cg! *discriminator*)
                                  ($cg! *generator*)
                                  (when (and (zerop (rem bidx 10)) (eq j *j*))
                                    (prn "  LG:" ($data l))
                                    (prn "  GF:" ($mean ($data gf))))))
                      (gcf))))

(-> (samplez)
    (generate)
    (discriminate)
    ($mean)
    (prn))

(-> ($ *mnist-train-image-batches* (random ($count *mnist-train-image-batches*)))
    ($constant)
    (discriminate)
    ($mean)
    (prn))

(-> (samplez)
    (generate)
    ($data)
    ($sum)
    (prn))

(-> ($ *mnist-train-image-batches* (random ($count *mnist-train-image-batches*)))
    ($sum)
    (prn))

(ql:quickload :opticl)

(defun outpng (data fname)
  (let ((img (opticl:make-8-bit-gray-image 28 28))
        (d ($reshape data 28 28)))
    (loop :for i :from 0 :below 28
          :do (loop :for j :from 0 :below 28
                    :do (progn
                          (setf (aref img i j) (round (* 255 ($ d i j)))))))
    (opticl:write-png-file fname img)))

(defparameter *samples* (-> (samplez)
                            (generate)
                            ($data)))
(defparameter *data* ($ *mnist-train-image-batches*
                        (random ($count *mnist-train-image-batches*))))

(let ((datag ($reshape ($index *samples* 0 (random *batch-size*)) 28 28))
      (datad ($reshape ($index *data* 0 (random *batch-size*)) 28 28)))
  (prn datag)
  (prn datad)
  (outpng datag "/Users/Sungjin/Desktop/testg.png")
  (outpng datad "/Users/Sungjin/Desktop/testd.png")
  ($cg! *discriminator*)
  ($cg! *generator*))
