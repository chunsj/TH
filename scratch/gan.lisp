;; from
;; https://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/

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

(defparameter *mnist-train-label-batches*
  (loop :for i :from 0 :below 60
        :for rng = (loop :for k :from (* i 1000) :below (* (1+ i) 1000)
                         :collect k)
        :collect ($contiguous! ($index ($ *mnist* :train-labels) 0 rng))))

(defparameter *discriminator* (parameters))
(defparameter *generator* (parameters))

;; discriminator network
(defparameter *dw1* ($parameter *discriminator* (vxavier (list 784 128))))
(defparameter *db1* ($parameter *discriminator* (zeros 1 128)))
(defparameter *dw2* ($parameter *discriminator* (vxavier (list 128 1))))
(defparameter *db2* ($parameter *discriminator* (zeros 1 1)))

;; generator network
(defparameter *gw1* ($parameter *generator* (vxavier (list 100 128))))
(defparameter *gb1* ($parameter *generator* (zeros 1 128)))
(defparameter *gw2* ($parameter *generator* (vxavier (list 128 784))))
(defparameter *gb2* ($parameter *generator* (zeros 1 784)))

(defparameter *os* (ones *batch-size* 1))

(defun generate (z)
  (let* ((gh1 ($relu ($+ ($@ z *gw1*) ($@ ($constant *os*) *gb1*))))
         (glogprob ($+ ($@ gh1 *gw2*) ($@ ($constant *os*) *gb2*))))
    ($sigmoid glogprob)))

(defun discriminate (x)
  (let* ((dh1 ($relu ($+ ($@ x *dw1*) ($@ ($constant *os*) *db1*))))
         (dlogit ($+ ($@ dh1 *dw2*) ($@ ($constant *os*) *db2*))))
    ($sigmoid dlogit)))

(defun samplez () ($constant ($ru! (tensor *batch-size* 100) -1 1)))

($cg! *discriminator*)
($cg! *generator*)

(defparameter *k* 1)
(defparameter *epoch* 10)

(loop :for epoch :from 1 :to *epoch*
      :do (progn
            (loop :for k :from 1 :to *k*
                  :for iter = 0
                  :for tloss = 0
                  :for mdreal = 0
                  :for mdfake = 0
                  :do (progn
                        (loop :for input :in *mnist-train-image-batches*
                              :for x = ($constant input)
                              :for z = (samplez)
                              :for g = (generate z)
                              :for d-real = (discriminate x)
                              :for d-fake = (discriminate g)
                              :for d-loss = ($neg ($mean ($+ ($log d-real)
                                                             ($log ($- ($constant 1) d-fake)))))
                              :do (progn
                                    (incf iter)
                                    (incf tloss ($data d-loss))
                                    (incf mdreal ($mean ($data d-real)))
                                    (incf mdfake ($mean ($data d-fake)))
                                    ($adgd! *discriminator*)
                                    ($cg! *generator*)))
                        (when (eq k *k*)
                          (prn "[DL]" epoch (/ tloss iter))
                          (prn "  PR:" (/ mdreal iter))
                          (prn "  PF:" (/ mdfake iter)))
                        (gcf)))
            (loop :for k :from 1 :to *k*
                  :for iter = 0
                  :for tloss = 0
                  n                  :for mdfake = 0
                  :do (progn
                        (loop :for i :from 0 :below ($count *mnist-train-image-batches*)
                              :for z = (samplez)
                              :for g = (generate z)
                              :for g-fake = (discriminate g)
                              :for g-loss = ($neg ($mean ($log g-fake)))
                              :do (progn
                                    (incf iter)
                                    (incf tloss ($data g-loss))
                                    (incf mdfake ($mean ($data g-fake)))
                                    ($adgd! *generator*)
                                    ($cg! *discriminator*)))
                        (when (eq k *k*)
                          (prn "[GL]" epoch (/ tloss iter))
                          (prn "  PG:" (/ mdfake iter)))
                        (gcf)))))

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
