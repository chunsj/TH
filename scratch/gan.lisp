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

(defparameter *batch-size* 1000)

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

(defparameter *k* 5)
(defparameter *epoch* 10)

(loop :for epoch :from 1 :to *epoch*
      :do (progn
            (loop :for k :from 1 :to *k*
                  :for tloss = 0
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
                                    (incf tloss ($data d-loss))
                                    ($adgd! *discriminator*)
                                    ($cg! *generator*)))
                        (when (eq k *k*) (prn "DL:" (/ tloss *k*)))
                        (gcf)))
            (loop :for k :from 1 :to *k*
                  :for tloss = 0
                  :do (progn
                        (loop :for i :from 0 :below ($count *mnist-train-image-batches*)
                              :for z = (samplez)
                              :for g = (generate z)
                              :for g-fake = (discriminate g)
                              :for g-loss = ($neg ($mean ($log g-fake)))
                              :do (progn
                                    (incf tloss ($data g-loss))
                                    ($adgd! *generator*)
                                    ($cg! *discriminator*)))
                        (when (eq k *k*) (prn "GL:" (/ tloss *k*)))
                        (gcf)))))
