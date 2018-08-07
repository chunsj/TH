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

(defparameter *gan* (parameters))

;; discriminator network
(defparameter *dw1* ($parameter *gan* (vxavier (list 784 128))))
(defparameter *db1* ($parameter *gan* (zeros 1 128)))
(defparameter *dw2* ($parameter *gan* (vxavier (list 128 1))))
(defparameter *db2* ($parameter *gan* (zeros 1 1)))

;; generator network
(defparameter *gw1* ($parameter *gan* (vxavier (list 100 128))))
(defparameter *gb1* ($parameter *gan* (zeros 1 128)))
(defparameter *gw2* ($parameter *gan* (vxavier (list 128 784))))
(defparameter *gb2* ($parameter *gan* (zeros 1 784)))

(defparameter *os* (ones *batch-size* 1))

(defun generator (z)
  (let* ((gh1 ($relu ($+ ($@ z *gw1*) ($@ ($constant *os*) *gb1*))))
         (glogprob ($+ ($@ gh1 *gw2*) ($@ ($constant *os*) *gb2*))))
    ($sigmoid glogprob)))

(defun discriminator-logit (x)
  (let* ((dh1 ($relu ($+ ($@ x *dw1*) ($@ ($constant *os*) *db1*))))
         (dlogit ($+ ($@ dh1 *dw2*) ($@ ($constant *os*) *db2*))))
    dlogit))

(defun discriminator (logit) ($sigmoid logit))
