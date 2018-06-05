(defpackage :dlfs-07
  (:use #:common-lisp
        #:mu
        #:th
        #:th.db.mnist))

(in-package :dlfs-07)

;; prepare data for later use, it takes some time to load
(defparameter *mnist* (read-mnist-data))
(print *mnist*)

(defparameter *mnist-train-image-batches*
  (loop :for i :from 0 :below 6
        :for rng = (loop :for k :from (* i 10000) :below (* (1+ i) 10000)
                         :collect k)
        :collect ($contiguous! ($index ($ *mnist* :train-images) 0 rng))))

(defparameter *mnist-train-label-batches*
  (loop :for i :from 0 :below 6
        :for rng = (loop :for k :from (* i 10000) :below (* (1+ i) 10000)
                         :collect k)
        :collect ($contiguous! ($index ($ *mnist* :train-labels) 0 rng))))

;; run convolution with mnist data
(let* ((indices '(0 1 2 3 4))
       (nbatch ($count indices))
       (nch 1)
       (x ($index ($ *mnist* :train-images) 0 indices))
       (x ($reshape x nbatch nch 28 28))
       (k (tensor '((((1 1 1) (1 1 1) (1 1 1))))))
       (b (tensor '(1))))
  (print x)
  (print k)
  (print ($conv2d x k b)))

;; more systematic
(defun mkfilter (fn nc kw kh) (tensor fn nc kw kh))
(defun mkfbias (fn) (tensor fn))

;; use helper functions
(let* ((indices '(0 1 2 3 4))
       (nbatch ($count indices))
       (nfilter 1)
       (x ($index ($ *mnist* :train-images) 0 indices))
       (nch 1)
       (x ($reshape x nbatch nch 28 28))
       (k (-> (mkfilter nfilter nch 3 3) ($fill! 1)))
       (b (-> (mkfbias nfilter) ($fill! 1))))
  (print x)
  (print k)
  (print ($conv2d x k b)))

;; with max pooling
(let* ((indices '(0 1 2 3 4))
       (nbatch ($count indices))
       (nfilter 1)
       (x ($index ($ *mnist* :train-images) 0 indices))
       (nch 1)
       (x ($reshape x nbatch nch 28 28))
       (k (-> (mkfilter nfilter nch 3 3) ($fill! 1)))
       (b (-> (mkfbias nfilter) ($fill! 1)))
       (c ($conv2d x k b))
       (p ($maxpool2d c 2 2)))
  (print c)
  (print p))
