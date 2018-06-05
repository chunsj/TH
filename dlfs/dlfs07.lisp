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

;; constructing network - smaller samples and single step only
(let* ((indices '(0 1 2 3 4))
       (nbatch ($count indices))
       (nch 1)
       (nfilter 30)
       (imgw 28)
       (imgh 28)
       (kw 5)
       (kh 5)
       (pw 2)
       (ph 2)
       (nl2 100)
       (nl3 10)
       (k (-> (mkfilter nfilter nch kw kh)
              ($uniform! 0 1)
              ($div (sqrt (/ 2.0 (* imgw imgh))))
              ($variable)))
       (bk (-> (mkfbias nfilter)
               ($fill! 0)
               ($variable)))
       (w2 ($variable (rndn (* nfilter 12 12) nl2)))
       (b2 ($variable (zeros nl2)))
       (w3 ($variable (rndn nl2 nl3)))
       (b3 ($variable (rndn nl3)))
       (x ($constant ($index ($ *mnist* :train-images) 0 indices)))
       (y ($constant ($index ($ *mnist* :train-labels) 0 indices)))
       (c ($conv2d ($reshape x nbatch nch imgw imgh) k bk))
       (l1 ($relu c))
       (p1 ($maxpool2d l1 pw ph 2 2))
       (o1 ($reshape p1 nbatch (* nfilter 12 12)))
       (z2 ($xwpb o1 w2 b2))
       (l2 ($relu z2))
       (z3 ($xwpb l2 w3 b3))
       (l3 ($softmax z3))
       (er ($cee l3 y)))
  (print er))
