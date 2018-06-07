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
  (loop :for i :from 0 :below 60
        :for rng = (loop :for k :from (* i 1000) :below (* (1+ i) 1000)
                         :collect k)
        :collect ($contiguous! ($index ($ *mnist* :train-images) 0 rng))))

(defparameter *mnist-train-label-batches*
  (loop :for i :from 0 :below 60
        :for rng = (loop :for k :from (* i 1000) :below (* (1+ i) 1000)
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
(let* ((indices '(0 1 2 3 4 5 6 7 8 9))
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

;; with full data
(let* ((indices (loop :for i :from 0 :below 1000 :collect i))
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
       (c (time ($conv2d ($reshape x nbatch nch imgw imgh) k bk)))
       (l1 ($relu c))
       (p1 (time ($maxpool2d l1 pw ph 2 2)))
       (o1 (time ($reshape p1 nbatch (* nfilter 12 12))))
       (z2 (time ($xwpb o1 w2 b2)))
       (l2 ($relu z2))
       (z3 (time ($xwpb l2 w3 b3)))
       (l3 ($softmax z3))
       (er ($cee l3 y)))
  (print er))

;; checking convolution speed
(let* ((indices (loop :for i :from 0 :below 10000 :collect i))
       (nbatch ($count indices))
       (nfilter 30)
       (x ($index ($ *mnist* :train-images) 0 indices))
       (nch 1)
       (x ($reshape x nbatch nch 28 28))
       (k (-> (mkfilter nfilter nch 3 3) ($fill! 1)))
       (b (-> (mkfbias nfilter) ($fill! 1))))
  (print (time ($conv2d x k b))))

(defparameter *filter-number* 30)
(defparameter *channel-number* 1)
(defparameter *filter-width* 5)
(defparameter *filter-height* 5)
(defparameter *pool-width* 2)
(defparameter *pool-height* 2)
(defparameter *pool-stride-width* 2)
(defparameter *pool-stride-height* 2)
(defparameter *pool-out-width* 12)
(defparameter *pool-out-height* 12)
(defparameter *l2-output* 100)
(defparameter *l3-output* 10)
(defparameter *k* (-> (mkfilter *filter-number* *channel-number*
                                *filter-width* *filter-height*)
                      ($uniform! 0 0.01)
                      ($variable)))
(defparameter *kb* (-> (mkfbias *filter-number*)
                       ($zero!)
                       ($variable)))
(defparameter *w2* (-> (rnd (* *filter-number* *pool-out-width* *pool-out-height*)
                            *l2-output*)
                       ($mul! 0.01)
                       ($variable)))
(defparameter *b2* (-> (zeros *l2-output*)
                       ($variable)))
(defparameter *w3* (-> (rnd *l2-output* *l3-output*)
                       ($mul! 0.01)
                       ($variable)))
(defparameter *b3* (-> (zeros *l3-output*)
                       ($variable)))

(defun mnist-write-weight-to (w fname)
  (let ((f (file.disk fname "w")))
    ($fwrite ($data w) f)
    ($fclose f)))

(defun mnist-cnn-write-weights ()
  (mnist-write-weight-to *k* "dlfs/mnist-cnn-k.dat")
  (mnist-write-weight-to *kb* "dlfs/mnist-cnn-kb.dat")
  (mnist-write-weight-to *w2* "dlfs/mnist-cnn-w2.dat")
  (mnist-write-weight-to *b2* "dlfs/mnist-cnn-b2.dat")
  (mnist-write-weight-to *w3* "dlfs/mnist-cnn-w3.dat")
  (mnist-write-weight-to *b3* "dlfs/mnist-cnn-b3.dat"))

(defun mnist-read-weight-from (w fname)
  (let ((f (file.disk fname "r")))
    ($fread ($data w) f)
    ($fclose f)))

(defun mnist-cnn-read-weights ()
  (mnist-read-weight-from *k* "dlfs/mnist-cnn-k.dat")
  (mnist-read-weight-from *kb* "dlfs/mnist-cnn-kb.dat")
  (mnist-read-weight-from *w2* "dlfs/mnist-cnn-w2.dat")
  (mnist-read-weight-from *b2* "dlfs/mnist-cnn-b2.dat")
  (mnist-read-weight-from *w3* "dlfs/mnist-cnn-w3.dat")
  (mnist-read-weight-from *b3* "dlfs/mnist-cnn-b3.dat"))

;; x should have been reshaped before entering
(defun mnist-predict (x)
  (-> x
      ($conv2d *k* *kb*)
      ($relu)
      ($maxpool2d *pool-width* *pool-height*
                  *pool-stride-width* *pool-stride-height*)
      ($reshape ($size x 0) (* *filter-number* *pool-out-width* *pool-out-height*))
      ($xwpb *w2* *b2*)
      ($relu)
      ($xwpb *w3* *b3*)
      ($softmax)))

;; use batches for performance
(loop :for epoch :from 1 :to 50
      :do (loop :for i :from 0 :below 100
                :for xi = ($ *mnist-train-image-batches* i)
                :for x = (-> xi
                             ($reshape ($size xi 0) *channel-number* 28 28)
                             ($constant))
                :for y = (-> ($ *mnist-train-label-batches* i)
                             ($constant))
                :for y* = (mnist-predict x)
                :for loss = ($cee y* y)
                :do (progn
                      (format t "[~A|~A]: ~A~%" (1+ i) epoch ($data loss))
                      (finish-output)
                      ($bp! loss)
                      ($agd! loss 0.01)
                      (gcf))))

;; test
(let* ((xtest ($ *mnist* :test-images))
       (ytest ($ *mnist* :test-labels)))
  (print ($data ($cee (mnist-predict (-> xtest
                                         ($reshape ($size xtest 0) 1 28 28)
                                         ($constant)))
                      ($constant ytest)))))

;; write weights
(mnist-cnn-write-weights)

;; read weights
(mnist-cnn-read-weights)
