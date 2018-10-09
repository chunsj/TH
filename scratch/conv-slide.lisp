(defpackage :convolution-sliding
  (:use #:common-lisp
        #:mu
        #:th
        #:th.image
        #:th.m.vgg16
        #:th.m.imagenet))

;; XXX can i make fully connected weight as convolution kernel?
;; need to check computation index and reference between them

(in-package :convolution-sliding)

(defparameter *bn* 2)
(defparameter *vinput* ($* 0.01 (rndn *bn* 3 28 28)))

(defparameter *k1* ($* 0.01 (rndn 16 3 5 5)))
(defparameter *k2* ($* 0.01 (rndn 400 16 5 5)))
(defparameter *k3* ($* 0.01 (rndn 400 400 1 1)))
(defparameter *k4* ($* 0.01 (rndn 4 400 1 1)))

(defparameter *w3* ($* 0.01 (rndn 400 400)))
(defparameter *b3* (zeros 1 400))
(defparameter *w4* ($* 0.01 (rndn 400 400)))
(defparameter *b4* (zeros 1 400))
(defparameter *w5* ($* 0.01 (rndn 400 4)))
(defparameter *b5* (zeros 1 4))

(prn (-> *vinput*
         ($conv2d *k1*)
         ($relu)
         ($maxpool2d 2 2 2 2)
         ($reshape *bn* 400)
         ($affine *w3* *b3*)
         ($relu)
         ($affine *w4* *b4*)
         ($relu)
         ($affine *w5* *b5*)
         ($softmax)))

;; for larger image case - result from this should be the same one as later code
(prn (-> *vinput*
         ($subview 0 1 0 3 0 16 0 14)
         ($conv2d *k1*)
         ($relu)
         ($maxpool2d 2 2 2 2)
         ($conv2d *k2*)
         ($relu)
         ($conv2d *k3*)
         ($relu)
         ($conv2d *k4*)
         ($permute 0 2 3 1)
         ($reshape 2 4)
         ;;($reshape 1 4)
         ;;($reshape 64 4)
         ;;($softmax)
         ))

;; for original size - this should match above results
(prn (-> *vinput*
         ($subview 0 1 0 3 0 14 0 14)
         ($conv2d *k1*)
         ($relu)
         ($maxpool2d 2 2 2 2)
         ($conv2d *k2*)
         ($relu)
         ($conv2d *k3*)
         ($relu)
         ($conv2d *k4*)
         ($permute 0 2 3 1)
         ($reshape 1 4)
         ;;($reshape 1 4)
         ;;($reshape 64 4)
         ;;($softmax)
         ))

;; vertically one-stride(2) lower (from maxpooling
(prn (-> *vinput*
         ($subview 0 1 0 3 2 14 0 14)
         ($conv2d *k1*)
         ($relu)
         ($maxpool2d 2 2 2 2)
         ($conv2d *k2*)
         ($relu)
         ($conv2d *k3*)
         ($relu)
         ($conv2d *k4*)
         ($permute 0 2 3 1)
         ($reshape 1 4)
         ;;($reshape 1 4)
         ;;($reshape 64 4)
         ($softmax)
         ))

($index *vinput* 0 1)
($subview *vinput* 0 2 0 3 0 14 0 14)

;; when input is 4-D 2nd dimension is used for softmax
(prn (-> *vinput*
         ($subview 0 1 0 3 2 14 0 14)
         ($conv2d *k1*)
         ($relu)
         ($maxpool2d 2 2 2 2)
         ($conv2d *k2*)
         ($relu)
         ($conv2d *k3*)
         ($relu)
         ($conv2d *k4*)
         ($softmax)
         ($max 1)
         (cadr)
         ($squeeze)))

(prn (-> *vinput*
         ;;($subview 0 2 0 3 0 16 0 16)
         ;; without above subviewing 8x8 slide-windows emit each results
         ($conv2d *k1*)
         ($relu)
         ($maxpool2d 2 2 2 2)
         ($conv2d *k2*)
         ($relu)
         ($conv2d *k3*)
         ($relu)
         ($conv2d *k4*)
         ($softmax)
         ($max 1)
         (cadr)
         ($squeeze)))

;; vgg16
(defparameter *vgg16-weights* (read-vgg16-weights))

;; it seems this is right, of course we need to test
(let ((k14 ($reshape ($transpose (getf *vgg16-weights* :w14)) 4096 512 7 7))
      (b14 ($squeeze (getf *vgg16-weights* :b14)))
      (k15 ($reshape ($transpose (getf *vgg16-weights* :w15)) 4096 4096 1 1))
      (b15 ($squeeze (getf *vgg16-weights* :b15)))
      (k16 ($reshape ($transpose (getf *vgg16-weights* :w16)) 1000 4096 1 1))
      (b16 ($squeeze (getf *vgg16-weights* :b16)))
      (x (rndn 1 512 7 7)))
  ($ k14 0 0)
  (prn k14)
  (prn b14)
  (prn k15)
  (prn b15)
  (prn k16)
  (prn b16)
  (prn (-> x
           ($conv2d k14 b14 7 7)
           ($conv2d k15 b15)
           ($conv2d k16 b16))))

(let* ((rgb (tensor-from-png-file "data/cat.vgg16.png"))
       (x (imagenet-input rgb))
       (f (vgg16 :all *vgg16-weights*)))
  (prn (funcall f x)))

(let* ((rgb (tensor-from-png-file "data/cat.vgg16.png"))
       (x (imagenet-input rgb))
       (f (vgg16fcn *vgg16-weights*)))
  (prn (funcall f x)))
