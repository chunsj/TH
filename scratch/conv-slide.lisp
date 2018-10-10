(defpackage :convolution-sliding
  (:use #:common-lisp
        #:mu
        #:th
        #:th.image
        #:th.m.vgg16
        #:th.m.vgg19
        #:th.m.squeezenet11
        #:th.m.imagenet))

;; XXX can i make fully connected weight as convolution kernel?
;; need to check computation index and reference between them

(in-package :convolution-sliding)

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

(let* (;;(rgb (tensor-from-png-file "data/cat.vgg16.png"))
       ;;(rgb (tensor-from-jpeg-file "data/cat.vgg16.jpg"))
       ;;(rgb (tensor-from-jpeg-file "data/cat.vgg16.jpg" :resize-dimension '(224 224)))
       (rgb (tensor-from-jpeg-file "data/dog.vgg16.jpg"))
       ;;(rgb (tensor-from-jpeg-file "data/dog.vgg16.jpg" :resize-dimension '(224 224)))
       (x (imagenet-input rgb t))
       (f (vgg16fcn *vgg16-weights*))
       (m (funcall f x)))
  (prn (imagenet-top5-matches m)))
