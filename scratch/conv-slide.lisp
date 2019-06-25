(defpackage :convolution-sliding
  (:use #:common-lisp
        #:mu
        #:th
        #:th.image
        #:th.m.vgg16
        #:th.m.vgg19
        #:th.m.squeezenet11
        #:th.m.resnet50
        #:th.m.densenet161
        #:th.m.imagenet))

;; XXX can i make fully connected weight as convolution kernel?
;; need to check computation index and reference between them

(in-package :convolution-sliding)

;; weights
(defparameter *vgg16-weights* (read-vgg16-weights))
(defparameter *vgg19-weights* (read-vgg19-weights))
(defparameter *squeezenet11-weights* (read-squeezenet11-weights))
(defparameter *resnet50-weights* (read-resnet50-weights))
(defparameter *densenet161-weights* (read-densenet161-weights))

(let* ((rgb (tensor-from-png-file "data/cat.vgg16.png"))
       (x (imagenet-input rgb))
       (f (vgg16 :all *vgg16-weights*))
       (m (funcall f x)))
  (prn (imagenet-top5-matches m)))

(let* ((rgb (tensor-from-png-file "data/cat.vgg16.png"))
       (x (imagenet-input rgb))
       (f (vgg19 :all *vgg19-weights*))
       (m (funcall f x)))
  (prn (imagenet-top5-matches m)))

(let* ((rgb (tensor-from-png-file "data/cat.vgg16.png"))
       (x (imagenet-input rgb))
       (f (resnet50 :all *resnet50-weights*))
       (m (funcall f x)))
  (prn (imagenet-top5-matches m)))

(let* (;;(rgb (tensor-from-png-file "data/cat.vgg16.png"))
       ;;(rgb (tensor-from-jpeg-file "data/cat.vgg16.jpg"))
       ;;(rgb (tensor-from-jpeg-file "data/cat.vgg16.jpg" :resize-dimension '(224 224)))
       ;;(rgb (tensor-from-jpeg-file "data/dog.vgg16.jpg"))
       (rgb (tensor-from-jpeg-file "data/dog.vgg16.jpg" :resize-dimension '(224 224)))
       (x (imagenet-input rgb t))
       (f (vgg19fcn *vgg19-weights*))
       (m (funcall f x)))
  (prn (imagenet-top5-matches m)))

(let* (;;(rgb (tensor-from-png-file "data/cat.vgg16.png"))
       ;;(rgb (tensor-from-jpeg-file "data/cat.vgg16.jpg"))
       (rgb (tensor-from-jpeg-file "data/cat.vgg16.jpg" :resize-dimension '(224 224)))
       (x (imagenet-input rgb t))
       (f (squeezenet11fcn *squeezenet11-weights*))
       (m (funcall f x)))
  (prn (imagenet-top5-matches m))
  (prn ($size m)))

(let* ((rgb (tensor-from-png-file "data/cat.vgg16.png"))
       ;;(rgb (tensor-from-jpeg-file "data/cat.vgg16.jpg"))
       ;;(rgb (tensor-from-jpeg-file "data/cat.vgg16.jpg" :resize-dimension '(224 224)))
       (x (imagenet-input rgb t))
       (f (resnet50fcn *resnet50-weights*))
       (m (funcall f x)))
  (prn (imagenet-top5-matches m))
  (prn ($size m)))

(let* (;;(rgb (tensor-from-png-file "data/cat.vgg16.png"))
       (rgb (tensor-from-jpeg-file "data/cat.vgg16.jpg"))
       ;;(rgb (tensor-from-jpeg-file "data/cat.vgg16.jpg" :resize-dimension '(224 224)))
       (x (imagenet-input rgb t))
       (f (densenet161fcn *densenet161-weights*))
       (m (funcall f x)))
  (prn (imagenet-top5-matches m))
  (prn ($size m)))

;; sometimes we need to check output dimenstion for safety
(let* ((rgb (tensor-from-png-file "data/cat.vgg16.png"))
       (x (imagenet-input rgb))
       (f (densenet161fcn *densenet161-weights*))
       (m (funcall f x)))
  (prn (imagenet-top5-matches m)))

(setf *vgg16-weights* nil)
(setf *vgg19-weights* nil)
(setf *squeezenet11-weights* nil)
(setf *resnet50-weights* nil)
(setf *densenet161-weights* nil)
(gcf)
