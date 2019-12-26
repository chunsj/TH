(defpackage :vgg19-example
  (:use #:common-lisp
        #:mu
        #:th
        #:th.m.vgg19
        #:th.m.imagenet
        #:th.image))

(in-package :vgg19-example)

;; load weights - takes some time
(defparameter *vgg19-weights* (read-vgg19-weights))
(defparameter *vgg19-function* (vgg19 :all *vgg19-weights*))

;; cat categorizing - input should be 3x224x224 BGR image
(let* ((rgb (tensor-from-png-file "data/cat.vgg16.png"))
       (bgr (imagenet-input rgb)))
  (let ((vgg19-result (funcall *vgg19-function* bgr)))
    (prn "TOP-5 MATCHES:" (imagenet-top5-matches vgg19-result))))

(let* ((rgb (tensor-from-jpeg-file "data/cat.vgg16.jpg" :resize-dimension '(224 224)))
       (bgr (imagenet-input rgb)))
  (let ((vgg19-result (funcall *vgg19-function* bgr)))
    (prn "TOP-5 MATCHES:" (imagenet-top5-matches vgg19-result))))

(let* ((rgb (tensor-from-jpeg-file "data/dog.vgg16.jpg" :resize-dimension '(224 224)))
       (bgr (imagenet-input rgb)))
  (let ((vgg19-result (funcall *vgg19-function* bgr)))
    (prn "TOP-5 MATCHES:" (imagenet-top5-matches vgg19-result))))

(setf *vgg19-weights* nil)
(setf *vgg19-function* nil)
(gcf)
