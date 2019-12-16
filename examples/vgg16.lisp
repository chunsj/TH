(defpackage :vgg16-example
  (:use #:common-lisp
        #:mu
        #:th
        #:th.m.vgg16
        #:th.m.imagenet
        #:th.image))

(in-package :vgg16-example)

;; load weights - takes some time
(defparameter *vgg16-weights* (read-vgg16-weights))
(defparameter *vgg16-function* (vgg16 :all *vgg16-weights*))

;; cat categorizing - input should be 3x224x224 RGB image
(let* ((rgb (tensor-from-png-file "data/cat.vgg16.png"))
       (x (imagenet-input rgb)))
  (let ((vgg16-result (funcall *vgg16-function* x)))
    (prn "TOP-5 MATCHES:" (imagenet-top5-matches vgg16-result))))

(let* ((rgb (tensor-from-jpeg-file "data/cat.vgg16.jpg" :resize-dimension '(224 224)))
       (x (imagenet-input rgb)))
  (let ((vgg16-result (funcall *vgg16-function* x)))
    (prn "TOP-5 MATCHES:" (imagenet-top5-matches vgg16-result))))

(let* ((rgb (tensor-from-jpeg-file "data/dog.vgg16.jpg" :resize-dimension '(224 224)))
       (x (imagenet-input rgb)))
  (let ((vgg16-result (funcall *vgg16-function* x)))
    (prn "TOP-5 MATCHES:" (imagenet-top5-matches vgg16-result))))

(setf *vgg16-weights* nil)
(setf *vgg16-function* nil)
(gcf)
