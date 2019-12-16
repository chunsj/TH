(defpackage :resnet50-example
  (:use #:common-lisp
        #:mu
        #:th
        #:th.m.resnet50
        #:th.m.imagenet
        #:th.image))

(in-package :resnet50-example)

;; load weights - takes some time
(defparameter *resnet50-weights* (read-resnet50-weights))
(defparameter *resnet50-function* (resnet50 :all *resnet50-weights*))

;; cat categorizing - input should be 3x224x224 RGB image
(let* ((rgb (tensor-from-png-file "data/cat.vgg16.png"))
       (x (imagenet-input rgb)))
  (let* ((resnet50-result (funcall *resnet50-function* x)))
    (prn "TOP-5 MATCHES:" (imagenet-top5-matches resnet50-result))))

(let* ((rgb (tensor-from-jpeg-file "data/cat.vgg16.jpg" :resize-dimension '(224 224)))
       (x (imagenet-input rgb)))
  (let* ((resnet50-result (funcall *resnet50-function* x)))
    (prn "TOP-5 MATCHES:" (imagenet-top5-matches resnet50-result))))

(let* ((rgb (tensor-from-jpeg-file "data/dog.vgg16.jpg" :resize-dimension '(224 224)))
       (x (imagenet-input rgb)))
  (let* ((resnet50-result (funcall *resnet50-function* x)))
    (prn "TOP-5 MATCHES:" (imagenet-top5-matches resnet50-result))))

(setf *resnet50-weights* nil)
(setf *resnet50-function* nil)
(gcf)
