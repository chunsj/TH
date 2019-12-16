(defpackage :resnet152-example
  (:use #:common-lisp
        #:mu
        #:th
        #:th.m.resnet152
        #:th.m.imagenet
        #:th.image))

(in-package :resnet152-example)

;; load weights - takes some time
(defparameter *resnet152-weights* (read-resnet152-weights))
(defparameter *resnet152-function* (resnet152 :all *resnet152-weights*))

;; cat categorizing - input should be 3x224x224 RGB image
(let* ((rgb (tensor-from-png-file "data/cat.vgg16.png"))
       (x (imagenet-input rgb)))
  (let* ((resnet152-result (funcall *resnet152-function* x)))
    (prn "TOP-5 MATCHES:" (imagenet-top5-matches resnet152-result))))

(let* ((rgb (tensor-from-jpeg-file "data/cat.vgg16.jpg" :resize-dimension '(224 224)))
       (x (imagenet-input rgb)))
  (let* ((resnet152-result (funcall *resnet152-function* x)))
    (prn "TOP-5 MATCHES:" (imagenet-top5-matches resnet152-result))))

(let* ((rgb (tensor-from-jpeg-file "data/dog.vgg16.jpg" :resize-dimension '(224 224)))
       (x (imagenet-input rgb)))
  (let* ((resnet152-result (funcall *resnet152-function* x)))
    (prn "TOP-5 MATCHES:" (imagenet-top5-matches resnet152-result))))

(setf *resnet152-weights* nil)
(setf *resnet152-function* nil)
(gcf)
