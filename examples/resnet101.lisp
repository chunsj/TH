(defpackage :resnet101-example
  (:use #:common-lisp
        #:mu
        #:th
        #:th.m.resnet101
        #:th.m.imagenet
        #:th.image))

(in-package :resnet101-example)

;; load weights - takes some time
(defparameter *resnet101-weights* (read-resnet101-weights))
(defparameter *resnet101-function* (resnet101 :all *resnet101-weights*))

;; cat categorizing - input should be 3x224x224 RGB image
(let* ((rgb (tensor-from-png-file "data/cat.vgg16.png"))
       (x (imagenet-input rgb)))
  (let* ((resnet101-result (funcall *resnet101-function* x))
         (max-val-idx ($max resnet101-result 1))
         (category-val ($ (car max-val-idx) 0 0))
         (category-idx ($ (cadr max-val-idx) 0 0)))
    (prn "SOFTMAX:" category-val)
    (prn "CATEGORY INDEX:" category-idx)
    (prn "CATEGORY DESCRIPTION:" ($ (imagenet-categories) category-idx))
    (prn "TOP-5 MATCHES:" (imagenet-top5-matches resnet101-result))))

(let* ((rgb (tensor-from-jpeg-file "data/cat.vgg16.jpg" :resize-dimension '(224 224)))
       (x (imagenet-input rgb)))
  (let* ((resnet101-result (funcall *resnet101-function* x))
         (max-val-idx ($max resnet101-result 1))
         (category-val ($ (car max-val-idx) 0 0))
         (category-idx ($ (cadr max-val-idx) 0 0)))
    (prn "SOFTMAX:" category-val)
    (prn "CATEGORY INDEX:" category-idx)
    (prn "CATEGORY DESCRIPTION:" ($ (imagenet-categories) category-idx))
    (prn "TOP-5 MATCHES:" (imagenet-top5-matches resnet101-result))))

(let* ((rgb (tensor-from-jpeg-file "data/dog.vgg16.jpg" :resize-dimension '(224 224)))
       (x (imagenet-input rgb)))
  (let* ((resnet101-result (funcall *resnet101-function* x))
         (max-val-idx ($max resnet101-result 1))
         (category-val ($ (car max-val-idx) 0 0))
         (category-idx ($ (cadr max-val-idx) 0 0)))
    (prn "SOFTMAX:" category-val)
    (prn "CATEGORY INDEX:" category-idx)
    (prn "CATEGORY DESCRIPTION:" ($ (imagenet-categories) category-idx))
    (prn "TOP-5 MATCHES:" (imagenet-top5-matches resnet101-result))))

(setf *resnet101-weights* nil)
(setf *resnet101-function* nil)
(gcf)
