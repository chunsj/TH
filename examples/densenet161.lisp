(defpackage :densenet161-example
  (:use #:common-lisp
        #:mu
        #:th
        #:th.m.densenet161
        #:th.m.imagenet
        #:th.image))

(in-package :densenet161-example)

;; load weights - takes some time
(defparameter *densenet161-weights* (read-densenet161-weights))
(defparameter *densenet161-function* (densenet161 :all *densenet161-weights*))

;; cat categorizing - input should be 3x224x224 RGB image
(let* ((rgb (tensor-from-png-file "data/cat.vgg16.png"))
       (x (imagenet-input rgb)))
  (let ((densenet161-result (funcall *densenet161-function* x)))
    (prn "TOP-5 MATCHES:" (imagenet-top5-matches densenet161-result))))

(let* ((rgb (tensor-from-jpeg-file "data/cat.vgg16.jpg" :resize-dimension '(224 224)))
       (x (imagenet-input rgb)))
  (let ((densenet161-result (funcall *densenet161-function* x)))
    (prn "TOP-5 MATCHES:" (imagenet-top5-matches densenet161-result))))

(let* ((rgb (tensor-from-jpeg-file "data/dog.vgg16.jpg" :resize-dimension '(224 224)))
       (x (imagenet-input rgb)))
  (let ((densenet161-result (funcall *densenet161-function* x)))
    (prn "TOP-5 MATCHES:" (imagenet-top5-matches densenet161-result))))

(setf *densenet161-weights* nil)
(setf *densenet161-function* nil)
(gcf)

(w :p0)
