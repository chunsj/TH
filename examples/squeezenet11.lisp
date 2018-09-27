(defpackage :squeezenet11-example
  (:use #:common-lisp
        #:mu
        #:th
        #:th.m.squeezenet11
        #:th.m.imagenet
        #:th.image))

(in-package :squeezenet11-example)

;; load weights - takes some time
(defparameter *squeezenet11-weights* (read-squeezenet11-weights))
(defparameter *squeezenet11-function* (squeezenet11 *squeezenet11-weights*))

;; cat categorizing - input should be 3x224x224 RGB image
(let* ((rgb (tensor-from-png-file "data/cat.vgg16.png"))
       (x (imagenet-input rgb)))
  (prn x)
  (let ((squeezenet11-result (funcall *squeezenet11-function* x)))
    (prn "SQUEEZENET11 RESULT:" squeezenet11-result)
    (prn "TOP-5 MATCHES:" (imagenet-top5-matches squeezenet11-result))))

(let* ((rgb (tensor-from-jpeg-file "data/cat.vgg16.jpg" :resize-dimension '(224 224)))
       (x (imagenet-input rgb)))
  (prn x)
  (let ((squeezenet11-result (funcall *squeezenet11-function* x)))
    (prn "SQUEEZENET11 RESULT:" squeezenet11-result)
    (prn "TOP-5 MATCHES:" (imagenet-top5-matches squeezenet11-result))))

(let* ((rgb (tensor-from-jpeg-file "data/dog.vgg16.jpg" :resize-dimension '(224 224)))
       (x (imagenet-input rgb)))
  (prn x)
  (let ((squeezenet11-result (funcall *squeezenet11-function* x)))
    (prn "SQUEEZENET11 RESULT:" squeezenet11-result)
    (prn "TOP-5 MATCHES:" (imagenet-top5-matches squeezenet11-result))))

(setf *squeezenet11-weights* nil)
(setf *squeezenet11-function* nil)
(gcf)
