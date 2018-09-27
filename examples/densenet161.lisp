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
  (prn x)
  (let* ((densenet161-result (funcall *densenet161-function* x))
         (max-val-idx ($max densenet161-result 1))
         (category-val ($ (car max-val-idx) 0 0))
         (category-idx ($ (cadr max-val-idx) 0 0)))
    (prn "DENSENET161 RESULT:" densenet161-result)
    (prn "SOFTMAX:" category-val)
    (prn "CATEGORY INDEX:" category-idx)
    (prn "CATEGORY DESCRIPTION:" ($ (imagenet-categories) category-idx))))

(let* ((rgb (tensor-from-jpeg-file "data/cat.vgg16.jpg" :resize-dimension '(224 224)))
       (x (imagenet-input rgb)))
  (prn x)
  (let* ((densenet161-result (funcall *densenet161-function* x))
         (max-val-idx ($max densenet161-result 1))
         (category-val ($ (car max-val-idx) 0 0))
         (category-idx ($ (cadr max-val-idx) 0 0)))
    (prn "DENSENET161 RESULT:" densenet161-result)
    (prn "SOFTMAX:" category-val)
    (prn "CATEGORY INDEX:" category-idx)
    (prn "CATEGORY DESCRIPTION:" ($ (imagenet-categories) category-idx))))

(let* ((rgb (tensor-from-jpeg-file "data/dog.vgg16.jpg" :resize-dimension '(224 224)))
       (x (imagenet-input rgb)))
  (prn x)
  (let* ((densenet161-result (funcall *densenet161-function* x))
         (max-val-idx ($max densenet161-result 1))
         (category-val ($ (car max-val-idx) 0 0))
         (category-idx ($ (cadr max-val-idx) 0 0)))
    (prn "DENSENET161 RESULT:" densenet161-result)
    (prn "SOFTMAX:" category-val)
    (prn "CATEGORY INDEX:" category-idx)
    (prn "CATEGORY DESCRIPTION:" ($ (imagenet-categories) category-idx))))

(setf *densenet161-weights* nil)
(setf *densenet161-function* nil)
(gcf)
