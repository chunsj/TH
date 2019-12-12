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
  (let* ((vgg16-result (funcall *vgg16-function* x))
         (max-val-idx ($max vgg16-result 1))
         (category-val ($ (car max-val-idx) 0 0))
         (category-idx ($ (cadr max-val-idx) 0 0)))
    (prn "VGG16 RESULT:" vgg16-result)
    (prn "SOFTMAX:" category-val)
    (prn "CATEGORY INDEX:" category-idx)
    (prn "CATEGORY DESCRIPTION:" ($ (imagenet-categories) category-idx))))

(let* ((rgb (tensor-from-jpeg-file "data/cat.vgg16.jpg" :resize-dimension '(224 224)))
       (x (imagenet-input rgb)))
  (let* ((vgg16-result (funcall *vgg16-function* x))
         (max-val-idx ($max vgg16-result 1))
         (category-val ($ (car max-val-idx) 0 0))
         (category-idx ($ (cadr max-val-idx) 0 0)))
    (prn "VGG16 RESULT:" vgg16-result)
    (prn "SOFTMAX:" category-val)
    (prn "CATEGORY INDEX:" category-idx)
    (prn "CATEGORY DESCRIPTION:" ($ (imagenet-categories) category-idx))))

(let* ((rgb (tensor-from-jpeg-file "data/dog.vgg16.jpg" :resize-dimension '(224 224)))
       (x (imagenet-input rgb)))
  (let* ((vgg16-result (funcall *vgg16-function* x))
         (max-val-idx ($max vgg16-result 1))
         (category-val ($ (car max-val-idx) 0 0))
         (category-idx ($ (cadr max-val-idx) 0 0)))
    (prn "VGG16 RESULT:" vgg16-result)
    (prn "SOFTMAX:" category-val)
    (prn "CATEGORY INDEX:" category-idx)
    (prn "CATEGORY DESCRIPTION:" ($ (imagenet-categories) category-idx))))

(setf *vgg16-weights* nil)
(setf *vgg16-function* nil)
(gcf)
