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
  (let* ((vgg19-result (funcall *vgg19-function* bgr))
         (max-val-idx ($max vgg19-result 1))
         (category-val ($ (car max-val-idx) 0 0))
         (category-idx ($ (cadr max-val-idx) 0 0)))
    (prn "VGG19 RESULT:" vgg19-result)
    (prn "SOFTMAX:" category-val)
    (prn "CATEGORY INDEX:" category-idx)
    (prn "CATEGORY DESCRIPTION:" ($ (imagenet-categories) category-idx))))

(let* ((rgb (tensor-from-jpeg-file "data/cat.vgg16.jpg" :resize-dimension '(224 224)))
       (bgr (imagenet-input rgb)))
  (let* ((vgg19-result (funcall *vgg19-function* bgr))
         (max-val-idx ($max vgg19-result 1))
         (category-val ($ (car max-val-idx) 0 0))
         (category-idx ($ (cadr max-val-idx) 0 0)))
    (prn "VGG19 RESULT:" vgg19-result)
    (prn "SOFTMAX:" category-val)
    (prn "CATEGORY INDEX:" category-idx)
    (prn "CATEGORY DESCRIPTION:" ($ (imagenet-categories) category-idx))))

(let* ((rgb (tensor-from-jpeg-file "data/dog.vgg16.jpg" :resize-dimension '(224 224)))
       (bgr (imagenet-input rgb)))
  (let* ((vgg19-result (funcall *vgg19-function* bgr))
         (max-val-idx ($max vgg19-result 1))
         (category-val ($ (car max-val-idx) 0 0))
         (category-idx ($ (cadr max-val-idx) 0 0)))
    (prn "VGG19 RESULT:" vgg19-result)
    (prn "SOFTMAX:" category-val)
    (prn "CATEGORY INDEX:" category-idx)
    (prn "CATEGORY DESCRIPTION:" ($ (imagenet-categories) category-idx))))

(setf *vgg19-weights* nil)
(setf *vgg19-function* nil)
(gcf)
