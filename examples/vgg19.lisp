(defpackage :vgg19-example
  (:use #:common-lisp
        #:mu
        #:th
        #:th.m.vgg19
        #:th.image))

(in-package :vgg19-example)

;; load weights - takes some time
(defparameter *vgg19-weights* (read-vgg19-weights))
(defparameter *vgg19-function* (vgg19 :all *vgg19-weights*))

;; cat categorizing - input should be 3x224x224 BGR image
(let* ((rgb (tensor-from-png-file "data/cat.vgg16.png" :normalize nil))
       (bgr (convert-to-vgg19-input rgb)))
  (prn bgr)
  (let* ((vgg19-result (funcall *vgg19-function* bgr))
         (max-val-idx ($max vgg19-result 1))
         (category-val ($ (car max-val-idx) 0 0))
         (category-idx ($ (cadr max-val-idx) 0 0)))
    (prn "VGG19 RESULT:" vgg19-result)
    (prn "SOFTMAX:" category-val)
    (prn "CATEGORY INDEX:" category-idx)
    (prn "CATEGORY DESCRIPTION:" ($ (vgg19-categories) category-idx))))

(let* ((rgb (tensor-from-jpeg-file "data/cat.vgg16.jpg" :resize-dimension '(224 224) :normalize nil))
       (bgr (convert-to-vgg19-input rgb)))
  (prn bgr)
  (let* ((vgg19-result (funcall *vgg19-function* bgr))
         (max-val-idx ($max vgg19-result 1))
         (category-val ($ (car max-val-idx) 0 0))
         (category-idx ($ (cadr max-val-idx) 0 0)))
    (prn "VGG19 RESULT:" vgg19-result)
    (prn "SOFTMAX:" category-val)
    (prn "CATEGORY INDEX:" category-idx)
    (prn "CATEGORY DESCRIPTION:" ($ (vgg19-categories) category-idx))))

(let* ((rgb (tensor-from-jpeg-file "data/dog.vgg16.jpg" :resize-dimension '(224 224) :normalize nil))
       (bgr (convert-to-vgg19-input rgb)))
  (prn bgr)
  (let* ((vgg19-result (funcall *vgg19-function* bgr))
         (max-val-idx ($max vgg19-result 1))
         (category-val ($ (car max-val-idx) 0 0))
         (category-idx ($ (cadr max-val-idx) 0 0)))
    (prn "VGG19 RESULT:" vgg19-result)
    (prn "SOFTMAX:" category-val)
    (prn "CATEGORY INDEX:" category-idx)
    (prn "CATEGORY DESCRIPTION:" ($ (vgg19-categories) category-idx))))
