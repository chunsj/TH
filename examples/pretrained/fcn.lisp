(defpackage :fcn-models
  (:use #:common-lisp
        #:mu
        #:th
        #:th.image
        #:th.m.imagenet
        #:th.m.vgg16
        #:th.m.vgg19
        #:th.m.resnet50
        #:th.m.densenet161
        #:th.m.squeezenet11))

(in-package :fcn-models)

(defun cats (f)
  (let* ((rgb1 (tensor-from-png-file "data/cat.vgg16.png"))
         (rgb2 (tensor-from-jpeg-file "data/cat.vgg16.jpg"))
         (x1 (imagenet-input rgb1 t))
         (x2 (imagenet-input rgb2 t))
         (m1 (funcall f x1))
         (m2 (funcall f x2)))
    (prn (imagenet-top5-matches m1))
    (prn (imagenet-top5-matches m2))))

(let* ((weights (read-vgg16-weights))
       (f (vgg16fcn weights)))
  (cats f)
  (gcf))

(let* ((weights (read-vgg19-weights))
       (f (vgg19fcn weights)))
  (cats f)
  (gcf))

(let* ((weights (read-resnet50-weights))
       (f (resnet50fcn weights)))
  (cats f)
  (gcf))

(let* ((weights (read-densenet161-weights))
       (f (densenet161fcn weights)))
  (cats f)
  (gcf))

(let* ((weights (read-squeezenet11-weights))
       (f (squeezenet11fcn weights)))
  (cats f)
  (gcf))
