(defpackage :cats-and-dogs-data-work
  (:use #:common-lisp
        #:mu
        #:th
        #:th.image))

(in-package :cats-and-dogs-data-work)

(defparameter +cnd-path+ ($concat (namestring (user-homedir-pathname))
                                  ".th/datasets/cats-and-dogs/"))

(loop :for idx :from 1 :to 25
      :do (let* ((nd 500)
                 (dataset-cat (tensor.byte nd 3 64 64))
                 (dataset-dog (tensor.byte nd 3 64 64))
                 (offset (* nd (1- idx))))
            (loop :for i :from (+ offset 1) :below (+ nd offset)
                  :for fname-cat = (format nil "~A/train/cat.~A.jpg" +cnd-path+ i)
                  :for fname-dog = (format nil "~A/train/dog.~A.jpg" +cnd-path+ i)
                  :for tx-cat = (tensor-from-jpeg-file fname-cat
                                                       :resize-dimension '(64 64)
                                                       :normalize nil)
                  :for tx-dog = (tensor-from-jpeg-file fname-dog
                                                       :resize-dimension '(64 64)
                                                       :normalize nil)
                  :do (progn
                        (setf ($ dataset-cat (- (1- i) offset)) tx-cat)
                        (setf ($ dataset-dog (- (1- i) offset)) tx-dog)
                        (setf tx-cat nil
                              tx-dog nil)
                        (gcf)))
            (let ((fcat (file.disk (format nil "cat64-train-~2,'0D.tensor" idx) "w"))
                  (fdog (file.disk (format nil "dog64-train-~2,'0D.tensor" idx) "w")))
              (setf ($fbinaryp fcat) t)
              (setf ($fbinaryp fdog) t)
              ($fwrite dataset-cat fcat)
              ($fwrite dataset-dog fdog)
              ($fclose fcat)
              ($fclose fdog))
            (setf dataset-cat nil
                  dataset-dog nil)
            (gcf)))
