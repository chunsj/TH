(defpackage th.m.imagenet
  (:use #:common-lisp
        #:mu
        #:th)
  (:export #:imagenet-categories))

(in-package th.m.imagenet)

(defparameter +model-location+ ($concat (namestring (user-homedir-pathname)) ".th.models"))

(defun imagenet-categories ()
  (with-open-file (in (format nil "~A/imagenet/categories.txt" +model-location+) :direction :input)
    (coerce (loop :for i :from 0 :below 1000
                  :for line = (read-line in nil)
                  :for catn = (subseq line 0 9)
                  :for desc = (subseq line 10)
                  :collect (list catn desc))
            'vector)))
