(defpackage :th.m.densenet161
  (:use #:common-lisp
        #:mu
        #:th
        #:th.image
        #:th.m.imagenet))

(in-package :th.m.densenet161)

;;(defparameter +model-location+ ($concat (namestring (user-homedir-pathname)) ".th.models"))
(defparameter +model-location+ ($concat (namestring (user-homedir-pathname)) "Desktop"))

(defun read-text-weight-file (wn &optional (readp t))
  (when readp
    (let ((f (file.disk (format nil "~A/densenet161/densenet161-~A.txt" +model-location+ wn) "r"))
          (tx (tensor)))
      ($fread tx f)
      ($fclose f)
      tx)))

(defun read-weight-file (wn &optional (readp t))
  (when readp
    (let ((f (file.disk (format nil "~A/densenet161/densenet161-~A.dat" +model-location+ wn) "r"))
          (tx (tensor)))
      (setf ($fbinaryp f) t)
      ($fread tx f)
      ($fclose f)
      tx)))

(defun kw (str) (values (intern (string-upcase str) "KEYWORD")))

(defun read-densenet161-text-weights ()
  (append (loop :for i :from 0 :to 4
                :for nm = (format nil "p~A" i)
                :for k = (kw nm)
                :append (list k (read-text-weight-file nm)))
          (loop :for i :from 1 :to 161
                :for nm = (format nil "v~A" i)
                :for k = (kw nm)
                :append (list k (read-text-weight-file nm)))
          (loop :for i :from 1 :to 161
                :for nm = (format nil "m~A" i)
                :for k = (kw nm)
                :append (list k (read-text-weight-file nm)))))

(defun w (w wn) (getf w wn))

(w (read-densenet161-text-weights) :m1)

(defun input-blk (x ws)
  (-> x
      ($conv2d (w ws :p0) nil 2 2 3 3)
      ($bn (w ws :p1) (w ws :p2) (w ws :m1) (w ws :v1))
      ($relu)
      ($dlmaxpool2d 3 3 2 2 1 1 1 1)))

(let* ((rgb (tensor-from-png-file "data/cat.vgg16.png"))
       (x (imagenet-input rgb))
       (input (apply #'$reshape x (cons 1 ($size x))))
       (ws (read-densenet161-text-weights)))
  (-> input
      (input-blk ws)))
