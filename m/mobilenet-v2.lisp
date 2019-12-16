(defpackage :th.m.mobilenet-v2
  (:use #:common-lisp
        #:mu
        #:th)
  (:export #:read-mobilenet-v2-weights
           #:mobilenet-v2
           #:mobilenet-v2-fcn))

(in-package :th.m.mobilenet-v2)

(defparameter +model-location+ ($concat (namestring (user-homedir-pathname)) ".th/models"))

(defun wfname-txt (wn)
  (format nil "~A/mobilenet_v2/mobilenet_v2-~A.txt"
          +model-location+
          (string-downcase wn)))

(defun wfname-bin (wn)
  (format nil "~A/mobilenet_v2/mobilenet_v2-~A.dat"
          +model-location+
          (string-downcase wn)))

(defun read-text-weight-file (wn &optional (readp t))
  (when readp
    (let ((f (file.disk (wfname-txt wn) "r"))
          (tx (tensor)))
      ($fread tx f)
      ($fclose f)
      tx)))

(defun read-weight-file (wn &optional (readp t))
  (when readp
    (let ((f (file.disk (wfname-bin wn) "r"))
          (tx (tensor)))
      (setf ($fbinaryp f) t)
      ($fread tx f)
      ($fclose f)
      tx)))

(defun kw (str) (values (intern (string-upcase str) "KEYWORD")))
(defun w (w wn) (getf w wn))

(defun kwn (k i) (kw (format nil "~A~A" k i)))

(defun read-mobilenet-v2-text-weights ()
  (loop :for i :from 0 :to 51
        :for nm = (format nil "p~A" i)
        :for k = (kw nm)
        :append (list k (read-text-weight-file nm))))

(defun read-mobilenet-v2-weights ()
  (loop :for i :from 0 :to 51
        :for nm = (format nil "p~A" i)
        :for k = (kw nm)
        :append (list k (read-weight-file nm))))

(defun write-binary-weight-file (w filename)
  (let ((f (file.disk filename "w")))
    (setf ($fbinaryp f) t)
    ($fwrite w f)
    ($fclose f)))

(defun write-mobilenet-v2-binary-weights (&optional weights)
  (let ((weights (or weights (read-mobilenet-v2-text-weights))))
    (loop :for wk :in weights :by #'cddr
          :for w = (getf weights wk)
          :do (write-binary-weight-file w (wfname-bin wk)))))
