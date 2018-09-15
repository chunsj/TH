(defpackage :th.m.vgg16
  (:use #:common-lisp
        #:mu
        #:th)
  (:export #:read-vgg16-weights))

(in-package :th.m.vgg16)

(defparameter +model-location+ ($concat (namestring (user-homedir-pathname)) ".th.models"))

(defun read-weight-file (wn)
  (let ((f (file.disk (format nil "~A/vgg16/vgg16-~A.txt" +model-location+ wn) "r"))
        (tx (tensor)))
    ($fread tx f)
    ($fclose f)
    tx))

;; XXX maybe binary blob will be faster to read/create tensors
(defun read-vgg16-weights ()
  (list :k1 (read-weight-file "k1") :b1 (read-weight-file "b1")
        :k2 (read-weight-file "k2") :b2 (read-weight-file "b2")
        :k3 (read-weight-file "k3") :b3 (read-weight-file "b3")
        :k4 (read-weight-file "k4") :b4 (read-weight-file "b4")
        :k5 (read-weight-file "k5") :b5 (read-weight-file "b5")
        :k6 (read-weight-file "k6") :b6 (read-weight-file "b6")
        :k7 (read-weight-file "k7") :b7 (read-weight-file "b7")
        :k8 (read-weight-file "k8") :b8 (read-weight-file "b8")
        :k9 (read-weight-file "k9") :b9 (read-weight-file "b9")
        :k10 (read-weight-file "k10") :b10 (read-weight-file "b10")
        :k11 (read-weight-file "k11") :b11 (read-weight-file "b11")
        :k12 (read-weight-file "k12") :b12 (read-weight-file "b12")
        :k13 (read-weight-file "k13") :b13 (read-weight-file "b13")
        :w14 (read-weight-file "w14") :b14 (read-weight-file "b14")
        :w15 (read-weight-file "w15") :b15 (read-weight-file "b15")
        :w16 (read-weight-file "w16") :b16 (read-weight-file "b16")))
