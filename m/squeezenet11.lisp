(defpackage :th.m.squeezenet11
  (:use #:common-lisp
        #:mu
        #:th)
  (:export #:read-squeezenet11-weights
           #:squeezenet11
           #:squeezenet11fcn))

(in-package :th.m.squeezenet11)

(defparameter +model-location+ ($concat (namestring (user-homedir-pathname)) ".th/models"))

(defun wfname-txt (wn)
  (format nil "~A/squeezenet11/squeezenet11-~A.txt"
          +model-location+
          (string-downcase wn)))

(defun wfname-bin (wn)
  (format nil "~A/squeezenet11/squeezenet11-~A.dat"
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

(defun read-squeezenet11-text-weights ()
  (loop :for i :from 0 :to 51
        :for nm = (format nil "p~A" i)
        :for k = (kw nm)
        :append (list k (read-text-weight-file nm))))

(defun read-squeezenet11-weights ()
  (loop :for i :from 0 :to 51
        :for nm = (format nil "p~A" i)
        :for k = (kw nm)
        :append (list k (read-weight-file nm))))

(defun write-binary-weight-file (w filename)
  (let ((f (file.disk filename "w")))
    (setf ($fbinaryp f) t)
    ($fwrite w f)
    ($fclose f)))

(defun write-squeezenet11-binary-weights (&optional weights)
  (let ((weights (or weights (read-squeezenet11-text-weights))))
    (loop :for wk :in weights :by #'cddr
          :for w = (getf weights wk)
          :do (write-binary-weight-file w (wfname-bin wk)))))

(defun fire (x ws k1 b1 k2 b2 k3 b3)
  (let* ((x (-> ($conv2d x (w ws k1) (w ws b1) 1 1)
                ($relu)))
         (e1x1 (-> ($conv2d x (w ws k2) (w ws b2) 1 1)
                   ($relu)))
         (e3x3 (-> ($conv2d x (w ws k3) (w ws b3) 1 1 1 1)
                   ($relu))))
    ($cat e1x1 e3x3 1)))

(defun squeezenet11 (&optional weights)
  (let ((ws (or weights (read-squeezenet11-weights))))
    (lambda (x)
      (when (and x (>= ($ndim x) 3) (equal (last ($size x) 3) (list 3 224 224)))
        (let ((x (if (eq ($ndim x) 3)
                     ($reshape x 1 3 224 224)
                     x)))
          (-> x
              ($conv2d (w ws :p0) (w ws :p1) 2 2)
              ($relu)
              ($maxpool2d 3 3 2 2 0 0 T)
              (fire ws :p2 :p3 :p4 :p5 :p6 :p7)
              (fire ws :p8 :p9 :p10 :p11 :p12 :p13)
              ($maxpool2d 3 3 2 2 0 0 T)
              (fire ws :p14 :p15 :p16 :p17 :p18 :p19)
              (fire ws :p20 :p21 :p22 :p23 :p24 :p25)
              ($maxpool2d 3 3 2 2 0 0 T)
              (fire ws :p26 :p27 :p28 :p29 :p30 :p31)
              (fire ws :p32 :p33 :p34 :p35 :p36 :p37)
              (fire ws :p38 :p39 :p40 :p41 :p42 :p43)
              (fire ws :p44 :p45 :p46 :p47 :p48 :p49)
              ($conv2d (w ws :p50) (w ws :p51))
              ($relu)
              ($avgpool2d 13 13 1 1)
              ($reshape ($size x 0) 1000)
              ($softmax)))))))

(defun squeezenet11fcn (&optional weights)
  (let ((ws (or weights (read-squeezenet11-weights))))
    (lambda (x)
      (when (and x (>= ($ndim x) 3))
        (let ((x (if (eq ($ndim x) 3)
                     ($unsqueeze x 0)
                     x)))
          (-> x
              ($conv2d (w ws :p0) (w ws :p1) 2 2)
              ($relu)
              ($maxpool2d 3 3 2 2)
              (fire ws :p2 :p3 :p4 :p5 :p6 :p7)
              (fire ws :p8 :p9 :p10 :p11 :p12 :p13)
              ($maxpool2d 3 3 2 2)
              (fire ws :p14 :p15 :p16 :p17 :p18 :p19)
              (fire ws :p20 :p21 :p22 :p23 :p24 :p25)
              ($maxpool2d 3 3 2 2)
              (fire ws :p26 :p27 :p28 :p29 :p30 :p31)
              (fire ws :p32 :p33 :p34 :p35 :p36 :p37)
              (fire ws :p38 :p39 :p40 :p41 :p42 :p43)
              (fire ws :p44 :p45 :p46 :p47 :p48 :p49)
              ($conv2d (w ws :p50) (w ws :p51))
              ($relu)
              ($avgpool2d 13 13 1 1)
              ($softmax)))))))
