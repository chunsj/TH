(defpackage :stats
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :stats)

;; https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_PyMC2.ipynb

(defparameter *sms-data* (->> (slurp "./data/sms.txt")
                              (mapcar #'parse-float)
                              (mapcar #'round)))
(defparameter *N* ($count *sms-data*))

(defun mean (vs) (/ (reduce #'+ vs) ($count vs)))

(defparameter *alpha* (/ 1D0 (mean *sms-data*)))

(random/exponential 0.5)
(random/exponential (/ 1 10))

(let ((l 2))
  (mean (loop :repeat 1000 :collect (random/exponential (/ 1 l)))))

(defgeneric $log-probability (distribution x))

(defclass distribution () ())

(defclass normal-distribution (distribution)
  ((mn :initform nil)
   (sd :initform nil)))

(defun normal-distribution (m s)
  (let ((d (make-instance 'normal-distribution)))
    (with-slots (mn sd) d
      (setf mn m
            sd s))
    d))

(defmethod $log-probability ((d normal-distribution) x)
  (with-slots (mn sd) d
    (let ((n ($count x)))
      (if (> ($scalar sd) 0)
          (let* ((var ($* sd sd))
                 (v2 ($* 2D0 var))
                 (c1 ($- ($* (/ n 2) ($log ($* pi v2)) )))
                 (c2 ($/ -1D0 v2))
                 (d ($- x ($expand mn (list n))))
                 (d2 ($square d)))
            ($+ c1 ($* c2 ($sum d2))))
          (tensor (list most-negative-single-float))))))

(defclass uniform-distribution (distribution)
  ((mx :initform nil)
   (mn :initform nil)))

(defun uniform-distribution (min max)
  (let ((d (make-instance 'uniform-distribution)))
    (with-slots (mx mn) d
      (setf mn min
            mx max))
    d))

(defun $val (tp) (if ($parameterp tp) ($data tp) tp))

(defmethod $log-probability ((d uniform-distribution) x)
  (with-slots (mn mx) d
    (let ((n ($count x)))
      (if (and (> ($scalar mx) ($scalar mn))
               (eq n ($sum ($ge ($val x) ($scalar mn))))
               (eq n ($sum ($le ($val x) ($scalar mx)))))
          (let ((lp ($log ($- mx mn))))
            ($* -1D0 n lp))
          (tensor (list most-negative-single-float))))))

(defparameter *group1*
  (tensor '(-0.01232848  0.63928471  0.14409147 -0.20178967  0.53556889
            -1.46360526 -0.79586204 -0.78776574 -0.00517005 -0.17374837
            0.09940546  0.65519677  0.98951772  0.71058968 -0.26497844
            0.89939069  0.13706369  1.9002145   0.9816272   0.3148801 )))

(defparameter *group2*
  (tensor '(-0.08777963 -0.98211783 0.12169048 -1.1374373 0.34900257
            -1.8585131  -1.1671818  1.4248968   1.4965653 1.289932
            -1.8117453 -1.4983072  -1.4501432  -1.6939069 0.22726403
            -0.4897347 -5.285065E-4 -0.4902526 -0.79320943 2.0488987)))

(let ((mean1 ($parameter (tensor '(0))))
      (sd1 ($parameter (tensor '(1))))
      (mean2 ($parameter (tensor '(0))))
      (sd2 ($parameter (tensor '(1)))))
  (loop :repeat 2000
        :for i :from 1
        :for l1 = ($log-probability (normal-distribution mean1 sd1) *group1*)
        :for l2 = ($log-probability (normal-distribution mean2 sd2) *group2*)
        :for pm = ($log-probability (normal-distribution (tensor '(0)) (tensor '(1)))
                                    ($cat mean1 mean2))
        :for ps = ($log-probability (uniform-distribution (tensor '(0)) (tensor '(100)))
                                    ($cat sd1 sd2))
        :do (let ((loss ($- ($+ l1 l2 pm ps))))
              (when (zerop (rem i 100)) (prn "LOSS" i ($scalar loss)))
              ($amgd! (list mean1 sd1 mean2 sd2) 0.003)))
  (prn mean1 sd1 mean2 sd2))
