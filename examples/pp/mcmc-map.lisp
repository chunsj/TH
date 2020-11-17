(defpackage :mcmc-map
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :mcmc-map)

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

(defparameter *inf* -1E10)

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
          (tensor (list *inf*))))))

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
          (tensor (list *inf*))))))

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

(defun log-likelihood (model)
  (let ((mn1 ($0 model))
        (sd1 ($1 model))
        (mn2 ($2 model))
        (sd2 ($3 model)))
    ($+ ($log-probability (normal-distribution mn1 sd1) *group1*)
        ($log-probability (normal-distribution mn2 sd2) *group2*)
        ($log-probability (normal-distribution (tensor '(0)) (tensor '(1)))
                          ($cat mn1 mn2))
        ($log-probability (uniform-distribution (tensor '(0)) (tensor '(100)))
                          ($cat sd1 sd2)))))

(defun find-map (model &optional (n 2000))
  (let ((lr 0.003))
    (loop :repeat n
          :for i :from 1
          :for ll = (log-likelihood model)
          :do (let ((loss ($scalar ($- ll))))
                (when (zerop (rem i 100)) (prn i loss))
                ($amgd! model lr)))))

(defun print-model (model)
  (let ((mn1 ($0 model))
        (sd1 ($1 model))
        (mn2 ($2 model))
        (sd2 ($3 model)))
    (prn "MEAN:" (format nil "~8,4F" ($scalar mn1)) (format nil "~8,4F" ($scalar mn2)))
    (prn "STDV:" (format nil "~8,4F" ($scalar sd1)) (format nil "~8,4F" ($scalar sd2)))))

(defparameter *mn1* ($parameter (tensor '(0))))
(defparameter *sd1* ($parameter (tensor '(1))))
(defparameter *mn2* ($parameter (tensor '(0))))
(defparameter *sd2* ($parameter (tensor '(1))))

(defparameter *model* (list *mn1* *sd1* *mn2* *sd2*))

(find-map *model*)
(print-model *model*)

(defun proposal (model &optional (scale 1D0))
  (let ((mn1-new (random/normal ($scalar ($0 model)) scale))
        (sd1-new (random/normal ($scalar ($1 model)) scale))
        (mn2-new (random/normal ($scalar ($2 model)) scale))
        (sd2-new (random/normal ($scalar ($3 model)) scale)))
    (mapcar (lambda (v) (tensor (list v))) (list mn1-new sd1-new mn2-new sd2-new))))

(defun mean-accepted (accepted)
  (let ((na ($count accepted)))
    (mapcar #'$/ (reduce (lambda (s v) (mapcar #'$+ s v)) accepted) (list na na na na))))

(let* ((m *model*)
       (n 10000)
       (nb 5000)
       (l (log-likelihood m))
       (scale 0.1)
       (accepted '())
       (rejected '()))
  (loop :repeat (+ n nb)
        :for i :from 1
        :for m-new = (proposal m scale)
        :for l-new = (log-likelihood m-new)
        :do (let ((dl (- ($scalar l-new) ($scalar l)))
                  (r (log (random 1D0))))
              (if (< r dl)
                  (progn
                    (when (>= i nb) (push m-new accepted))
                    (setf l l-new
                          m m-new))
                  (when (>= i nb) (push m-new rejected)))))
  (let ((na ($count accepted))
        (nr ($count rejected)))
    (prn "ACCEPTED/REJECTED" na "/" nr)
    (prn "MODEL")
    (print-model *model*)
    (prn "ACCEPTED")
    (print-model (mean-accepted accepted)))
  (let ((dm (mapcar (lambda (p)
                      (let ((m1 ($0 p))
                            (m2 ($2 p)))
                        (if (> ($scalar m1) ($scalar m2))
                            1
                            0)))
                    accepted))
        (na ($count accepted)))
    (prn "P(M1>M2)" (* 1D0 (/ (reduce #'+ dm) na)))))
