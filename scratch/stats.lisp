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
