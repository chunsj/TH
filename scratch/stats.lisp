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

;; XXX following functions should have derivative function implementations
($lgamma (ones 10))
($gamma (ones 10))
($beta (ones 10) (ones 10))
($lbeta (ones 10) (ones 10))
($polygamma (ones 10) 0)
($erf (ones 10))
($erfc (ones 10))

($gamma (tensor '(1/3)))
($gamma (tensor '(4.8)))
