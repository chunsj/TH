(defpackage :em-example
  (:use #:common-lisp
        #:mu
        #:th
        #:th.pp))

(in-package :em-example)

(defparameter *ts* nil)

(defun p (p1 p2 &optional (m 0.5))
  (if (< (random 1.0) m)
      (max p1 p2)
      (min p1 p2)))

;;
;; PROBLEM 1.
;;
;; we have a fair coin. flip it and if it shows head, then sample from a binomial
;; distribution with p = p1 = 0.8, if tail, sample from another binomial distribution
;; with p = p2 = 0.45.
;; the sampling shows the data of 5 9 8 4 7.
;;
;; if we do not know p1 and p2, then how can we estimate them from the data?
;; this is the problem for explaining EM algorithm, but here, i'd like to estimate
;; them using MCMC. (to know whether it is possible)
;;

(defparameter *data* (tensor '(5 9 8 4 7)))

(defun posterior (p1 p2)
  (let ((prior-p1 (score/uniform p1 0.0 1.0))
        (prior-p2 (score/uniform p2 0.0 1.0)))
    (when (and (> p1 p2) prior-p1 prior-p2)
      (let ((l (loop :for i :from 0 :below ($count *data*)
                     :for p = (p p1 p2)
                     :summing (score/binomial ($ *data* i) p 10))))
        ($+ prior-p1 prior-p2 l)))))

(let ((traces (mcmc/mh '(0.51 0.5) #'posterior)))
  (setf *ts* traces)
  (loop :for trace :in traces
        :do (prn trace (trace/hpd trace))))

;;
;; PROBLEM 2.
;;
;; now the coin is not fair, its bias is m = 0.6, which is another unknown.
;; estimate p1, p2 and m.

(defparameter *data2* (tensor '(10 4 3 7 8)))

(defun posterior2 (p1 p2 m)
  (let ((prior-p1 (score/uniform p1 0.0 1.0))
        (prior-p2 (score/uniform p2 0.0 1.0))
        (prior-m (score/uniform m 0.0 1.0)))
    (when (and (> p1 p2) prior-p1 prior-p2 prior-m)
      (let ((l (loop :for i :from 0 :below ($count *data2*)
                     :for p = (p p1 p2 m)
                     :summing (score/binomial ($ *data2* i) p 10))))
        ($+ prior-p1 prior-p2 prior-m l)))))

(let ((traces (mcmc/mh '(0.51 0.5 0.5) #'posterior2)))
  (setf *ts* traces)
  (loop :for trace :in traces
        :do (prn trace (trace/hpd trace))))

(trace/values ($2 *ts*))
