(defpackage :mcmc-work
  (:use #:common-lisp
        #:mu
        #:th
        #:mplot))

(in-package :mcmc-work)

;; PROBLEM DESCRIPTION
;;
;; estimate standard deviation of population using partial observations.
;; of course, this case should have analytic solution but i like to apply
;; markov chain monte caro, or metropolis-hastings algorithm.

(defun model1 (n) ($normal! (tensor n) 10 3))

;; population from N(10, 3) distribution
(defparameter *population* (model1 30000))
;; select random 10000 samples from population
(defparameter *observation* (tensor (loop :repeat 10000
                                          :collect ($ *population* (random 30000)))))
;; we assume that we know the mean
(defparameter *mu-obs* ($mean *observation*))

;; proposal or transition
;; to make standard deviation proposal as positive, log-normal is used
(defun transition-model (theta)
  (let ((mean (car theta))
        (sd (cadr theta)))
    (list mean (exp (random/normal (log sd) 0.01)))))

;; redundant in this case, however, we need one
(defun log-prior (theta)
  (declare (ignore theta))
  (log 1))

;; for numerical stability, we'll use log-likelihood values
;; our sample model is normal so this is log-likelihood for normal density
(defun log-likelihood-normal (theta data)
  (let ((mean (car theta))
        (sd (cadr theta))
        (n ($count data)))
    (- (* (/ n -2) (log (* 2 pi sd sd)))
       (/ ($sum ($square ($- data mean))) (* 2 sd sd)))))

;; compare current likelihood and new likelihood, then choose the proposed theta
(defun acceptance (like newlike)
  (let ((d (- newlike like)))
    (if (> d 0)
        T
        (< (log (random 1D0)) d))))

;; metropolis hastings algorithm
(defun metropolis-hastings (lfn log-prior transition theta0 iterations data acceptance-rule)
  (let ((theta theta0)
        (accepted '())
        (rejected '()))
    (loop :repeat iterations
          :for theta-new = (funcall transition theta)
          :for theta-like = (funcall lfn theta data)
          :for theta-new-like = (funcall lfn theta-new data)
          :do (if (funcall acceptance-rule
                           ($+ theta-like (funcall log-prior theta))
                           ($+ theta-new-like (funcall log-prior theta-new)))
                  (progn
                    (setf theta theta-new)
                    (push theta-new accepted))
                  (push theta-new rejected)))
    (prn "ACCEPTED/REJECTED:" ($count accepted) "/" ($count rejected))
    (list :accepted (reverse accepted)
          :rejected (reverse rejected))))

(defun simulation/accepted (simulation) (getf simulation :accepted))
(defun simulation/rejected (simulation) (getf simulation :rejected))

;; generate simulation data
(defparameter *simulation* (time
                            (metropolis-hastings #'log-likelihood-normal
                                                 #'log-prior
                                                 #'transition-model
                                                 (list *mu-obs* 0.1)
                                                 50000
                                                 *observation*
                                                 #'acceptance)))

(defun mean (vs) (/ (reduce #'+ vs) ($count vs)))

;; compare estimation vs sample standard deviation
(let* ((accepted (simulation/accepted *simulation*))
       (n ($count accepted))
       (sn (round (* n 0.25)))
       (thetas (subseq accepted sn))
       (esd (mean (mapcar #'cadr thetas))))
  (list esd ($sd *observation*)))

;; accepted theta-sd, C-c C-i
(mplot:plot-points (loop :for theta :in (simulation/accepted *simulation*)
                         :for i :from 0
                         :collect (list i (cadr theta))))

;; check burn-in
(mplot:plot-points (loop :for theta :in (simulation/accepted *simulation*)
                         :for i :from 0 :below 1000
                         :collect (list i (cadr theta))))

;; last values
(mplot:plot-points (loop :for theta :in (last (simulation/accepted *simulation*) 5000)
                         :for i :from 0
                         :collect (list i (cadr theta)))
                   :yrange '(2.5 3.5))
