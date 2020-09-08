(defpackage :mcmc-work
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :mcmc-work)

(defun model1 (n) ($normal! (tensor n) 10 3))

(defparameter *population* (model1 30000))
(defparameter *observation* (tensor (loop :repeat 10000
                                          :collect ($ *population* (random 30000)))))
(defparameter *mu-obs* ($mean *observation*))

(defun transition-model (theta)
  (let ((mean (car theta))
        (sd (cadr theta)))
    (list mean (exp (random/normal (log sd) 0.01)))))

(defun log-prior (theta)
  (let ((sd (cadr theta)))
    (if (<= sd 0)
        -30
        0)))

(defun log-likelihood-normal (theta data)
  (let ((mean (car theta))
        (sd (cadr theta))
        (n ($count data)))
    (- (* (/ n -2) (log (* 2 pi sd sd)))
       (/ ($sum ($square ($- data mean))) (* 2 sd sd)))))

(defun acceptance (like newlike)
  (let ((d (- newlike like)))
    (if (> d 0)
        T
        (< (log (random 1D0)) d))))

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
    (list :accepted accepted
          :rejected rejected)))

(defparameter *simulation* (time
                            (metropolis-hastings #'log-likelihood-normal
                                                 #'log-prior
                                                 #'transition-model
                                                 (list *mu-obs* 0.1)
                                                 50000
                                                 *observation*
                                                 #'acceptance)))

(prn *simulation*)
(prn ($count (getf *simulation* :accepted)))
(prn ($count (getf *simulation* :rejected)))

(let* ((n ($count (getf *simulation* :accepted)))
       (m (round (/ n 4)))
       (avs (subseq (reverse (getf *simulation* :accepted)) m))
       (vs (loop :for x :in avs :collect (cadr x)))
       (sum (reduce #'+ vs))
       (sd (* 1D0 (/ sum ($count vs)))))
  (list sd ($sd *observation*)))
