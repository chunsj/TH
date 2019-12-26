(defpackage :hmm-simple
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :hmm-simple)

(defclass hmm ()
  ((n :accessor state-counts)
   (p0 :accessor initial-state-probabilities)
   (tm :accessor transition-matrix)
   (em :accessor emission-matrix)))

(defun get-emission (hmm iobs)
  (let ((nr ($size (emission-matrix hmm) 0)))
    ($ (emission-matrix hmm) (list 0 nr) (list iobs 1))))

(defun forward-init (hmm iobs)
  (let ((pobs (get-emission hmm iobs)))
    ($* (initial-state-probabilities hmm) pobs)))

(defun forward-step (hmm iobs fwd)
  (let ((transitions ($@ fwd ($transpose (get-emission hmm iobs)))))
    (let ((weighted-transitions ($* transitions (transition-matrix hmm))))
      (let ((nfwd ($sum weighted-transitions 0)))
        (apply #'$reshape nfwd ($size fwd))))))

(defun decode-step (hmm iobs viterbi)
  (let ((transitions ($@ viterbi ($transpose (get-emission hmm iobs)))))
    (let ((weighted-transitions ($* transitions (transition-matrix hmm))))
      (let ((nviterbi (car ($max weighted-transitions 0))))
        (apply #'$reshape nviterbi ($size viterbi))))))

(defun backpt-step (hmm viterbi)
  (let ((back-transitions ($@ viterbi (ones 1 (state-counts hmm)))))
    (let ((weighted-back-transitions ($* back-transitions (transition-matrix hmm))))
      (let ((rmax ($max weighted-back-transitions 0)))
        (cadr rmax)))))

(defun forward-algorithm (hmm observations)
  (let ((fwd (forward-init hmm (car observations))))
    (loop :for iobs :in (cdr observations)
          :for nfwd = (forward-step hmm iobs fwd)
          :do (setf fwd nfwd))
    ($sum fwd)))

(defun viterbi-decode (hmm observations)
  (let ((viterbi (forward-init hmm (car observations)))
        (backpts (tensor.long ($* (ones (state-counts hmm) ($count observations)) -1))))
    (loop :for iobs :in (cdr observations)
          :for i :from 1
          :for nviterbi = (decode-step hmm iobs viterbi)
          :for backpt = (backpt-step hmm viterbi)
          :do (setf viterbi nviterbi
                    ($ backpts (list 0 ($size backpts 0)) (list i 1)) backpt))
    (let ((tokens (list ($ (cadr ($max viterbi 0)) 0 0))))
      (loop :for i :from (1- ($count observations)) :downto 1
            :for lt = (car tokens)
            :for bt = ($ backpts lt i)
            :do (push bt tokens))
      tokens)))

(defparameter *hmm* (make-instance 'hmm))

(defparameter *initial-state-probabilities* (tensor '((0.6) (0.4))))
(defparameter *transition-matrix* (tensor '((0.7 0.3) (0.4 0.6))))
(defparameter *emission-matrix* (tensor '((0.1 0.4 0.5) (0.6 0.3 0.1))))
(defparameter *observations* '(0 1 1 2 1))

(setf (state-counts *hmm*) ($count *initial-state-probabilities*))
(setf (initial-state-probabilities *hmm*) *initial-state-probabilities*)
(setf (transition-matrix *hmm*) *transition-matrix*)
(setf (emission-matrix *hmm*) *emission-matrix*)

(forward-algorithm *hmm* *observations*)
(viterbi-decode *hmm* *observations*)
