(defpackage :gdl-ch04
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :gdl-ch04)

;; error
(let ((knob-weight 0.5)
      (input 0.5)
      (goal-pred 0.8))
  (let* ((pred (* input knob-weight))
         (err (expt (- pred goal-pred) 2)))
    (prn err)))

;; simplest form of neural learning
(let ((weight 0.1)
      (lr 0.01)
      (number-of-toes '(8.5))
      (win-or-lose-binary '(1)))
  (defun neural-network (input weight)
    (* input weight))
  (let ((input ($ number-of-toes 0))
        (y ($ win-or-lose-binary 0)))
    (let* ((pred (neural-network input weight))
           (err (expt (- pred y) 2)))
      (let* ((p-up (neural-network input (+ weight lr)))
             (e-up (expt (- p-up y) 2))
             (p-dn (neural-network input (- weight lr)))
             (e-dn (expt (- p-dn y) 2)))
        (if (or (> err e-dn) (> err e-up))
            (if (< e-dn e-up)
                (setf weight (- weight lr))
                (setf weight (+ weight lr))))
        (print weight)))))

;; hot and cold learning
(defparameter *weight* 0.5)
(defparameter *input* 0.5)
(defparameter *goal-prediction* 0.8)

(defparameter *step-amount* 0.001)

(loop :for i :from 0 :below 1101
      :for prediction = (* *input* *weight*)
      :for err = (expt (- prediction *goal-prediction*) 2)
      :do (let* ((up-prediction (* *input* (+ *weight* *step-amount*)))
                 (dn-prediction (* *input* (- *weight* *step-amount*)))
                 (up-error (expt (- up-prediction *goal-prediction*) 2))
                 (dn-error (expt (- dn-prediction *goal-prediction*) 2)))
            (print (list err prediction))
            (when (< dn-error err)
              (setf *weight* (- *weight* *step-amount*)))
            (when (< up-error err)
              (setf *weight* (+ *weight* *step-amount*)))))

;; gradient descent
(defparameter *weight* 0.0)
(defparameter *input* 0.5)
(defparameter *goal-prediction* 0.8)

(loop :for i :from 0 :below 40
      :for pred = (* *input* *weight*)
      :for err = (expt (- pred *goal-prediction*) 2)
      :for delta = (- pred *goal-prediction*)
      :for weight-delta = (* delta *input*)
      :do (let ((new-weight (- *weight* weight-delta)))
            (setf *weight* new-weight)
            (print (list err pred))))

;; chapter 05 is so tedious...
