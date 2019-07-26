(defpackage :rl-simple
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :rl-simple)

;; XXX this should not use other library
(defparameter *prices* (tensor (reverse (ma:opens (ma:bars "005930" :start "1980-01-01")))))

(defclass decision-policy ()
  ((actions :accessor policy-actions)))

(defgeneric select-action (policy current-state step))
(defgeneric update-Q (policy state action reward next-state))

(defmethod select-action ((policy decision-policy) current-state step))
(defmethod update-Q ((policy decision-policy) state action reward next-state))

(defclass random-decision-policy (decision-policy) ())

(defun random-decision-policy (actions)
  (let ((n (make-instance 'random-decision-policy)))
    (setf (policy-actions n) actions)
    n))

(defmethod select-action ((policy random-decision-policy) current-state step)
  ($ (policy-actions policy) (random ($count (policy-actions policy)))))

(defun run-simulation (policy initial-budget initial-num-stocks prices hist)
  (let ((budget initial-budget)
        (num-stocks initial-num-stocks)
        (share-value 0)
        (transitions (list)))
    (loop :for i :from 0 :below (- ($count prices) hist 1)
          :for current-state = ($cat ($ prices (list i hist)) (tensor (list budget num-stocks)))
          :for current-portfolio = (+ budget (* num-stocks share-value))
          :for action = (select-action policy current-state i)
          :do (progn
                (setf share-value ($ prices (+ i hist 1)))
                (cond ((and (eq action :buy) (>= budget share-value))
                       (progn
                         (decf budget share-value)
                         (incf num-stocks)))
                      ((and (eq action :shell) (> num-stocks 0))
                       (progn
                         (incf budget share-value)
                         (decf num-stocks)))
                      (t (setf action :hold)))
                (let* ((new-portfolio (+ budget (* num-stocks share-value)))
                       (reward (- new-portfolio current-portfolio))
                       (next-state ($cat ($ prices (list (1+ i) hist))
                                         (tensor (list budget num-stocks)))))
                  (push (list current-state action reward next-state) transitions)
                  (update-Q policy current-state action reward next-state))))
    (+ budget (* num-stocks share-value))))

(defun run-simulations (policy budget num-stocks prices hist)
  (let ((num-tries 10))
    (loop :for i :from 0 :below num-tries
          :for final-portfolio = (run-simulation policy budget num-stocks prices hist)
          :collect (progn
                     (prn final-portfolio)
                     final-portfolio))))

(defparameter *actions* '(:buy :sell :hold))
(defparameter *policy* (random-decision-policy *actions*))
(defparameter *budget* 100000D0)
(defparameter *num-stocks* 0)

(run-simulations *policy* *budget* *num-stocks* ($ *prices* (list 5000 3000)) 3)
