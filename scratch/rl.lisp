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

(defclass q-learning-decision-policy (decision-policy)
  ((epsilon :initform 0.95D0 :accessor q-learning-epsilon)
   (gamma :initform 0.3D0 :accessor q-learning-gamma)
   (w1 :accessor q-learning-w1)
   (b1 :accessor q-learning-b1)
   (w2 :accessor q-learning-w2)
   (b2 :accessor q-learning-b2)
   (q :accessor policy-q)))

(defun q-learning-decision-policy (actions input-dim)
  (let ((n (make-instance 'q-learning-decision-policy)))
    (setf (policy-actions n) actions)
    (setf (q-learning-w1 n) ($parameter (rndn input-dim 20)))
    (setf (q-learning-b1 n) ($parameter ($* 0.1 (ones 20))))
    (setf (q-learning-w2 n) ($parameter (rndn 20 ($count actions))))
    (setf (q-learning-b2 n) ($parameter ($* 0.1 (ones ($count actions)))))
    n))

(defun q-learning-parameters (policy)
  (list (q-learning-w1 policy) (q-learning-b1 policy)
        (q-learning-w2 policy) (q-learning-b2 policy)))

(defun reset-gradients (policy) ($cg! (q-learning-parameters policy)))

(defun compute-q-value (policy x)
  (-> x
      ($affine (q-learning-w1 policy) (q-learning-b1 policy))
      ($relu)
      ($affine (q-learning-w2 policy) (q-learning-b2 policy))
      ($relu)))

(defun train-q-value (policy x y)
  (let* ((q (compute-q-value policy x))
         (d ($- y q)))
    ($@ d d))
  ($amgd! (q-learning-parameters policy) 0.001))

(defun q-value (policy x)
  (let ((q (compute-q-value policy x)))
    (reset-gradients policy)
    ($data q)))

(defmethod select-action ((policy q-learning-decision-policy) current-state step)
  (let ((threshold (min (q-learning-epsilon policy) (/ step 1000D0))))
    (if (< (random 1D0) threshold)
        (let* ((action-q-value (q-value policy current-state))
               (maxd ($max action-q-value 0))
               (argmax ($ (cadr maxd) 0)))
          ($ (policy-actions policy) argmax))
        ($ (policy-actions policy) (random ($count (policy-actions policy)))))))

(defmethod update-Q ((policy q-learning-decision-policy) state action reward next-state)
  (let* ((q (q-value policy state))
         (nq (q-value policy next-state))
         (nmaxd ($max nq 0))
         (nargmax ($ (cadr nmaxd) 0)))
    (setf ($ q nargmax)
          (+ reward (* (q-learning-gamma policy) ($ nq nargmax))))
    (train-q-value policy state q)))

(defparameter *actions* '(:buy :sell :hold))
(defparameter *policy* (q-learning-decision-policy *actions* (+ 3 2)))
(defparameter *budget* 100000D0)
(defparameter *num-stocks* 0)

(run-simulations *policy* *budget* *num-stocks* ($ *prices* (list 5000 3000)) 3)

(gcf)
