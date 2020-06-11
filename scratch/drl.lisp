(defpackage :drl-ch02
  (:use #:common-lisp
        #:mu
        #:th
        #:th.env)
  (:import-from #:th.env.examples))

(in-package :drl-ch02)

;; representation of deterministic bandit walk environment.
;; 1 non-terminal state(1), and 2 terminal states(0,2).
;; avaliable actions are left or right (0 or 1)
;; only reward is at the waking to the right-most cell(1 to 2).
;;
;; shape: state - action - transitions (probability, next-state, reward, terminalp).
(defparameter *bandit-walk-p* #{0 #{0 '((1D0 0 0D0 T))
                                    1 '((1D0 0 0D0 T))}
                                1 #{0 '((1D0 0 0D0 T))
                                    1 '((1D0 2 1D0 T))}
                                2 #{0 '((1D0 2 0D0 T))
                                    1 '((1D0 2 0D0 T))}})

(defclass bandit-walk-env ()
  ((p :initform *bandit-walk-p* :accessor env-p)
   (s :initform 1 :accessor env-state)))

(defun bandit-walk-env () (make-instance 'bandit-walk-env))

(defmethod env-step! ((env bandit-walk-env) action)
  (with-slots (p s) env
    (let* ((txs ($ ($ p s) action))
           (tx (cdar txs)))
      (setf s (car tx))
      tx)))

(defmethod env-reset! ((env bandit-walk-env))
  (with-slots (p s) env
    (setf s 1)
    env))

(let ((env (bandit-walk-env)))
  (env-reset! env)
  (env-step! env 1))

(defparameter *bandit-slippery-walk-p* #{0 #{0 '((1D0 0 0D0 T))
                                             1 '((1D0 0 0D0 T))}
                                         1 #{0 '((0.8D0 0 0D0 T) (0.2D0 2 1D0 T))
                                             1 '((0.8D0 2 1D0 T) (0.2D0 0 0D0 T))}
                                         2 #{0 '((1D0 2 0D0 T))
                                             1 '((1D0 2 0D0 T))}})

(defclass bandit-slippery-walk-env ()
  ((p :initform *bandit-slippery-walk-p* :accessor env-p)
   (s :initform 1 :accessor env-state)))

(defun bandit-slippery-walk-env () (make-instance 'bandit-slippery-walk-env))

(defmethod env-step! ((env bandit-slippery-walk-env) action)
  (with-slots (p s) env
    (let* ((txs ($ ($ p s) action))
           (tx (cdr ($choice txs (mapcar #'car txs)))))
      (setf s (car tx))
      tx)))

(defmethod env-reset! ((env bandit-slippery-walk-env))
  (with-slots (p s) env
    (setf s 1)
    env))

(let ((env (bandit-slippery-walk-env)))
  (env-reset! env)
  (env-step! env 1))
