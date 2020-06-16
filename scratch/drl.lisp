(defpackage :drl-ch02
  (:use #:common-lisp
        #:mu
        #:th
        #:th.env)
  (:import-from #:th.env.examples))

(in-package :drl-ch02)

(defun all-true (list)
  (let ((x (loop :for e :in list
                 :when (not e)
                   :do (return T))))
    (not x)))

(defun hash-table-values (ht)
  (loop :for k :in (hash-table-keys ht)
        :collect ($ ht k)))

(defun all-terminal-p (pas)
  (let* ((pss (hash-table-values pas)))
    (all-true (mapcar (lambda (ps) (all-true (mapcar (lambda (es) ($last es)) ps))) pss))))

(defun print-policy (policy p &key (action-symbols '("<" "v" ">" "^")) (ncols 4) (title "Policy:"))
  (format T "~A~%" title)
  (loop :for s :from 0 :below ($count p)
        :for a = (funcall policy s)
        :do (progn
              (format T "| ")
              (if (all-terminal-p ($ p s))
                  (format T "          ")
                  (format T "~2,'0D ~6@A " s ($ action-symbols a)))
              (if (zerop (rem (1+ s) ncols))
                  (format T "|~%")))))

(defun probability-success (env policy goal-state &key (nepisodes 100) (max-steps 200))
  (let ((results '()))
    (loop :for e :from 0 :below nepisodes
          :do (let ((state (env-reset! env))
                    (done nil))
                (loop :for steps :from 0 :below max-steps
                      :while (not done)
                      :do (let* ((tx (env-step! env (funcall policy state))))
                            (setf state (car tx))
                            (setf done ($last tx))))
                (push (if (eq state goal-state) 1D0 0) results)))
    (/ (reduce #'+ results) ($count results))))

(defun mean-return (env policy &key (nepisodes 100) (max-steps 200))
  (let ((results '()))
    (loop :for e :from 0 :below nepisodes
          :do (let ((state (env-reset! env))
                    (reward 0D0)
                    (done nil))
                (loop :for steps :from 0 :below max-steps
                      :while (not done)
                      :do (let* ((tx (env-step! env (funcall policy state))))
                            (setf state (car tx))
                            (setf reward (cadr tx))
                            (setf done ($last tx))))
                (push reward results)))
    (/ (reduce #'+ results) ($count results))))

(let* ((env (th.env.examples:slippery-walk-five-env))
       (p (env-p env))
       (policy (lambda (s) ($ '(0 0 0 0 0 0 0) s))))
  (print-policy policy p :action-symbols '("<" ">") :ncols 7)
  (list :success-rate (probability-success env policy 6)
        :mean-return (mean-return env policy)))
