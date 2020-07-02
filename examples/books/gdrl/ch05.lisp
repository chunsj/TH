(defpackage :gdrl-ch05
  (:use #:common-lisp
        #:mu
        #:mplot
        #:th
        #:th.env)
  (:import-from #:th.env.bandits))

(in-package :gdrl-ch05)

(defun policy-evaluation (policy p &key (gamma 1D0) (theta 1E-10))
  (let ((prev-v (zeros ($count p)))
        (v nil)
        (keep-running-p T))
    (loop :while keep-running-p
          :for iter :from 0
          :do (progn
                (setf v (zeros ($count p)))
                (loop :for s :from 0 :below ($count p)
                      :for action = (funcall policy s)
                      :for txs = ($ ($ p s) action)
                      :do (loop :for tx :in txs
                                :for prob = ($0 tx)
                                :for next-state = ($1 tx)
                                :for reward = ($2 tx)
                                :for done = ($3 tx)
                                :do (incf ($ v s) (* prob (+ reward (* gamma ($ prev-v next-state)
                                                                       (if done 0D0 1D0)))))))
                (when (< ($max ($abs ($- prev-v v))) theta)
                  (setf keep-running-p nil))
                (setf prev-v ($clone v))))
    v))

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
  (format T "~%~A~%" title)
  (loop :for s :from 0 :below ($count p)
        :for a = (funcall policy s)
        :do (progn
              (format T "| ")
              (if (all-terminal-p ($ p s))
                  (format T "          ")
                  (format T "~2,'0D ~6@A " s ($ action-symbols a)))
              (if (zerop (rem (1+ s) ncols))
                  (format T "|~%")))))

(defun print-state-value-function (v p &key (ncols 4) (prec 4) (title "State-Value Function:"))
  (format T "~%~A~%" title)
  (loop :for s :from 0 :below ($count p)
        :for val = ($ v s)
        :do (progn
              (format T "| ")
              (if (all-terminal-p ($ p s))
                  (format T "          ")
                  (format T (format nil "~~2,'0D ~~6,~A@F " prec) s val))
              (if (zerop (rem (1+ s) ncols))
                  (format T "|~%")))))

(defun print-action-value-function (q &key optimal-q (action-symbols '("<" ">"))
                                        (title "Action-Value Function:"))
  (format T "~%~A~%" title))

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
