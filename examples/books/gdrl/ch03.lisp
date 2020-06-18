(defpackage :gdrl-ch03
  (:use #:common-lisp
        #:mu
        #:th
        #:th.env)
  (:import-from #:th.env.examples))

(in-package :gdrl-ch03)

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

(defun policy-improvement (v p &key (gamma 1D0))
  (let ((q (zeros ($count p) ($count ($ p 0)))))
    (loop :for s :from 0 :below ($count p)
          :do (loop :for a :from 0 :below ($count ($ p 0))
                    :for txs = ($ ($ p s) a)
                    :do (loop :for tx :in txs
                              :for prob = ($0 tx)
                              :for next-state = ($1 tx)
                              :for reward = ($2 tx)
                              :for done = ($3 tx)
                              :do (incf ($ q s a) (* prob (+ reward (* gamma ($ v next-state)
                                                                       (if done 0D0 1D0))))))))
    (let ((maxres ($max q 1)))
      (lambda (s)
        (let ((argmax (cadr maxres)))
          ($ argmax s 0))))))

(defun policy-iteration (p &key (gamma 1D0) (theta 1E-10))
  (let* ((actions (hash-table-keys ($ p 0)))
         (random-actions (loop :repeat ($count p)
                               :collect ($choice actions (loop :for i :from 0 :below ($count actions)
                                                               :collect 1D0))))
         (policy (lambda (s) ($ random-actions s)))
         (keep-running-p T)
         (value-res nil)
         (policy-res nil))
    (loop :while keep-running-p
          :for iter :from 0
          :for old-policy-res = (loop :for s :from 0 :below ($count p)
                                      :collect (funcall policy s))
          :for v = (policy-evaluation policy p :gamma gamma :theta theta)
          :for new-policy = (policy-improvement v p :gamma gamma)
          :do (let ((new-policy-res (loop :for s :from 0 :below ($count p)
                                          :collect (funcall new-policy s))))
                (if (all-true (mapcar (lambda (o n) (eq o n)) old-policy-res new-policy-res))
                    (setf keep-running-p nil))
                (setf value-res v
                      policy-res new-policy
                      policy new-policy)))
    (list value-res policy-res)))

(defun value-iteration (p &key (gamma 1D0) (theta 1E-10))
  (let ((v (zeros ($count p)))
        (keep-running-p T)
        (argmax nil))
    (loop :while keep-running-p
          :for q = (zeros ($count p) ($count ($ p 0)))
          :do (progn
                (loop :for s :from 0 :below ($count p)
                      :do (loop :for a :from 0 :below ($count ($ p s))
                                :for txs = ($ ($ p s) a)
                                :do (loop :for tx :in txs
                                          :for prob = ($0 tx)
                                          :for next-state = ($1 tx)
                                          :for reward = ($2 tx)
                                          :for done = ($3 tx)
                                          :do (incf ($ q s a)
                                                    (* prob (+ reward (* gamma ($ v next-state)
                                                                         (if done 0D0 1D0))))))))
                (let* ((maxres ($max q 1))
                       (maxv ($reshape (car maxres) ($size (car maxres) 0))))
                  (when (< ($max ($abs ($- v maxv))) theta)
                    (setf keep-running-p nil))
                  (setf v maxv)
                  (setf argmax (cadr maxres)))))
    (list v (lambda (s) ($ argmax s 0)))))

(let* ((env (th.env.examples:slippery-walk-five-env))
       (p (env-p env))
       (policy (lambda (s) ($ '(0 0 0 0 0 0 0) s))))
  (print-policy policy p :action-symbols '("<" ">") :ncols 7))

(let* ((env (th.env.examples:slippery-walk-five-env))
       (policy (lambda (s) ($ '(0 0 0 0 0 0 0) s))))
  (list :success-rate (probability-success env policy 6)
        :mean-return (mean-return env policy)))

(let* ((env (th.env.examples:slippery-walk-five-env))
       (p (env-p env))
       (policy (lambda (s) ($ '(0 0 0 0 0 0 0) s)))
       (v (policy-evaluation policy p)))
  (print-state-value-function v p :ncols 7))

(let* ((env (th.env.examples:slippery-walk-five-env))
       (p (env-p env))
       (policy (lambda (s) ($ '(0 0 0 0 0 0 0) s)))
       (v (policy-evaluation policy p))
       (new-policy (policy-improvement v p)))
  (print-policy new-policy p :action-symbols '("<" ">") :ncols 7)
  (list :success-rate (probability-success env new-policy 6)
        :mean-return (mean-return env new-policy)))

(let* ((env (th.env.examples:slippery-walk-five-env))
       (p (env-p env))
       (policy (lambda (s) ($ '(0 0 0 0 0 0 0) s)))
       (v (policy-evaluation policy p))
       (new-policy (policy-improvement v p))
       (new-v (policy-evaluation new-policy p)))
  (print-state-value-function new-v p :ncols 7))

(let* ((env (th.env.examples:slippery-walk-five-env))
       (p (env-p env))
       (policy (lambda (s) ($ '(0 0 0 0 0 0 0) s)))
       (v (policy-evaluation policy p))
       (new-policy (policy-improvement v p))
       (new-v (policy-evaluation new-policy p))
       (new-new-policy (policy-improvement new-v p)))
  (print-policy new-new-policy p :action-symbols '("<" ">") :ncols 7)
  (list :success-rate (probability-success env new-new-policy 6)
        :mean-return (mean-return env new-new-policy)))

(let* ((env (th.env.examples:slippery-walk-five-env))
       (p (env-p env))
       (policy (lambda (s) ($ '(0 0 0 0 0 0 0) s)))
       (v (policy-evaluation policy p))
       (new-policy (policy-improvement v p))
       (new-v (policy-evaluation new-policy p))
       (new-new-policy (policy-improvement new-v p))
       (new-new-v (policy-evaluation new-new-policy p)))
  (print-state-value-function new-new-v p :ncols 7)
  ($equal new-v new-new-v))

(let* ((env (th.env.examples:slippery-walk-five-env))
       (p (env-p env))
       (res (policy-iteration p))
       (optimal-value-function (car res))
       (optimal-policy (cadr res)))
  (print-policy optimal-policy p :action-symbols '("<" ">") :ncols 7)
  (print-state-value-function optimal-value-function p :ncols 7)
  (list :success-rate (probability-success env optimal-policy 6)
        :mean-return (mean-return env optimal-policy)))

(let* ((env (th.env.examples:frozen-lake-env))
       (p (env-p env))
       (policy (lambda (s) ($ '(2 0 1 3
                           0 0 2 0
                           3 1 3 0
                           0 2 1 0)
                        s))))
  (print-policy policy p)
  (list :success-rate (probability-success env policy 15)
        :mean-return (mean-return env policy)))

(let* ((env (th.env.examples:frozen-lake-env))
       (p (env-p env))
       (policy (lambda (s) ($ '(2 2 1 0
                           1 0 1 0
                           2 2 1 0
                           0 2 2 0)
                        s))))
  (print-policy policy p)
  (list :success-rate (probability-success env policy 15)
        :mean-return (mean-return env policy)))

(let* ((env (th.env.examples:frozen-lake-env))
       (p (env-p env))
       (policy (lambda (s) ($ '(0 3 3 3
                           0 0 3 0
                           3 1 0 0
                           0 2 2 0)
                        s))))
  (print-policy policy p)
  (list :success-rate (probability-success env policy 15)
        :mean-return (mean-return env policy)))

(let* ((env (th.env.examples:frozen-lake-env))
       (p (env-p env))
       (policy (lambda (s) ($ '(0 3 3 3
                           0 0 3 0
                           3 1 0 0
                           0 2 2 0)
                        s)))
       (v (policy-evaluation policy p :gamma 0.99)))
  (print-state-value-function v p))

(let* ((env (th.env.examples:frozen-lake-env))
       (p (env-p env))
       (policy (lambda (s) ($ '(0 3 3 3
                           0 0 3 0
                           3 1 0 0
                           0 2 2 0)
                        s)))
       (v (policy-evaluation policy p :gamma 0.99))
       (new-policy (policy-improvement v p :gamma 0.99)))
  (print-policy new-policy p)
  (list :success-rate (probability-success env new-policy 15)
        :mean-return (mean-return env new-policy)))

(let* ((env (th.env.examples:frozen-lake-env))
       (p (env-p env))
       (policy (lambda (s) ($ '(0 3 3 3
                           0 0 3 0
                           3 1 0 0
                           0 2 2 0)
                        s)))
       (v (policy-evaluation policy p :gamma 0.99))
       (new-policy (policy-improvement v p :gamma 0.99))
       (new-v (policy-evaluation new-policy p :gamma 0.99)))
  (print-state-value-function new-v p)
  (print-state-value-function ($- new-v v) p))

(let* ((env (th.env.examples:frozen-lake-env))
       (p (env-p env))
       (impres (policy-iteration p :gamma 0.99))
       (v-best (car impres))
       (policy-best (cadr impres)))
  (print-policy policy-best p)
  (print-state-value-function v-best p)
  (list :success-rate (probability-success env policy-best 15)
        :mean-return (mean-return env policy-best)))

(let* ((env (th.env.examples:slippery-walk-five-env))
       (p (env-p env))
       (res (value-iteration p))
       (optimal-value-function (car res))
       (optimal-policy (cadr res)))
  (print-policy optimal-policy p :action-symbols '("<" ">") :ncols 7)
  (print-state-value-function optimal-value-function p :ncols 7)
  (list :success-rate (probability-success env optimal-policy 6)
        :mean-return (mean-return env optimal-policy)))

(let* ((env (th.env.examples:frozen-lake-env))
       (p (env-p env))
       (res (value-iteration p :gamma 0.99))
       (optimal-value-function (car res))
       (optimal-policy (cadr res)))
  (print-policy optimal-policy p)
  (print-state-value-function optimal-value-function p)
  (list :success-rate (probability-success env optimal-policy 15)
        :mean-return (mean-return env optimal-policy)))
