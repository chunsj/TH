(defpackage :gdrl-ch03
  (:use #:common-lisp
        #:mu
        #:th
        #:th.env)
  (:import-from #:th.env.examples))

(in-package :gdrl-ch03)

(let* ((env (th.env.examples:slippery-walk-five-env))
       (policy (lambda (s) ($ '(0 0 0 0 0 0 0) s))))
  (print-policy env policy :action-symbols '("<" ">") :ncols 7))

(let* ((env (th.env.examples:slippery-walk-five-env))
       (policy (lambda (s) ($ '(0 0 0 0 0 0 0) s))))
  (list :success-rate (success-probability env policy 6)
        :mean-return (mean-return env policy)))

(let* ((env (th.env.examples:slippery-walk-five-env))
       (policy (lambda (s) ($ '(0 0 0 0 0 0 0) s)))
       (v (policy-evaluation env policy)))
  (print-state-value-function env v :ncols 7))

(let* ((env (th.env.examples:slippery-walk-five-env))
       (policy (lambda (s) ($ '(0 0 0 0 0 0 0) s)))
       (v (policy-evaluation env policy))
       (new-policy (policy-improvement env v)))
  (print-policy env new-policy :action-symbols '("<" ">") :ncols 7)
  (list :success-rate (success-probability env new-policy 6)
        :mean-return (mean-return env new-policy)))

;; xxx
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
