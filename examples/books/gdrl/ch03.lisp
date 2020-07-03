(defpackage :gdrl-ch03
  (:use #:common-lisp
        #:mu
        #:th
        #:th.env)
  (:import-from #:th.env.examples))

(in-package :gdrl-ch03)

(let* ((env (th.env.examples:slippery-walk-five-env))
       (policy (lambda (s) ($ '(0 0 0 0 0 0 0) s))))
  (env/print-policy env policy :action-symbols '("<" ">") :ncols 7))

(let* ((env (th.env.examples:slippery-walk-five-env))
       (policy (lambda (s) ($ '(0 0 0 0 0 0 0) s))))
  (list :success-rate (env/success-probability env policy 6)
        :mean-return (env/mean-return env policy)))

(let* ((env (th.env.examples:slippery-walk-five-env))
       (policy (lambda (s) ($ '(0 0 0 0 0 0 0) s)))
       (v (env/policy-evaluation env policy)))
  (env/print-state-value-function env v :ncols 7))

(let* ((env (th.env.examples:slippery-walk-five-env))
       (policy (lambda (s) ($ '(0 0 0 0 0 0 0) s)))
       (v (env/policy-evaluation env policy))
       (new-policy (env/policy-improvement env v)))
  (env/print-policy env new-policy :action-symbols '("<" ">") :ncols 7)
  (list :success-rate (env/success-probability env new-policy 6)
        :mean-return (env/mean-return env new-policy)))

(let* ((env (th.env.examples:slippery-walk-five-env))
       (policy (lambda (s) ($ '(0 0 0 0 0 0 0) s)))
       (v (env/policy-evaluation env policy))
       (new-policy (env/policy-improvement env v))
       (new-v (env/policy-evaluation env new-policy)))
  (env/print-state-value-function env new-v :ncols 7))

(let* ((env (th.env.examples:slippery-walk-five-env))
       (policy (lambda (s) ($ '(0 0 0 0 0 0 0) s)))
       (v (env/policy-evaluation env policy))
       (new-policy (env/policy-improvement env v))
       (new-v (env/policy-evaluation env new-policy))
       (new-new-policy (env/policy-improvement env new-v)))
  (env/print-policy env new-new-policy :action-symbols '("<" ">") :ncols 7)
  (list :success-rate (env/success-probability env new-new-policy 6)
        :mean-return (env/mean-return env new-new-policy)))

(let* ((env (th.env.examples:slippery-walk-five-env))
       (policy (lambda (s) ($ '(0 0 0 0 0 0 0) s)))
       (v (env/policy-evaluation env policy))
       (new-policy (env/policy-improvement env v))
       (new-v (env/policy-evaluation env new-policy))
       (new-new-policy (env/policy-improvement env new-v))
       (new-new-v (env/policy-evaluation env new-new-policy)))
  (env/print-state-value-function env new-new-v :ncols 7)
  ($equal new-v new-new-v))

(let* ((env (th.env.examples:slippery-walk-five-env))
       (p (env/p env))
       (res (policy-iteration p))
       (optimal-value-function (car res))
       (optimal-policy (cadr res)))
  (env/print-policy optimal-policy p :action-symbols '("<" ">") :ncols 7)
  (env/print-state-value-function optimal-value-function p :ncols 7)
  (list :success-rate (env/success-probability env optimal-policy 6)
        :mean-return (env/mean-return env optimal-policy)))

(let* ((env (th.env.examples:frozen-lake-env))
       (p (env/p env))
       (policy (lambda (s) ($ '(2 0 1 3
                           0 0 2 0
                           3 1 3 0
                           0 2 1 0)
                        s))))
  (env/print-policy policy p)
  (list :success-rate (env/success-probability env policy 15)
        :mean-return (env/mean-return env policy)))

(let* ((env (th.env.examples:frozen-lake-env))
       (p (env/p env))
       (policy (lambda (s) ($ '(2 2 1 0
                           1 0 1 0
                           2 2 1 0
                           0 2 2 0)
                        s))))
  (env/print-policy policy p)
  (list :success-rate (env/success-probability env policy 15)
        :mean-return (env/mean-return env policy)))

(let* ((env (th.env.examples:frozen-lake-env))
       (p (env/p env))
       (policy (lambda (s) ($ '(0 3 3 3
                           0 0 3 0
                           3 1 0 0
                           0 2 2 0)
                        s))))
  (env/print-policy policy p)
  (list :success-rate (env/success-probability env policy 15)
        :mean-return (env/mean-return env policy)))

(let* ((env (th.env.examples:frozen-lake-env))
       (p (env/p env))
       (policy (lambda (s) ($ '(0 3 3 3
                           0 0 3 0
                           3 1 0 0
                           0 2 2 0)
                        s)))
       (v (env/policy-evaluation policy p :gamma 0.99)))
  (env/print-state-value-function v p))

(let* ((env (th.env.examples:frozen-lake-env))
       (p (env/p env))
       (policy (lambda (s) ($ '(0 3 3 3
                           0 0 3 0
                           3 1 0 0
                           0 2 2 0)
                        s)))
       (v (env/policy-evaluation policy p :gamma 0.99))
       (new-policy (env/policy-improvement v p :gamma 0.99)))
  (env/print-policy new-policy p)
  (list :success-rate (env/success-probability env new-policy 15)
        :mean-return (env/mean-return env new-policy)))

(let* ((env (th.env.examples:frozen-lake-env))
       (p (env/p env))
       (policy (lambda (s) ($ '(0 3 3 3
                           0 0 3 0
                           3 1 0 0
                           0 2 2 0)
                        s)))
       (v (env/policy-evaluation policy p :gamma 0.99))
       (new-policy (env/policy-improvement v p :gamma 0.99))
       (new-v (env/policy-evaluation new-policy p :gamma 0.99)))
  (env/print-state-value-function new-v p)
  (env/print-state-value-function ($- new-v v) p))

(let* ((env (th.env.examples:frozen-lake-env))
       (p (env/p env))
       (impres (policy-iteration p :gamma 0.99))
       (v-best (car impres))
       (policy-best (cadr impres)))
  (env/print-policy policy-best p)
  (env/print-state-value-function v-best p)
  (list :success-rate (env/success-probability env policy-best 15)
        :mean-return (env/mean-return env policy-best)))

(let* ((env (th.env.examples:slippery-walk-five-env))
       (p (env/p env))
       (res (env/value-iteration p))
       (optimal-value-function (car res))
       (optimal-policy (cadr res)))
  (env/print-policy optimal-policy p :action-symbols '("<" ">") :ncols 7)
  (env/print-state-value-function optimal-value-function p :ncols 7)
  (list :success-rate (env/success-probability env optimal-policy 6)
        :mean-return (env/mean-return env optimal-policy)))

(let* ((env (th.env.examples:frozen-lake-env))
       (p (env/p env))
       (res (env/value-iteration p :gamma 0.99))
       (optimal-value-function (car res))
       (optimal-policy (cadr res)))
  (env/print-policy optimal-policy p)
  (env/print-state-value-function optimal-value-function p)
  (list :success-rate (env/success-probability env optimal-policy 15)
        :mean-return (env/mean-return env optimal-policy)))
