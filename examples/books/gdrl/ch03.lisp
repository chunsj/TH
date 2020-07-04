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
       (res (env/policy-iteration env))
       (optimal-value-function (policy-iteration/optimal-value-function res))
       (optimal-policy (policy-iteration/optimal-policy res)))
  (env/print-policy env optimal-policy :action-symbols '("<" ">") :ncols 7)
  (env/print-state-value-function env optimal-value-function :ncols 7)
  (list :success-rate (env/success-probability env optimal-policy 6)
        :mean-return (env/mean-return env optimal-policy)))

(let* ((env (th.env.examples:frozen-lake-env))
       (policy (lambda (s) ($ '(2 0 1 3
                           0 0 2 0
                           3 1 3 0
                           0 2 1 0)
                        s))))
  (env/print-policy env policy)
  (list :success-rate (env/success-probability env policy 15)
        :mean-return (env/mean-return env policy)))

(let* ((env (th.env.examples:frozen-lake-env))
       (policy (lambda (s) ($ '(2 2 1 0
                           1 0 1 0
                           2 2 1 0
                           0 2 2 0)
                        s))))
  (env/print-policy env policy)
  (list :success-rate (env/success-probability env policy 15)
        :mean-return (env/mean-return env policy)))

(let* ((env (th.env.examples:frozen-lake-env))
       (policy (lambda (s) ($ '(0 3 3 3
                           0 0 3 0
                           3 1 0 0
                           0 2 2 0)
                        s))))
  (env/print-policy env policy)
  (list :success-rate (env/success-probability env policy 15)
        :mean-return (env/mean-return env policy)))

(let* ((env (th.env.examples:frozen-lake-env))
       (policy (lambda (s) ($ '(0 3 3 3
                           0 0 3 0
                           3 1 0 0
                           0 2 2 0)
                        s)))
       (v (env/policy-evaluation env policy :gamma 0.99)))
  (env/print-state-value-function env v))

(let* ((env (th.env.examples:frozen-lake-env))
       (policy (lambda (s) ($ '(0 3 3 3
                           0 0 3 0
                           3 1 0 0
                           0 2 2 0)
                        s)))
       (v (env/policy-evaluation env policy :gamma 0.99))
       (new-policy (env/policy-improvement env v :gamma 0.99)))
  (env/print-policy env new-policy)
  (list :success-rate (env/success-probability env new-policy 15)
        :mean-return (env/mean-return env new-policy)))

(let* ((env (th.env.examples:frozen-lake-env))
       (policy (lambda (s) ($ '(0 3 3 3
                           0 0 3 0
                           3 1 0 0
                           0 2 2 0)
                        s)))
       (v (env/policy-evaluation env policy :gamma 0.99))
       (new-policy (env/policy-improvement env v :gamma 0.99))
       (new-v (env/policy-evaluation env new-policy :gamma 0.99)))
  (env/print-state-value-function env new-v)
  (env/print-state-value-function env ($- new-v v)))

(let* ((env (th.env.examples:frozen-lake-env))
       (impres (env/policy-iteration env :gamma 0.99))
       (v-best (policy-iteration/optimal-value-function impres))
       (policy-best (policy-iteration/optimal-policy impres)))
  (env/print-policy env policy-best)
  (env/print-state-value-function env v-best)
  (list :success-rate (env/success-probability env policy-best 15)
        :mean-return (env/mean-return env policy-best)))

(let* ((env (th.env.examples:slippery-walk-five-env))
       (res (env/value-iteration env))
       (optimal-value-function (value-iteration/optimal-value-function res))
       (optimal-policy (value-iteration/optimal-policy res)))
  (env/print-policy env optimal-policy :action-symbols '("<" ">") :ncols 7)
  (env/print-state-value-function env optimal-value-function :ncols 7)
  (list :success-rate (env/success-probability env optimal-policy 6)
        :mean-return (env/mean-return env optimal-policy)))

(let* ((env (th.env.examples:frozen-lake-env))
       (res (env/value-iteration env :gamma 0.99))
       (optimal-value-function (value-iteration/optimal-value-function res))
       (optimal-policy (value-iteration/optimal-policy res)))
  (env/print-policy env optimal-policy)
  (env/print-state-value-function env optimal-value-function)
  (list :success-rate (env/success-probability env optimal-policy 15)
        :mean-return (env/mean-return env optimal-policy)))
