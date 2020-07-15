(defpackage :gdrl-ch07
  (:use #:common-lisp
        #:mu
        #:th
        #:th.env)
  (:import-from #:th.env.examples))

(in-package :gdrl-ch07)

(defun decay-schedule (v0 minv decay-ratio max-steps &key (log-start -2) (log-base 10))
  (let* ((decay-steps (round (* max-steps decay-ratio)))
         (rem-steps (- max-steps decay-steps))
         (vs (-> ($/ (logspace log-start 0 decay-steps) (log log-base 10))
                 ($list)
                 (reverse)
                 (tensor)))
         (minvs ($min vs))
         (maxvs ($max vs))
         (rngv (- maxvs minvs))
         (vs ($/ ($- vs minvs) rngv))
         (vs ($+ minv ($* vs (- v0 minv)))))
    ($cat vs ($fill! (tensor rem-steps) ($last vs)))))

(let* ((env (th.env.examples:slippery-walk-seven-env))
       (optres (env/value-iteration env :gamma 0.99D0))
       (opt-v (value-iteration/optimal-value-function optres))
       (opt-p (value-iteration/optimal-policy optres))
       (opt-q (value-iteration/optimal-action-value-function optres)))
  (env/print-state-value-function env opt-v :ncols 9)
  (env/print-policy env opt-p :action-symbols '("<" ">") :ncols 9)
  (prn opt-q))
