(defpackage :linear-regression
  (:use #:common-lisp
        #:mu
        #:th
        #:th.pp))

(in-package :linear-regression)

(defvar *xs* (tensor '(1 2 3 4 5 6 7 8)))
(defvar *ys* ($+ ($normal (tensor ($count *xs*)) 0 1) ($* 2 *xs*)))

(defun lr-posterior (b0 b1 s)
  (let ((prior-b0 (score/normal b0 0 1))
        (prior-b1 (score/normal b1 1 1))
        (prior-s (score/normal s 0 1)))
    (when (and prior-b0 prior-b1 prior-s)
      (let ((likelihood-ys (score/mvn *ys* ($+ ($* b1 *xs*) b0) ($* ($exp s) (eye ($count *xs*))))))
        (when likelihood-ys
          ($+ prior-b0 prior-b1 likelihood-ys))))))

(let ((traces (mcmc/mh (list (r/variable 0) (r/variable 0) (r/variable 0.5))
                       #'lr-posterior)))
  (prn traces))

;; XXX does not work
(let ((traces (mcmc/hmc (list (r/variable 0) (r/variable 0) (r/variable 0.5))
                        #'lr-posterior)))
  (prn traces))
