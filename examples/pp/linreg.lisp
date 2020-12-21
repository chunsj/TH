(defpackage :linear-regression
  (:use #:common-lisp
        #:mu
        #:th
        #:th.pp))

(in-package :linear-regression)

(defparameter *xs* (tensor (loop :for i :from 1 :to 200 :collect i)))
(defparameter *ys* ($+ ($normal (tensor ($count *xs*)) 0 1) ($* 2 *xs*)))

(defun lr-posterior (b0 b1 s)
  (let ((prior-b0 (score/normal b0 0 1))
        (prior-b1 (score/normal b1 1 1))
        (prior-s (score/normal s 0 1)))
    (when (and prior-b0 prior-b1 prior-s)
      (let ((ms ($add b0 ($mul b1 *xs*)))
            (ll-failed nil))
        (let ((lls (loop :for i :from 0 :below ($count *xs*)
                         :for m = ($ ms i)
                         :for y = ($ *ys* i)
                         :for ll = (score/gaussian y m ($exp s))
                         :collect (progn
                                    (when (null ll)
                                      (setf ll-failed T))
                                    ll))))
          (unless ll-failed
            (let ((likelihood-ys (reduce #'$add lls)))
              ($+ ($+ prior-b0 prior-b1 prior-s likelihood-ys)))))))))

(let ((traces (mcmc/mh (list (r/variable 0) (r/variable 1) (r/variable 0))
                       #'lr-posterior)))
  (prn traces))
