(defpackage :linear-regression
  (:use #:common-lisp
        #:mu
        #:th
        #:th.pp))

(in-package :linear-regression)

(defvar *xs* (tensor '(1 2 3 4 5 6 7 8 9 10)))
(defvar *ys* ($+ ($normal (tensor ($count *xs*)) 0 1) ($* 2 *xs*)))

(setf *xs* (let ((nsz 100))
             (tensor (loop :for n :from 0 :below nsz
                           :collect (* 10D0 (/ n nsz))))))
(setf *ys* ($+ ($normal (tensor ($count *xs*)) 0 1) ($* 2 *xs*)))

(defun lr-posterior (b0 b1 s)
  (let ((prior-b0 (score/normal b0 0 1))
        (prior-b1 (score/normal b1 1 1))
        ;;(prior-s (score/normal s 0 0.1))
        (prior-s (score/uniform s -1 1)))
    (when (and prior-b0 prior-b1 prior-s)
      (let ((likelihood-ys (score/mvn *ys* ($+ ($* b1 *xs*) b0) ($* ($exp s) (eye ($count *xs*))))))
        (when likelihood-ys
          ($+ prior-b0 prior-b1 prior-s likelihood-ys))))))

(let ((traces (mcmc/mh (list (r/variable 0) (r/variable 0) (r/variable 0.5))
                       #'lr-posterior)))
  (prn traces))

;; XXX does not work
(let ((traces (mcmc/hmc (list (r/variable 0.3) (r/variable 1.9) (r/variable 0.4))
                        #'lr-posterior)))
  (prn traces))

(lr-posterior 0.39 1.92 0.08)
(lr-posterior 0.37 1.91 0.06)


(let ((b0 ($parameter 0.39))
      (b1 ($parameter 1.92))
      (s ($parameter 0.08)))
  (prn "L0:" ($data (lr-posterior b0 b1 s)))
  ($cg! (list b0 b1 s))
  (loop :repeat 10000
        :for iter :from 1
        :for p = (lr-posterior b0 b1 s)
        :do (if p
                (let ((loss ($neg p)))
                  (when (zerop (rem iter 1000))
                    (prn iter ($data loss)))
                  ($adgd! (list b0 b1 s) 0.1))
                ($cg! (list b0 b1 s))))
  (prn "LF:" ($data (lr-posterior b0 b1 s)))
  (prn (list b0 b1 s)))

(lr-posterior -0.57 2.15 0.02)
(lr-posterior -0.55 2.11 -0.02)
(lr-posterior -0.52 2.15 0.004)
