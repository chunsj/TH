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
        (prior-s (score/normal s 0 0.1)))
    (when (and prior-b0 prior-b1 prior-s)
      (let ((likelihood-ys (score/mvn *ys* ($+ ($* b1 *xs*) b0) ($* ($exp s) (eye ($count *xs*))))))
        (when likelihood-ys
          ($+ prior-b0 prior-b1 prior-s likelihood-ys))))))

(let ((traces (mcmc/mh (list (r/variable 0) (r/variable 0) (r/variable 0.5))
                       #'lr-posterior)))
  (prn traces))

;; XXX does not work
(let ((traces (mcmc/hmc (list (r/variable 0) (r/variable 1) (r/variable 0.1))
                        #'lr-posterior)))
  (prn traces))


(let ((b0 ($parameter -0.57))
      (b1 ($parameter 2.15))
      (s ($parameter 0.02)))
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
(lr-posterior -0.52 2.15 0.004)
(lr-posterior -0.54 2.14 0.004)
(lr-posterior -0.53755 2.1339 0.004)
(lr-posterior -0.483 2.092 0.025)
(lr-posterior -0.33 2.1 0.36)

(let ((b0 ($parameter -0.52))
      (b1 ($parameter 2.15))
      (s ($parameter 0.01)))
  (prn "L0:" ($data (lr-posterior b0 b1 s)))
  ($cg! (list b0 b1))
  (loop :repeat 50000
        :for iter :from 1
        :for p = (lr-posterior b0 b1 s)
        :do (if p
                (let ((loss ($neg p)))
                  (when (zerop (rem iter 1000))
                    (prn iter ($data loss)))
                  ($adgd! (list b0 b1)))
                ($cg! (list b0 b1))))
  (prn "LF:" ($data (lr-posterior b0 b1 s)))
  (prn (list b0 b1 s)))

(let ((x ($parameter (tensor '((1 2 3) (4 5 6) (7 8 9))))))
  (prn ($sum ($diag ($data x))))
  (loop :repeat 10000
        :for loss = ($square ($sub 1 ($sum ($diag x))))
        :do ($amgd! x))
  ($sum ($diag ($data x))))

(let (;;(data (tensor '(0.5377    3.5784   -0.1241    0.4889   -1.0689)))
      (data (tensor '(-0.4336    0.7147    0.7172    0.8884    1.3703))))
  (exp (score/mvn data (zeros 5) (eye 5))))
