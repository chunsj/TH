(defpackage :linear-regression
  (:use #:common-lisp
        #:mu
        #:th
        #:th.pp))

(in-package :linear-regression)

(defparameter *xs* (tensor (loop :for i :from 1 :to 20 :collect i)))
(defparameter *ys* ($+ ($normal (tensor ($count *xs*)) 0 1) ($* 2 *xs*)))

(defun lr-posterior (b0 b1 s)
  (let ((prior-b0 (score/normal b0 0 1))
        (prior-b1 (score/normal b1 1 1))
        (prior-s (score/uniform s 0.001 10)))
    (when (and prior-b0 prior-b1 prior-s)
      (let ((m ($* s (eye ($count *xs*)))))
        (let ((likelihood-ys (score/mvn *ys* ($+ ($* b1 *xs*) b0) m)))
          (when likelihood-ys
            ($+ prior-b0 prior-b1 prior-s likelihood-ys)))))))

;; MH - WORKS
(let ((traces (mcmc/mh (list (r/variable 0) (r/variable 0) (r/variable 0.5))
                       #'lr-posterior)))
  (prn traces))

;; HMC - REQUIRES PROPER INITIAL POINTS
(let ((traces (mcmc/hmc (list (r/variable 0) (r/variable 2) (r/variable 1))
                        #'lr-posterior)))
  (prn traces))

;; FIND PROPER INITIAL POINTS
(let ((b0 ($parameter 0))
      (b1 ($parameter 1)))
  (loop :repeat 10000
        :for iter :from 1
        :for loss = ($sum ($square ($sub ($+ b0 ($* b1 *xs*)) *ys*)))
        :do ($amgd! (list b0 b1)))
  (let ((loss ($sum ($square ($sub ($+ b0 ($* b1 *xs*)) *ys*)))))
    (prn (list ($data b0) ($data b1) ($sqrt ($/ ($data loss) ($count *xs*)))))
    ;; HMC WITH PROPER INITIAL POINTS - WORKS
    (let ((traces (mcmc/hmc (list (r/variable ($data b0)) (r/variable ($data b1))
                                  (r/variable ($sqrt ($/ ($data loss) ($count *xs*)))))
                            #'lr-posterior)))
      (prn traces))))

;; NUTS - REQUIRES PROPER INITIAL POINTS
(let ((traces (mcmc/nuts (list (r/variable 0) (r/variable 2) (r/variable 1))
                         #'lr-posterior)))
  (prn traces))

;; FIND PROPER INITIAL POINTS
(let ((b0 ($parameter 0))
      (b1 ($parameter 1)))
  (loop :repeat 10000
        :for iter :from 1
        :for loss = ($sum ($square ($sub ($+ b0 ($* b1 *xs*)) *ys*)))
        :do ($amgd! (list b0 b1)))
  (let ((loss ($sum ($square ($sub ($+ b0 ($* b1 *xs*)) *ys*)))))
    (prn (list ($data b0) ($data b1) ($sqrt ($/ ($data loss) ($count *xs*)))))
    ;; HMC WITH PROPER INITIAL POINTS - WORKS
    (let ((traces (mcmc/nuts (list (r/variable ($data b0)) (r/variable ($data b1))
                                   (r/variable ($sqrt ($/ ($data loss) ($count *xs*)))))
                             #'lr-posterior)))
      (prn traces))))
