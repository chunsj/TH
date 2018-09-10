;; from
;; https://jellis18.github.

(defpackage :mh-sample
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :mh-sample)

(defun mh-sampler (x0 lnprob-fn prop-fn &key prob-fn-args (iterations 100000))
  (let* ((ndim ($size x0 0))
         (chain (zeros iterations ndim))
         (lnprob (zeros iterations))
         (accept-rate (zeros iterations))
         (lnprob0 nil)
         (naccept 0))
    (setf ($ chain 0) x0)
    (setf lnprob0 (funcall lnprob-fn x0))
    (setf ($ lnprob 0) lnprob0)
    (loop :for i :from 1 :below iterations
          :for pres = (apply prop-fn (cons x0 prob-fn-args))
          :for x-star = (getf pres :new)
          :for factor = (getf pres :factor)
          :for u = (apply #'rnd ($size x0))
          :for lnprob-star = (funcall lnprob-fn x-star)
          :for h = ($* factor ($exp ($- lnprob-star lnprob0)))
          :do (let ((nx0 ($clone x0)))
                ;; (prn u h)
                ;; XXX how to deal with (< u h) for multidimensional x?
                (setf ($ nx0 ($lt u h)) x-star)
                (setf ($ lnprob0 ($lt u h)) lnprob-star)
                (when (< u h)
                  (setf x0 x-star)
                  (setf lnprob0 lnprob-star)
                  (incf naccept))
                (setf ($ chain i) x0)
                (setf ($ lnprob i) lnprob0)
                (setf ($ accept-rate i) (* 1D0 (/ naccept i)))))
    (list :chain chain
          :acceptance-rate accept-rate
          :log-posteriors lnprob)))

;; gaussian proposal
(defun gaussian-proposal (x &key (sigma 0.1))
  (let ((x-star ($+ x ($* (rndn ($size x 0)) sigma)))
        (qxx 1))
    (list :new x-star
          :factor qxx)))

(defun simple-gaussian-lnpost (x)
  (let ((mu 0)
        (std 1)
        (r (apply #'zeros ($size x))))
    ($fill! r -1E6)
    (setf ($ r ($* ($lt x 10) ($gt x -10)))
          ($* -0.5 ($/ ($expt ($- x mu) 2) (* std std))))
    r))

;; XXX not work yet
(mh-sampler (tensor '(0 0 0)) #'simple-gaussian-lnpost #'gaussian-proposal :iterations 10)
