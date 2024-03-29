(defpackage :pp-disaster
  (:use #:common-lisp
        #:mu
        #:th
        #:th.pp))

(in-package :pp-disaster)

;; mining disaster problem
(defparameter *disasters* (tensor
                           '(4 5 4 0 1 4 3 4 0 6 3 3 4 0 2 6
                             3 3 5 4 5 3 1 4 4 1 5 5 3 4 2 5
                             2 2 3 4 2 1 3 2 2 1 1 1 1 3 0 0
                             1 0 1 1 0 0 3 1 0 3 2 2 0 1 1 1
                             0 1 0 1 0 0 0 2 1 0 0 0 1 1 0 2
                             3 3 1 1 2 1 1 1 1 2 4 2 0 0 1 4
                             0 0 0 1 0 0 0 0 0 1 0 0 1 0 1)))
(defparameter *mean* ($mean *disasters*))
(defparameter *rate* (/ 1D0 *mean*))

(defun disaster-posterior (switch-point early-mean late-mean)
  (let ((prior-switch-point (score/discrete-uniform switch-point 1 (- ($count *disasters*) 2)))
        (prior-early-mean (score/exponential early-mean *rate*))
        (prior-late-mean (score/exponential late-mean *rate*)))
    (when (and prior-switch-point
               prior-early-mean
               prior-late-mean)
      (let ((disasters-early ($slice *disasters* 0 switch-point))
            (disasters-late ($slice *disasters* switch-point)))
        (let ((likelihood-early-mean (score/poisson disasters-early early-mean))
              (likelihood-late-mean (score/poisson disasters-late late-mean)))
          (when (and likelihood-early-mean
                     likelihood-late-mean)
            ($+ prior-switch-point
                prior-early-mean
                prior-late-mean
                likelihood-early-mean
                likelihood-late-mean)))))))

(defparameter *ts* nil)

(time
 (let ((traces (mcmc/mh '(50 2.0 2.0) #'disaster-posterior)))
   (setf *ts* traces)
   (loop :for trc :in traces
         :for lbl :in '(:switch-point :early-mean :late-mean)
         :do (prn lbl trc))))

;; fitting
(prn (map/fit #'disaster-posterior '(50 2.0 2.0)))

;; checking autocorrelation
(prn ($slice (trace/acr ($0 *ts*)) 0 20))

;; thinning
(let ((nt (trace/thin ($0 *ts*) 13)))
  (prn ($slice (trace/acr nt) 0 10)))
