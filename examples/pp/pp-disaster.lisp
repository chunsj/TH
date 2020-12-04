(defpackage :pp-disaster
  (:use #:common-lisp
        #:mu
        #:th
        #:th.pp))

(in-package :pp-disaster)

;; mining disaster problem
(defvar *disasters* '(4 5 4 0 1 4 3 4 0 6 3 3 4 0 2 6
                      3 3 5 4 5 3 1 4 4 1 5 5 3 4 2 5
                      2 2 3 4 2 1 3 2 2 1 1 1 1 3 0 0
                      1 0 1 1 0 0 3 1 0 3 2 2 0 1 1 1
                      0 1 0 1 0 0 0 2 1 0 0 0 1 1 0 2
                      3 3 1 1 2 1 1 1 1 2 4 2 0 0 1 4
                      0 0 0 1 0 0 0 0 0 1 0 0 1 0 1))
(defvar *mean* ($mean *disasters*))
(defvar *rate* (/ 1D0 *mean*))

;; for SMS problem
(setf *disasters* (->> (slurp "./data/sms.txt")
                       (mapcar #'parse-float)
                       (mapcar #'round)))
(setf *mean* ($mean *disasters*))
(setf *rate* (/ 1D0 ($mean *disasters*)))

(defun disaster-posterior (switch-point early-mean late-mean)
  (let ((prior-switch-point (score/discrete-uniform switch-point 1 (- ($count *disasters*) 2)))
        (prior-early-mean (score/exponential early-mean *rate*))
        (prior-late-mean (score/exponential late-mean *rate*)))
    (when (and prior-switch-point
               prior-early-mean
               prior-late-mean)
      (let ((disasters-early (tensor (subseq *disasters* 0 switch-point)))
            (disasters-late (tensor (subseq *disasters* switch-point))))
        (let ((likelihood-early-mean (score/poisson disasters-early early-mean))
              (likelihood-late-mean (score/poisson disasters-late late-mean)))
          (when (and likelihood-early-mean
                     likelihood-late-mean)
            ($+ prior-switch-point
                prior-early-mean
                prior-late-mean
                likelihood-early-mean
                likelihood-late-mean)))))))

(time
 (let ((r/switch-point (r/variable (round (* 0.5 ($count *disasters*))) :discrete))
       (r/early-mean (r/variable *mean*))
       (r/late-mean (r/variable *mean*)))
   (let ((traces (mcmc/mh (list r/switch-point r/early-mean r/late-mean)
                          #'disaster-posterior)))
     (loop :for trc :in traces
           :for lbl :in '(:switch-point :early-mean :late-mean)
           :do (prn lbl trc)))))

(time
 (let ((r/switch-point (r/variable (round (* 0.5 ($count *disasters*))) :discrete))
       (r/early-mean (r/variable *mean*))
       (r/late-mean (r/variable *mean*)))
   (let ((traces (mcmc/hmc (list r/switch-point r/early-mean r/late-mean)
                           #'disaster-posterior
                           :iterations 2000)))
     (loop :for trc :in traces
           :for lbl :in '(:switch-point :early-mean :late-mean)
           :do (prn lbl trc)))))
