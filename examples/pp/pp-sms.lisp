(defpackage :pp-sms
  (:use #:common-lisp
        #:mu
        #:th
        #:th.pp))

(in-package :pp-sms)

(defparameter *sms* (->> (slurp "./data/sms.txt")
                         (mapcar #'parse-float)
                         (mapcar #'round)
                         (tensor)))

(defparameter *mean* ($mean *sms*))
(defparameter *rate* (/ 1D0 *mean*))

(defun sms-posterior (switch-point early-mean late-mean)
  (let ((data *sms*))
    (let ((prior-switch-point (score/discrete-uniform switch-point 1 (- ($count data) 2)))
          (prior-early-mean (score/exponential early-mean *rate*))
          (prior-late-mean (score/exponential late-mean *rate*)))
      (when (and prior-switch-point
                 prior-early-mean
                 prior-late-mean)
        (let ((sms-early ($slice data 0 switch-point))
              (sms-late ($slice data switch-point)))
          (let ((likelihood-early-mean (score/poisson sms-early early-mean))
                (likelihood-late-mean (score/poisson sms-late late-mean)))
            (when (and likelihood-early-mean
                       likelihood-late-mean)
              ($+ prior-switch-point
                  prior-early-mean
                  prior-late-mean
                  likelihood-early-mean
                  likelihood-late-mean))))))))

(defun sms-posterior2 (switch-point1 switch-point2 early-mean mid-mean late-mean)
  (let ((data *sms*)
        (n ($count *sms*))
        (switch-point1 (min switch-point1 switch-point2))
        (switch-point2 (max switch-point1 switch-point2)))
    (when (< switch-point1 switch-point2)
      (let ((prior-switch-point1 (score/discrete-uniform switch-point1 1 (- n 3)))
            (prior-switch-point2 (score/discrete-uniform switch-point2 switch-point1 (- n 2)))
            (prior-early-mean (score/exponential early-mean *rate*))
            (prior-mid-mean (score/exponential mid-mean *rate*))
            (prior-late-mean (score/exponential late-mean *rate*)))
        (when (and prior-switch-point1
                   prior-switch-point2
                   prior-early-mean
                   prior-mid-mean
                   prior-late-mean)
          (let ((sms-early ($slice data 0 switch-point1))
                (sms-mid ($slice data switch-point1 switch-point2))
                (sms-late ($slice data switch-point2)))
            (let ((likelihood-early-mean (score/poisson sms-early early-mean))
                  (likelihood-mid-mean (score/poisson sms-mid mid-mean))
                  (likelihood-late-mean (score/poisson sms-late late-mean)))
              (when (and likelihood-early-mean
                         likelihood-mid-mean
                         likelihood-late-mean)
                ($+ prior-switch-point1
                    prior-switch-point2
                    prior-early-mean
                    prior-mid-mean
                    prior-late-mean
                    likelihood-early-mean
                    likelihood-mid-mean
                    likelihood-late-mean)))))))))

(time
 (let ((traces (mcmc/mh '(37 20.0 20.0) #'sms-posterior)))
   (loop :for trc :in traces
         :for lbl :in '(:switch-point :early-mean :late-mean)
         :do (prn lbl trc (trace/hpd trc) (trace/act trc)))))

(time
 (let ((traces (mcmc/mh '(37 20.0 20.0) #'sms-posterior :type :em)))
   (loop :for trc :in traces
         :for lbl :in '(:switch-point :early-mean :late-mean)
         :do (prn lbl trc (trace/hpd trc) (trace/act trc)))))

(time
 (let ((traces (mcmc/mh '(37 20.0 20.0) #'sms-posterior :type :ae)))
   (loop :for trc :in traces
         :for lbl :in '(:switch-point :early-mean :late-mean)
         :do (prn lbl trc (trace/hpd trc) (trace/act trc)))))

(time
 (let ((traces (mcmc/mh '(37 20.0 20.0) #'sms-posterior :type :sc)))
   (loop :for trc :in traces
         :for lbl :in '(:switch-point :early-mean :late-mean)
         :do (prn lbl trc (trace/hpd trc) (trace/act trc)))))

(time
 (let ((traces (mcmc/mh '(40 50 20.0 20.0 20.0) #'sms-posterior2)))
   (loop :for trc :in traces
         :for lbl :in '(:switch-point1 :switch-point2 :early-mean :mid-mean :late-mean)
         :do (prn lbl trc (trace/hpd trc) (trace/act trc)))))

(time
 (let ((traces (mcmc/mh '(40 50 20.0 20.0 20.0) #'sms-posterior2 :type :em)))
   (loop :for trc :in traces
         :for lbl :in '(:switch-point1 :switch-point2 :early-mean :mid-mean :late-mean)
         :do (prn lbl trc (trace/hpd trc) (trace/act trc)))))

(time
 (let ((traces (mcmc/mh '(40 50 20.0 20.0 20.0) #'sms-posterior2 :type :ae)))
   (loop :for trc :in traces
         :for lbl :in '(:switch-point1 :switch-point2 :early-mean :mid-mean :late-mean)
         :do (prn lbl trc (trace/hpd trc) (trace/act trc)))))

(time
 (let ((traces (mcmc/mh '(40 50 20.0 20.0 20.0) #'sms-posterior2 :type :sc)))
   (loop :for trc :in traces
         :for lbl :in '(:switch-point1 :switch-point2 :early-mean :mid-mean :late-mean)
         :do (prn lbl trc (trace/hpd trc) (trace/act trc)))))
