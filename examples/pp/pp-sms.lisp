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

(defun sms-posterior2 (switch-point delta early-mean mid-mean late-mean)
  (let ((data *sms*)
        (n ($count *sms*))
        (sw2 (+ switch-point delta)))
    (when (< sw2 (- n 2))
      (let ((prior-switch-point1 (score/discrete-uniform switch-point 1 (- sw2 2)))
            (prior-switch-point2 (score/discrete-uniform delta 1 (- n sw2 2)))
            (prior-early-mean (score/exponential early-mean *rate*))
            (prior-mid-mean (score/exponential mid-mean *rate*))
            (prior-late-mean (score/exponential late-mean *rate*)))
        (when (and prior-switch-point1
                   prior-switch-point2
                   prior-early-mean
                   prior-mid-mean
                   prior-late-mean)
          (let ((rate (-> ($one data)
                          ($fill! early-mean))))
            (setf ($slice rate switch-point sw2) mid-mean)
            (setf ($slice rate sw2) late-mean)
            (let ((likelihood-mean (score/poisson data rate)))
              (when likelihood-mean
                ($+ prior-switch-point1
                    prior-switch-point2
                    prior-early-mean
                    prior-mid-mean
                    prior-late-mean
                    likelihood-mean)))))))))

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
 (let ((traces (mcmc/mh '(40 10 20.0 20.0 20.0) #'sms-posterior2)))
   (loop :for trc :in traces
         :for lbl :in '(:switch-point1 :switch-point2 :early-mean :mid-mean :late-mean)
         :do (prn lbl trc (trace/hpd trc) (trace/act trc)))))

(time
 (let ((traces (mcmc/mh '(40 10 20.0 20.0 20.0) #'sms-posterior2 :type :em)))
   (loop :for trc :in traces
         :for lbl :in '(:switch-point1 :switch-point2 :early-mean :mid-mean :late-mean)
         :do (prn lbl trc (trace/hpd trc) (trace/act trc)))))

(time
 (let ((traces (mcmc/mh '(40 10 20.0 20.0 20.0) #'sms-posterior2 :type :ae)))
   (loop :for trc :in traces
         :for lbl :in '(:switch-point1 :switch-point2 :early-mean :mid-mean :late-mean)
         :do (prn lbl trc (trace/hpd trc) (trace/act trc)))))

(time
 (let ((traces (mcmc/mh '(40 10 20.0 20.0 20.0) #'sms-posterior2 :type :sc)))
   (loop :for trc :in traces
         :for lbl :in '(:switch-point1 :switch-point2 :early-mean :mid-mean :late-mean)
         :do (prn lbl trc (trace/hpd trc) (trace/act trc)))))

(defun sms-posterior3 (early-mean mid-mean late-mean)
  (let ((switch-point 45)
        (delta 2))
    (let ((data *sms*)
          (n ($count *sms*))
          (sw2 (+ switch-point delta)))
      (when (< sw2 (- n 2))
        (let ((prior-switch-point1 (score/discrete-uniform switch-point 1 (- sw2 2)))
              (prior-switch-point2 (score/discrete-uniform delta 1 (- n sw2 2)))
              (prior-early-mean (score/exponential early-mean *rate*))
              (prior-mid-mean (score/exponential mid-mean *rate*))
              (prior-late-mean (score/exponential late-mean *rate*)))
          (when (and prior-switch-point1
                     prior-switch-point2
                     prior-early-mean
                     prior-mid-mean
                     prior-late-mean)
            (let ((rate (-> ($one data)
                            ($fill! early-mean))))
              (setf ($slice rate switch-point sw2) mid-mean)
              (setf ($slice rate sw2) late-mean)
              (let ((likelihood-mean (score/poisson data rate)))
                (when likelihood-mean
                  ($+ prior-switch-point1
                      prior-switch-point2
                      prior-early-mean
                      prior-mid-mean
                      prior-late-mean
                      likelihood-mean))))))))))

(time
 (let ((traces (mcmc/mh '(2.0 2.0 2.0) #'sms-posterior3 :type :am)))
   (loop :for trc :in traces
         :for lbl :in '(:early-mean :mid-mean :late-mean)
         :do (prn lbl trc (trace/hpd trc) (trace/act trc)))))
