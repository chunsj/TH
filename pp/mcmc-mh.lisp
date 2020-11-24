(in-package :th.pp)

(defgeneric proposal/tune! (proposal))
(defgeneric proposal/accepted! (proposal acceptedp))
(defgeneric proposal/propose (proposal value))

(defgeneric r/proposal (rv))
(defgeneric r/propose! (rv proposal))
(defgeneric r/accept! (rv proposal acceptedp))

(defclass mcmc/proposal ()
  ((accepted :initform 0)
   (rejected :initform 0)
   (factor :initform 1D0)
   (pvalue :initform nil)))

(defmethod proposal/tune! ((proposal mcmc/proposal))
  (with-slots (accepted rejected factor) proposal
    (let ((total (+ accepted rejected)))
      (when (> total 0)
        (let ((r (/ accepted total)))
          (cond ((< r 0.001) (setf factor (* factor 0.1)))
                ((< r 0.05) (setf factor (* factor 0.5)))
                ((< r 0.2) (setf factor (* factor 0.9)))
                ((> r 0.95) (setf factor (* factor 10.0)))
                ((> r 0.75) (setf factor (* factor 2.0)))
                ((> r 0.5) (setf factor (* factor 1.1))))
          (setf accepted 0
                rejected 0))))))

(defmethod proposal/accepted! ((proposal mcmc/proposal) acceptedp)
  (with-slots (accepted rejected) proposal
    (if acceptedp
        (incf accepted)
        (incf rejected))
    proposal))

(defmethod r/propose! ((rv r/var) (proposal mcmc/proposal))
  (let ((ratio 0D0))
    (with-slots (pvalue) proposal
      (with-slots (value) rv
        (setf pvalue value)
        (let ((proposed (proposal/propose proposal value)))
          (setf value (car proposed))
          (setf ratio (cdr proposed)))))
    ratio))

(defmethod r/accept! ((rv r/var) (proposal mcmc/proposal) acceptedp)
  (proposal/accepted! proposal acceptedp)
  (unless acceptedp
    (with-slots (pvalue) proposal
      (when pvalue
        (with-slots (value) rv
          (setf value pvalue
                pvalue nil)))
      rv))
  rv)

(defclass proposal/gaussian (mcmc/proposal)
  ((scale :initform 1D0)))

(defun proposal/gaussian (&optional (scale 1D0))
  (let ((n (make-instance 'proposal/gaussian))
        (s scale))
    (with-slots (scale) n
      (setf scale s))
    n))

(defmethod proposal/propose ((proposal proposal/gaussian) value)
  (with-slots (scale factor) proposal
    (cons (sample/gaussian value (* factor scale)) 0D0)))

(defclass proposal/discrete-gaussian (mcmc/proposal)
  ((scale :initform 1D0)))

(defun proposal/discrete-gaussian (&optional (scale 1D0))
  (let ((n (make-instance 'proposal/discrete-gaussian))
        (s scale))
    (with-slots (scale) n
      (setf scale s))
    n))

(defmethod proposal/propose ((proposal proposal/gaussian) value)
  (with-slots (scale factor) proposal
    (cons (round (sample/gaussian value (* factor scale))) 0D0)))

(defmethod r/proposal ((rv r/discrete)) (proposal/discrete-gaussian))
(defmethod r/proposal ((rv r/continuous)) (proposal/gaussian))

(defun mh/accepted (prob nprob log-hastings-ratio)
  (when (and prob nprob log-hastings-ratio)
    (let ((alpha (+ (- nprob prob) log-hastings-ratio)))
      (> alpha (log (random 1D0))))))

(defun mcmc/mh (parameters posterior-function
                &key (iterations 50000) (tune-steps 100) (burn-in 1000) (thin 1))
  (labels ((posterior (vs) (funcall posterior-function vs))
           (vals (parameters) (mapcar #'r/value parameters)))
    (let ((prob (posterior (vals parameters)))
          (np ($count parameters)))
      (when prob
        (let ((proposals (mapcar #'r/proposal parameters))
              (traces (mcmc/traces np :burn-in burn-in :thin thin))
              (candidates (mapcar #'$clone parameters)))
          (loop :repeat (+ iterations burn-in)
                :for iter :from 1
                :for burning = (<= iter burn-in)
                :for tuneable = (zerop (rem iter tune-steps))
                :do (let ((tune (and (> iter 1) burning tuneable)))
                      (when tune
                        (loop :for proposal :in proposals :do (proposal/tune! proposal)))
                      (loop :for proposal :in proposals
                            :for candidate :in candidates
                            :for trace :in traces
                            :for lhr = (r/propose! candidate proposal)
                            :for nprob = (posterior (vals candidates))
                            :do (let ((accepted (mh/accepted prob nprob lhr)))
                                  (r/accept! candidate proposal accepted)
                                  (trace/push! (r/value candidate) trace)
                                  (when accepted (setf prob nprob))))))
          traces)))))
