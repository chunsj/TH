(in-package :th.pp)

(defgeneric proposal/tune! (proposal))
(defgeneric proposal/accepted! (proposal acceptedp))
(defgeneric proposal/propose (proposal value))

(defgeneric r/proposal (rv))
(defgeneric r/propose! (rv proposal))
(defgeneric r/revert! (rv proposal))

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

(defmethod r/revert! ((rv r/var) (proposal mcmc/proposal))
  (with-slots (pvalue) proposal
    (when pvalue
      (with-slots (value) rv
        (setf value pvalue))
      (setf pvalue nil))
    rv))

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

(defun mcmc/mh (parameters likelihoodfn
                &key (iterations 50000) (tune-steps 100) (burn-in 1000) (thin 1))
  (let ((lk (funcall likelihoodfn parameters)))
    (when lk
      (labels ((likelihood (params) (funcall likelihoodfn params)))
        (let ((proposals (mapcar #'r/proposal parameters))
              (parameters (mapcar #'$clone parameters))
              (traces (mcmc/traces ($count parameters) :burn-in burn-in :thin thin)))
          (loop :repeat (+ iterations burn-in)
                :for iter :from 1
                :for burning = (<= iter burn-in)
                :for tuneable = (zerop (rem iter tune-steps))
                :do (let ((tune (and (> iter 1) burning tuneable)))
                      (when tune
                        (loop :for proposal :in proposals :do (proposal/tune! proposal)))
                      (loop :for proposal :in proposals
                            :for parameter :in parameters
                            :for trace :in traces
                            :for lhr = (r/propose! parameter proposal)
                            :for nlk = (likelihood parameters)
                            :for u = (log (random 1D0))
                            :do (let ((accepted (and lhr nlk (> (+ (- nlk lk) lhr) u))))
                                  (proposal/accepted! proposal accepted)
                                  (if accepted
                                      (setf lk nlk)
                                      (r/revert! parameter proposal))
                                  (trace/push! parameter trace)))))
          traces)))))
