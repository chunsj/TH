(in-package :th.pp)

(defgeneric ll/poisson (data rate))
(defgeneric sample/poisson (rate &optional n))

(defun of-poisson-p (data rate) (and (of-plusp rate) (of-ge data 0)))

(defmethod ll/poisson ((data number) (rate number))
  (when (of-poisson-p data rate)
    (let ((lfac ($lgammaf (+ 1 data))))
      (- (* data (log rate)) (+ rate lfac)))))

(defmethod ll/poisson ((data number) (rate node))
  (when (of-poisson-p data ($data rate))
    (let ((lfac ($lgammaf (+ 1 data))))
      ($sub ($mul data ($log rate)) ($add rate lfac)))))

(defmethod ll/poisson ((data tensor) (rate number))
  (when (of-poisson-p data rate)
    (let ((lfac ($lgammaf ($add 1 data))))
      ($sum ($sub ($mul data (log rate)) ($add rate lfac))))))

(defmethod ll/poisson ((data tensor) (rate node))
  (when (of-poisson-p data ($data rate))
    (let ((lfac ($lgammaf ($add 1 data))))
      ($sum ($sub ($mul data ($log rate)) ($add rate lfac))))))

(defmethod sample/poisson ((rate number) &optional (n 1))
  (cond ((= n 1) (random/poisson rate))
        ((> n 1) ($poisson! (tensor n) rate))))

(defmethod sample/poisson ((rate node) &optional (n 1))
  (cond ((= n 1) (random/poisson ($data rate)))
        ((> n 1) ($poisson! (tensor n) ($data rate)))))

(defclass r/poisson (r/discrete)
  ((rate :initform 1)))

(defun r/poisson (&key (rate 1) observation)
  (let ((r rate)
        (rv (make-instance 'r/poisson)))
    (with-slots (rate) rv
      (setf rate r))
    (r/set-observation! rv observation)
    (r/set-sample! rv)
    rv))

(defmethod r/sample ((rv r/poisson))
  (with-slots (rate) rv
    (sample/poisson rate)))

(defmethod r/logp ((rv r/poisson))
  (with-slots (rate) rv
    (ll/poisson (r/value rv) rate)))
