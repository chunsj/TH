(in-package :th.pp)

(defgeneric score/exponential (data rate))
(defgeneric sample/exponential (rate &optional n))

(defun of-exponential-p (data rate) (and (of-plusp data) (of-plusp rate)))

(defmethod score/exponential ((data number) (rate number))
  (when (of-exponential-p data rate)
    (- (log rate) (* rate data))))

(defmethod score/exponential ((data number) (rate node))
  (when (of-exponential-p data ($data rate))
    ($sub ($log rate) ($mul rate data))))

(defmethod score/exponential ((data tensor) (rate number))
  (when (of-exponential-p data rate)
    ($sum ($sub ($log rate) ($mul rate data)))))

(defmethod score/exponential ((data tensor) (rate node))
  (when (of-exponential-p data ($data rate))
    ($sum ($sub ($log rate) ($mul rate data)))))

(defmethod sample/exponential ((rate number) &optional (n 1))
  (cond ((= n 1) (random/exponential rate))
        ((> n 1) ($exponential! (tensor n) rate))))

(defmethod sample/exponential ((rate node) &optional (n 1))
  (cond ((= n 1) (random/exponential ($data rate)))
        ((> n 1) ($exponential! (tensor n) ($data rate)))))

(defclass r/exponential (r/continuous)
  ((rate :initform 1)))

(defun r/exponential (&key (rate 1) observation)
  (let ((r rate)
        (rv (make-instance 'r/exponential)))
    (with-slots (rate) rv
      (setf rate r))
    (r/set-observation! rv observation)
    (r/set-sample! rv)
    rv))

(defmethod r/sample ((rv r/exponential))
  (with-slots (rate) rv
    (sample/exponential rate)))

(defmethod r/score ((rv r/exponential))
  (with-slots (rate) rv
    (score/exponential (r/value rv) rate)))

(defmethod $clone ((rv r/exponential))
  (let ((nrv (call-next-method)))
    (with-slots (rate) rv
      (let ((r ($clone rate)))
        (with-slots (rate) nrv
          (setf rate r))))
    nrv))
