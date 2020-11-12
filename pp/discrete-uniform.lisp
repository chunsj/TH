(in-package :th.pp)

(defgeneric ll/discrete-uniform (data lower upper))
(defgeneric sample/discrete-uniform (lower upper &optional n))

(defun of-uniform-p (data lower upper) (and (> upper lower) (of-ie data lower upper)))

(defmethod ll/discrete-uniform ((data number) (lower number) (upper number))
  (when (of-uniform-p data lower upper)
    (- (log (- upper lower)))))

(defmethod ll/discrete-uniform ((data number) (lower node) (upper number))
  (when (of-uniform-p data ($data lower) upper)
    ($neg ($log ($sub upper lower)))))

(defmethod ll/discrete-uniform ((data number) (lower number) (upper node))
  (when (of-uniform-p data lower ($data upper))
    ($neg ($log ($sub upper lower)))))

(defmethod ll/discrete-uniform ((data number) (lower node) (upper node))
  (when (of-uniform-p data ($data lower) ($data upper))
    ($neg ($log ($sub upper lower)))))

(defmethod ll/discrete-uniform ((data tensor) (lower number) (upper number))
  (when (of-uniform-p data lower upper)
    (let ((n ($count data)))
      ($mul (- n) ($log ($sub upper lower))))))

(defmethod ll/discrete-uniform ((data tensor) (lower node) (upper number))
  (when (of-uniform-p data ($data lower) upper)
    (let ((n ($count data)))
      ($mul (- n) ($log ($sub upper lower))))))

(defmethod ll/discrete-uniform ((data tensor) (lower number) (upper node))
  (when (of-uniform-p data lower ($data upper))
    (let ((n ($count data)))
      ($mul (- n) ($log ($sub upper lower))))))

(defmethod ll/discrete-uniform ((data tensor) (lower node) (upper node))
  (when (of-uniform-p data ($data lower) ($data upper))
    (let ((n ($count data)))
      ($mul (- n) ($log ($sub upper lower))))))

(defmethod sample/discrete-uniform ((lower number) (upper number) &optional (n 1))
  (cond ((= n 1) (random/discrete-uniform lower upper))
        ((> n 1) ($discrete-uniform! (tensor n) lower upper))))

(defmethod sample/discrete-uniform ((lower node) (upper number) &optional (n 1))
  (cond ((= n 1) (random/discrete-uniform ($data lower) upper))
        ((> n 1) ($discrete-uniform! (tensor n) ($data lower) upper))))

(defmethod sample/discrete-uniform ((lower number) (upper node) &optional (n 1))
  (cond ((= n 1) (random/discrete-uniform lower ($data upper)))
        ((> n 1) ($discrete-uniform! (tensor n) lower ($data upper)))))

(defmethod sample/discrete-uniform ((lower node) (upper node) &optional (n 1))
  (cond ((= n 1) (random/discrete-uniform ($data lower) ($data upper)))
        ((> n 1) ($discrete-uniform! (tensor n) ($data lower) ($data upper)))))

(defclass r/discrete-uniform (r/discrete)
  ((lower :initform 1)
   (upper :initform 6)))

(defun r/discrete-uniform (&key (lower 1) (upper 6) observation)
  (let ((l lower)
        (u upper)
        (rv (make-instance 'r/discrete-uniform)))
    (with-slots (lower upper) rv
      (setf lower l
            upper u))
    (r/set-observation! rv observation)
    (r/set-sample! rv)
    rv))

(defmethod r/sample ((rv r/discrete-uniform))
  (with-slots (lower upper) rv
    (sample/discrete-uniform lower upper)))

(defmethod $logp ((rv r/discrete-uniform))
  (with-slots (lower upper) rv
    (ll/discrete-uniform (r/value rv) lower upper)))
