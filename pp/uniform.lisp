(in-package :th.pp)

(defgeneric score/uniform (data lower upper))
(defgeneric sample/uniform (lower upper &optional n))

(defun of-uniform-p (data lower upper) (and (> upper lower) (of-ie data lower upper)))

(defmethod score/uniform ((data number) (lower number) (upper number))
  (when (of-uniform-p data lower upper)
    (- (log (- upper lower)))))

(defmethod score/uniform ((data number) (lower node) (upper number))
  (when (of-uniform-p data ($data lower) upper)
    ($neg ($log ($sub upper lower)))))

(defmethod score/uniform ((data number) (lower number) (upper node))
  (when (of-uniform-p data lower ($data upper))
    ($neg ($log ($sub upper lower)))))

(defmethod score/uniform ((data number) (lower node) (upper node))
  (when (of-uniform-p data ($data lower) ($data upper))
    ($neg ($log ($sub upper lower)))))

(defmethod score/uniform ((data tensor) (lower number) (upper number))
  (when (of-uniform-p data lower upper)
    (let ((n ($count data)))
      ($mul (- n) ($log ($sub upper lower))))))

(defmethod score/uniform ((data tensor) (lower node) (upper number))
  (when (of-uniform-p data ($data lower) upper)
    (let ((n ($count data)))
      ($mul (- n) ($log ($sub upper lower))))))

(defmethod score/uniform ((data tensor) (lower number) (upper node))
  (when (of-uniform-p data lower ($data upper))
    (let ((n ($count data)))
      ($mul (- n) ($log ($sub upper lower))))))

(defmethod score/uniform ((data tensor) (lower node) (upper node))
  (when (of-uniform-p data ($data lower) ($data upper))
    (let ((n ($count data)))
      ($mul (- n) ($log ($sub upper lower))))))

(defmethod sample/uniform ((lower number) (upper number) &optional (n 1))
  (cond ((= n 1) (random/uniform lower upper))
        ((> n 1) ($uniform! (tensor n) lower upper))))

(defmethod sample/uniform ((lower node) (upper number) &optional (n 1))
  (cond ((= n 1) (random/uniform ($data lower) upper))
        ((> n 1) ($uniform! (tensor n) ($data lower) upper))))

(defmethod sample/uniform ((lower number) (upper node) &optional (n 1))
  (cond ((= n 1) (random/uniform lower ($data upper)))
        ((> n 1) ($uniform! (tensor n) lower ($data upper)))))

(defmethod sample/uniform ((lower node) (upper node) &optional (n 1))
  (cond ((= n 1) (random/uniform ($data lower) ($data upper)))
        ((> n 1) ($uniform! (tensor n) ($data lower) ($data upper)))))

(defclass r/uniform (r/discrete)
  ((lower :initform 0)
   (upper :initform 1)))

(defun r/uniform (&key (lower 0) (upper 1) observation)
  (let ((l lower)
        (u upper)
        (rv (make-instance 'r/uniform)))
    (with-slots (lower upper) rv
      (setf lower l
            upper u))
    (r/set-observation! rv observation)
    (r/set-sample! rv)
    rv))

(defmethod r/sample ((rv r/uniform))
  (with-slots (lower upper) rv
    (sample/uniform lower upper)))

(defmethod r/score ((rv r/uniform))
  (with-slots (lower upper) rv
    (score/uniform (r/value rv) lower upper)))

(defmethod $clone ((rv r/uniform))
  (let ((nrv (call-next-method)))
    (with-slots (lower upper) rv
      (let ((l ($clone lower))
            (u ($clone upper)))
        (with-slots (lower upper) nrv
          (setf lower l
                upper u))))
    nrv))
