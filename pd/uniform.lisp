(in-package :th)

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

(defmethod score/uniform ((data node) (lower number) (upper number))
  (when (of-uniform-p ($data data) lower upper)
    (let ((n (if ($tensorp data) ($count data) 1)))
      ($mul (- n) ($log ($sub upper lower))))))

(defmethod score/uniform ((data node) (lower node) (upper number))
  (when (of-uniform-p data ($data lower) upper)
    (let ((n (if ($tensorp data) ($count data) 1)))
      ($mul (- n) ($log ($sub upper lower))))))

(defmethod score/uniform ((data node) (lower number) (upper node))
  (when (of-uniform-p ($data data) lower ($data upper))
    (let ((n (if ($tensorp data) ($count data) 1)))
      ($mul (- n) ($log ($sub upper lower))))))

(defmethod score/uniform ((data node) (lower node) (upper node))
  (when (of-uniform-p ($data data) ($data lower) ($data upper))
    (let ((n (if ($tensorp data) ($count data) 1)))
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
