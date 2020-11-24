(in-package :th.pp)

(defgeneric score/discrete-uniform (data lower upper))
(defgeneric sample/discrete-uniform (lower upper &optional n))

(defun of-uniform-p (data lower upper) (and (> upper lower) (of-ie data lower upper)))

(defmethod score/discrete-uniform ((data number) (lower number) (upper number))
  (when (of-uniform-p data lower upper)
    (- (log (+ 1 (- upper lower))))))

(defmethod score/discrete-uniform ((data number) (lower node) (upper number))
  (when (of-uniform-p data ($data lower) upper)
    ($neg ($log ($add 1 ($sub upper lower))))))

(defmethod score/discrete-uniform ((data number) (lower number) (upper node))
  (when (of-uniform-p data lower ($data upper))
    ($neg ($log ($add 1($sub upper lower))))))

(defmethod score/discrete-uniform ((data number) (lower node) (upper node))
  (when (of-uniform-p data ($data lower) ($data upper))
    ($neg ($log ($add 1 ($sub upper lower))))))

(defmethod score/discrete-uniform ((data tensor) (lower number) (upper number))
  (when (of-uniform-p data lower upper)
    (let ((n ($count data)))
      ($mul (- n) ($log ($add 1 ($sub upper lower)))))))

(defmethod score/discrete-uniform ((data tensor) (lower node) (upper number))
  (when (of-uniform-p data ($data lower) upper)
    (let ((n ($count data)))
      ($mul (- n) ($log ($add 1 ($sub upper lower)))))))

(defmethod score/discrete-uniform ((data tensor) (lower number) (upper node))
  (when (of-uniform-p data lower ($data upper))
    (let ((n ($count data)))
      ($mul (- n) ($log ($add 1 ($sub upper lower)))))))

(defmethod score/discrete-uniform ((data tensor) (lower node) (upper node))
  (when (of-uniform-p data ($data lower) ($data upper))
    (let ((n ($count data)))
      ($mul (- n) ($log ($add 1 ($sub upper lower)))))))

(defmethod score/discrete-uniform ((data node) (lower number) (upper number))
  (when (of-uniform-p ($data data) lower upper)
    (let ((n (if ($tensorp data) ($count data) 1)))
      ($mul (- n) ($log ($add 1 ($sub upper lower)))))))

(defmethod score/discrete-uniform ((data node) (lower node) (upper number))
  (when (of-uniform-p ($data data) ($data lower) upper)
    (let ((n (if ($tensorp data) ($count data) 1)))
      ($mul (- n) ($log ($add 1 ($sub upper lower)))))))

(defmethod score/discrete-uniform ((data node) (lower number) (upper node))
  (when (of-uniform-p ($data data) lower ($data upper))
    (let ((n (if ($tensorp data) ($count data) 1)))
      ($mul (- n) ($log ($add 1 ($sub upper lower)))))))

(defmethod score/discrete-uniform ((data node) (lower node) (upper node))
  (when (of-uniform-p ($data data) ($data lower) ($data upper))
    (let ((n (if ($tensorp data) ($count data) 1)))
      ($mul (- n) ($log ($add 1 ($sub upper lower)))))))

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
