(in-package :th.pp)

(defgeneric ll/categorical (data probs))
(defgeneric sample/categorical (probs &optional n))

(defun of-categorical-p (data probs)
  (let ((n ($count probs)))
    (and (of-ie data 0 (- n 1)) (of-ie probs 0 1))))

(defmethod ll/categorical ((data number) (probs tensor))
  (when (of-categorical-p data probs)
    ($sum ($gather ($log probs) 0 (tensor.long (list data))))))

(defmethod ll/categorical ((data number) (probs node))
  (when (of-categorical-p data ($data probs))
    ($sum ($gather ($log probs) 0 (tensor.long (list data))))))

(defmethod ll/categorical ((data tensor) (probs tensor))
  (when (of-categorical-p data probs)
    ($sum ($gather ($log probs) 0 (tensor.long data)))))

(defmethod ll/categorical ((data tensor) (probs node))
  (when (of-categorical-p data ($data probs))
    ($sum ($gather ($log probs) 0 (tensor.long data)))))

(defmethod sample/categorical ((probs tensor) &optional (n 1))
  (let ((s ($multinomial probs n)))
    (cond ((= n 1) ($0 s))
          ((> n 1) s))))

(defmethod sample/categorical ((probs node) &optional (n 1))
  (let ((s ($multinomial ($data probs) n)))
    (cond ((= n 1) ($0 s))
          ((> n 1) s))))
