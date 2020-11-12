(in-package :th.pp)

(defgeneric ll/bernoulli (data p))
(defgeneric sample/bernoulli (p &optional n))

(defun of-bernoulli-p (data p)
  (if ($tensorp data)
      (and (= ($count data) (+ ($sum ($eq data 0) ($eq data 1))))
           (of-it p 0 1))
      (and (or (= data 0) (= data 1)) (of-it p 0 1))))

(defmethod ll/bernoulli ((data number) (p number))
  (when (of-bernoulli-p data p)
    (+ (* data (log p)) (* (- 1 data) (log (- 1 p))))))

(defmethod ll/bernoulli ((data number) (p node))
  (when (of-bernoulli-p data ($data p))
    ($add ($mul data ($log p)) ($mul (- 1 data) ($log ($sub 1 p))))))

(defmethod ll/bernoulli ((data tensor) (p number))
  (when (of-bernoulli-p data p)
    ($sum ($add ($mul data (log p)) ($mul ($sub 1 data) (log (- 1 p)))))))

(defmethod ll/bernoulli ((data tensor) (p node))
  (when (of-bernoulli-p data ($data p))
    ($sum ($add ($mul data ($log p)) ($mul ($sub 1 data) ($log ($sub 1 p)))))))

(defmethod sample/bernoulli ((p number) &optional (n 1))
  (cond ((= n 1) (random/bernoulli p))
        ((> n 1) ($bernoulli! (tensor n) p))))

(defmethod sample/bernoulli ((p node) &optional (n 1))
  (cond ((= n 1) (random/bernoulli ($data p)))
        ((> n 1) ($bernoulli! (tensor n) ($data p)))))

(defclass r/bernoulli (r/discrete)
  ((p :initform 0.5)))

(defun r/bernoulli (&key (p 0.5) observation)
  (let ((prob p)
        (rv (make-instance 'r/bernoulli)))
    (with-slots (p) rv
      (setf p prob))
    (r/set-observation! rv observation)
    (r/set-sample! rv)
    rv))

(defmethod r/sample ((rv r/bernoulli))
  (with-slots (p) rv
    (sample/bernoulli p)))

(defmethod r/logp ((rv r/bernoulli))
  (with-slots (p) rv
    (ll/bernoulli (r/value rv) p)))
