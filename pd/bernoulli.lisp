(in-package :th)

(defgeneric score/bernoulli (data p))
(defgeneric sample/bernoulli (p &optional n))

(defun of-bernoulli-p (data p)
  (if ($tensorp data)
      (and (= ($count data) ($+ ($sum ($eq data 0)) ($sum ($eq data 1))))
           (of-it p 0 1))
      (and (or (= data 0) (= data 1)) (of-it p 0 1))))

(defmethod score/bernoulli ((data number) (p number))
  (when (of-bernoulli-p data p)
    (+ (* data (log p)) (* (- 1 data) (log (- 1 p))))))

(defmethod score/bernoulli ((data number) (p node))
  (when (of-bernoulli-p data ($data p))
    ($add ($mul data ($log p)) ($mul (- 1 data) ($log ($sub 1 p))))))

(defmethod score/bernoulli ((data tensor) (p number))
  (when (of-bernoulli-p data p)
    ($sum ($add ($mul data (log p)) ($mul ($sub 1 data) (log (- 1 p)))))))

(defmethod score/bernoulli ((data tensor) (p node))
  (when (of-bernoulli-p data ($data p))
    ($sum ($add ($mul data ($log p)) ($mul ($sub 1 data) ($log ($sub 1 p)))))))

(defmethod score/bernoulli ((data node) (p number))
  (when (of-bernoulli-p ($data data) p)
    ($sum ($add ($mul data (log p)) ($mul ($sub 1 data) (log (- 1 p)))))))

(defmethod score/bernoulli ((data node) (p node))
  (when (of-bernoulli-p ($data data) ($data p))
    ($sum ($add ($mul data ($log p)) ($mul ($sub 1 data) ($log ($sub 1 p)))))))

(defmethod sample/bernoulli ((p number) &optional (n 1))
  (cond ((= n 1) (random/bernoulli p))
        ((> n 1) ($bernoulli! (tensor n) p))))

(defmethod sample/bernoulli ((p node) &optional (n 1))
  (cond ((= n 1) (random/bernoulli ($data p)))
        ((> n 1) ($bernoulli! (tensor n) ($data p)))))
