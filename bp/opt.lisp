(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defgeneric $gd! (parameter &optional α) (:documentation "Executes gradient descent."))
(defgeneric $mgd! (parameter &optional α momentum) (:documentation "Executes momentum."))
(defgeneric $agd! (parameter &optional α) (:documentation "Executes adagrad."))
(defgeneric $amgd! (parameter &optional α β1 β2) (:documentation "Executes adam."))
(defgeneric $rmgd! (parameter &optional α λ) (:documentation "Executes rmsprop."))
(defgeneric $adgd! (parameter &optional λ) (:documentation "Executes adadelta."))

(defmethod $gd! ((object t) &optional (learning-rate 0.01)) (declare (ignore learning-rate)))

(defmethod $gd! ((parameter parameter) &optional (α 0.01))
  (let ((data ($data parameter))
        (grv ($gradient parameter)))
    (cond ((null grv) nil)
          ((numberp grv) (setf ($data parameter) (- data (* grv α))))
          (t ($axpy! (- α) grv data)))
    (setf ($gradient parameter) nil)))
