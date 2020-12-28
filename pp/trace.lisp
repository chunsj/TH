(in-package :th.pp)

(defgeneric $values (trace))

(defclass r/trace (r/variable)
  ((collection :initform nil)
   (burn-ins :initform 0)
   (thin :initform 0)
   (vals :initform nil)))

(defmethod $count ((trace r/trace))
  (with-slots (collection) trace
    ($count collection)))

(defmethod $ ((trace r/trace) index &rest others-and-default)
  (declare (ignore others-and-default))
  (with-slots (collection) trace
    ($ collection index)))

(defmethod (setf $) (value (trace r/trace) index &rest others)
  (declare (ignore others))
  (with-slots (collection) trace
    (setf ($ collection index) value)))

(defmethod $values ((trace r/trace))
  (with-slots (collection burn-ins thin vals) trace
    (unless vals
      (let ((vs (loop :for i :from burn-ins :below ($count collection) :by thin
                      :collect ($ collection i))))
        (setf vals (tensor vs))))
    vals))
