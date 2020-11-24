(in-package :th.pp)

(defgeneric r/continuousp (rv))
(defgeneric r/discretep (rv))

(defclass r/variable ()
  ((value :initform nil :accessor $data)))

(defmethod print-object ((rv r/variable) stream)
  (format stream "~A" ($data rv)))

(defmethod $clone ((rv r/variable))
  (let ((n (make-instance (class-of rv))))
    (with-slots (value) rv
      (let ((v value))
        (with-slots (value) n
          (setf value ($clone v)))))
    n))

(defmethod r/continuousp ((rv r/variable)) nil)
(defmethod r/discretep ((rv r/variable)) nil)

(defclass r/continuous (r/variable)
  ())

(defmethod r/continuousp ((rv r/continuous)) T)
(defmethod r/discretep ((rv r/continuous)) nil)

(defclass r/discrete (r/variable)
  ())

(defmethod r/continuousp ((rv r/discrete)) nil)
(defmethod r/discretep ((rv r/discrete)) T)

(defun r/variable (value &optional (type :continuous))
  (let ((rv (cond ((eq type :continuous) (make-instance 'r/continuous))
                  ((eq type :discrete) (make-instance 'r/discrete)))))
    (setf ($data rv) value)
    rv))
