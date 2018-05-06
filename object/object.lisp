(in-package :th)

(defclass th.object () ((handle :initform nil :accessor $handle)))
(defmethod $handle (null) +nil+)
