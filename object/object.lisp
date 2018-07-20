(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defclass th.object () ((handle :initform nil :accessor $handle)))
(defmethod $handle (null) +nil+)
