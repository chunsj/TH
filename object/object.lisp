(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defclass th.object () ((handle :initform nil :accessor $handle)))
(defmethod $handle (null) +nil+)

#+ccl
(defmethod initialize-instance :after ((x th.object) &rest initargs)
  (ccl:terminate-when-unreachable x))

#+ccl
(defun has-valid-handle-p (o) (and ($handle o) (not (cffi:null-pointer-p ($handle o)))))

#+ccl
(defun reset-handle (o) (setf ($handle o) nil))
