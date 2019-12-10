(in-package :th)

(defparameter *big-tensor* nil)
(gcf)
(report-foreign-memory-allocation)

(setf *big-tensor* ($zero! (tensor 10000 30000)))
(gcf)
(report-foreign-memory-allocation)

(setf *big-tensor* nil)
(gcf)
(report-foreign-memory-allocation)

(setf *big-tensor* (tensor 100 100))
(gcf)
(report-foreign-memory-allocation)
