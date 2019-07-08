(in-package :th)

(defun report-heap ()
  (prn "")
  (prn "SIZE:" (th-get-heap-size))
  (prn "DELT:" (th-get-heap-delta))
  (prn "SMAX:" (th-get-heap-softmax))
  (prn ""))

(defparameter *big-tensor* nil)
(gcf)
(report-heap)

(setf *big-tensor* ($zero! (tensor 10000 30000)))
(gcf)
(report-heap)

(setf *big-tensor* nil)
(gcf)
(report-heap)

(setf *big-tensor* (tensor 100 100))
(gcf)
(report-heap)
