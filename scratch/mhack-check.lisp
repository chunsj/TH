(in-package :th)

(th-get-current-heap-size)

(defparameter *x* (rndn 100 100))
(setf *x* nil)

(let ((x (tensor 1000 10000))
      (y (tensor 1000 10000)))
  (prn x)
  (prn y))
(gcf)

(loop :repeat 100 :do (tensor 1000 10000))
