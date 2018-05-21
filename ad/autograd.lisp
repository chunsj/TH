(in-package :th)

(defgeneric $tapep (object))
(defgeneric $bp! (tape &optional gradient))
(defgeneric $gd! (gradient &optional learning-rate))

(defgeneric $variant (object))
(defgeneric $constant (object))

(defgeneric $broadcast (constant matrix))

(defclass tape ()
  ((data :initform nil :accessor $data)
   (gradient :initform nil :accessor $gradient)
   (need-gradient-p :initform nil :accessor $gradientp)
   (children :initform nil :accessor $children)
   (backward-function :initform nil :accessor $bpfn)))

(defmethod print-object ((tape tape) stream) (print-object ($data tape) stream))

(defun $c0 (tape) ($0 ($children tape)))
(defun $c1 (tape) ($1 ($children tape)))

(defmethod $ ((tape tape) location &rest others-and-default)
  (apply #'$ ($children ($data tape)) (cons location others-and-default)))

(defmethod (setf $) (value (tape tape) location &rest others)
  (setf (apply #'$ ($data tape) (cons location others)) value))

(defmethod $tapep ((tape tape)) t)
(defmethod $tapep ((object t)) nil)

(defun default-bpfn (tape gradient)
  (setf ($gradient tape) gradient)
  tape)

(defun tape (data &optional need-gradient-p)
  (let ((n (make-instance 'tape)))
    (setf ($data n) data)
    (setf ($gradientp n) need-gradient-p)
    (setf ($bpfn n) #'default-bpfn)
    n))

(defmethod $variant ((tape tape)) (setf ($gradientp tape) t) tape)
(defmethod $constant ((tape tape)) (setf ($gradientp tape) nil) tape)

(defmethod $variant ((data list)) (tape (tensor data) t))
(defmethod $constant ((data list)) (tape (tensor data) nil))

(defmethod $variant ((data t)) (tape data t))
(defmethod $constant ((data t)) (tape data nil))

(defmethod $bp! ((tape tape) &optional (gradient 1)) (funcall ($bpfn tape) tape gradient))

(defmethod $gd! ((gradient tape) &optional (learning-rate 0.01))
  (let ((children ($children gradient))
        (data ($data gradient))
        (grv ($gradient gradient)))
    (cond ((null grv) nil)
          ((numberp grv) (setf ($data gradient) (- data (* grv learning-rate))))
          (t ($axpy! (- learning-rate) grv ($data gradient))))
    (loop :for c :in children :do ($gd! c learning-rate))
    gradient))

(defmethod $gd! ((object t) &optional (learning-rate 0.01)) (declare (ignore learning-rate)))

(defun variant (data) ($variant data))
(defun constant (data) ($constant data))
