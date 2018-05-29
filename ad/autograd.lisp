(in-package :th)

(defgeneric $bp! (node &optional gradient) (:documentation "Executes backward error propagation."))
(defgeneric $gd! (node &optional learning-rate) (:documentation "Executes gradient descent."))

(defgeneric $variable (object) (:documentation "Returns variable node."))
(defgeneric $constant (object) (:documentation "Returns constant node."))

(defgeneric $broadcast (constant matrix))

(defclass node ()
  ((data :initform nil :accessor $data)
   (gradient :initform nil :accessor $gradient)
   (need-gradient-p :initform nil :accessor $gradientp)
   (children :initform nil :accessor $children)
   (backward-function :initform nil :accessor $bpfn)))

(defmethod print-object ((node node) stream)
  (format stream "[NODE] ")
  (print-object ($data node) stream))

(defun $c0 (node) ($0 ($children node)))
(defun $c1 (node) ($1 ($children node)))

(defmethod $ ((node node) location &rest others-and-default)
  (apply #'$ ($children ($data node)) (cons location others-and-default)))

(defmethod (setf $) (value (node node) location &rest others)
  (setf (apply #'$ ($data node) (cons location others)) value))

(defmethod $tensorp ((node node)) ($tensorp ($data node)))

(defun default-bpfn (node gradient)
  (setf ($gradient node) gradient)
  node)

(defun node (data &optional need-gradient-p)
  (let ((n (make-instance 'node)))
    (setf ($data n) data)
    (setf ($gradientp n) need-gradient-p)
    (setf ($bpfn n) #'default-bpfn)
    n))

(defmethod $variable ((node node)) (setf ($gradientp node) t) node)
(defmethod $constant ((node node)) (setf ($gradientp node) nil) node)

(defmethod $variable ((data list)) (node (tensor data) t))
(defmethod $constant ((data list)) (node (tensor data) nil))

(defmethod $variable ((data t)) (node data t))
(defmethod $constant ((data t)) (node data nil))

(defmethod $bp! ((node node) &optional (gradient 1)) (funcall ($bpfn node) node gradient))

(defmethod $gd! ((gradient node) &optional (learning-rate 0.01))
  (let ((children ($children gradient))
        (data ($data gradient))
        (grv ($gradient gradient)))
    (cond ((null grv) nil)
          ((numberp grv) (setf ($data gradient) (- data (* grv learning-rate))))
          (t ($axpy! (- learning-rate) grv ($data gradient))))
    (loop :for c :in children :do ($gd! c learning-rate))
    gradient))

(defmethod $gd! ((object t) &optional (learning-rate 0.01)) (declare (ignore learning-rate)))

(defmethod $zero ((x node)) (node ($zero ($data x)) ($gradientp x)))
(defmethod $one ((x node)) (node ($one ($data x)) ($gradientp x)))
(defmethod $fill ((x node) value) (node ($fill ($data x) value) ($gradientp x)))
(defmethod $ndim ((x node)) ($ndim ($data x)))
(defmethod $count ((x node)) ($count ($data x)))

(defmethod $ ((x node) location &rest others-and-default)
  (apply #'$ ($data x) location others-and-default))

(defmethod $zero! ((x node))
  ($zero! ($data x))
  x)
(defmethod $one! ((x node))
  ($one! ($data x))
  x)
(defmethod $fill! ((x node) value)
  ($fill! ($data x) value)
  x)

(defmethod $empty ((node node))
  (let ((data ($data node)))
    (cond (($gradientp node) ($variable ($empty data)))
          (t ($constant ($empty data))))))

(defmethod $storage ((node node)) ($storage ($data node)))
(defmethod $offset ((node node)) ($offset ($data node)))
(defmethod $size ((node node) &optional dimension) ($size ($data node) dimension))
(defmethod $stride ((node node) &optional dimension) ($stride ($data node) dimension))
