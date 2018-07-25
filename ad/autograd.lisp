(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defgeneric $variable (object) (:documentation "Returns variable node."))
(defgeneric $constant (object) (:documentation "Returns constant node."))

(defgeneric $broadcast (constant matrix))

(defclass node ()
  ((nm :initform nil :accessor $name)
   (data :initform nil :accessor $data)
   (fns :initform nil :accessor $fns)
   (gradientv :initform nil :accessor $gradientv)
   (need-gradient-p :initform nil :accessor $gradientp)
   (attrs :initform #{} :accessor $attrs)))

(defmethod print-object ((node node) stream)
  (format stream "[~A] " (if (null ($name node))
                             (cond (($gradientp node) "VARIABLE")
                                   (t "CONSTANT"))
                             ($name node)))
  (format stream "~A" ($data node)))

(defmethod $tensorp ((node node)) ($tensorp ($data node)))

(defgeneric $gradient (node))

(defmethod $gradient ((node node))
  (if ($gradientp node)
      (progn
        (unless ($gradientv node)
          (if ($fns node)
              (let ((gv (reduce #'$+ (mapcar (lambda (fn) (funcall fn)) (reverse ($fns node))))))
                (setf ($fns node) nil)
                (setf ($gradientv node) gv))
              (setf ($gradientv node) (if ($tensorp ($data node))
                                          ($one ($data node))
                                          1))))
        ($gradientv node))
      (let ((o (if ($tensorp ($data node))
                   (apply #'zeros ($size ($data node)))
                   0)))
        (setf ($gradientv node) o)
        (setf ($fns node) nil)
        ($gradientv node))))

(defun $pfn! (node fn) (when ($gradientp node) (push fn ($fns node))))

(defgeneric $gs! (node &optional gradient)
  (:documentation "Set the gradient value, mostly for backpropagation."))

(defmethod $gs! ((node node) &optional gradient)
  (setf ($fns node) nil)
  (let ((gradient (or gradient (if ($tensorp ($data node))
                                   ($one ($data node))
                                   1))))
    (setf ($gradientv node) gradient)))

(defun $gp! (node input &rest inputs)
  (setf ($gradientp node) (reduce (lambda (r i) (or r ($gradientp i))) inputs
                                  :initial-value ($gradientp input))))

(defgeneric $cg! (node))

(defmethod $cg! ((node node))
  (setf ($fns node) nil)
  (setf ($gradientv node) nil))

(defun node (data &optional need-gradient-p)
  (let ((n (make-instance 'node)))
    (setf ($data n) data)
    (setf ($gradientp n) need-gradient-p)
    n))

(defmethod $variable ((node node)) (setf ($gradientp node) t) node)
(defmethod $constant ((node node)) (setf ($gradientp node) nil) node)

(defmethod $variable ((data list)) (node (tensor data) t))
(defmethod $constant ((data list)) (node (tensor data) nil))

(defmethod $variable ((data t)) (node data t))
(defmethod $constant ((data t)) (node data nil))

(defmethod $zero ((x node)) (node ($zero ($data x)) ($gradientp x)))
(defmethod $one ((x node)) (node ($one ($data x)) ($gradientp x)))
(defmethod $fill ((x node) value) (node ($fill ($data x) value) ($gradientp x)))
(defmethod $ndim ((x node)) ($ndim ($data x)))
(defmethod $count ((x node)) ($count ($data x)))

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

(defgeneric $attr (node key &optional default))
(defmethod $attr ((node node) key &optional default)
  (let ((v ($ ($attrs node) key nil)))
    (when (and (null v) default)
      (setf ($ ($attrs node) key) default)
      (setf v default))
    v))

(defmethod (setf $attr) (value (node node) key)
  (setf ($ ($attrs node) key) value)
  value)
