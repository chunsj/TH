(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defgeneric $parameter (object) (:documentation "Returns a wrapped, differentiable  object."))
(defgeneric $parameterp (object) (:documentation "Returns whether object is a parameter or not."))
(defgeneric $gradient (node) (:documentation "Returns gradient value."))
(defgeneric $attr (node key &optional default) (:documentation "An attribute for key in node."))

(defgeneric $cg! (node) (:documentation "Clear gradient value."))

(defclass node ()
  ((nm :initform :parameter :accessor $name)
   (data :initform nil :accessor $data)
   (fns :initform nil :accessor $fns)
   (gradientv :initform nil :accessor $gradientv)
   (attrs :initform #{} :accessor $attrs))
  (:documentation "Represents a computational node for differentiable parameter."))

(defmethod print-object ((node node) stream)
  (format stream "[~A] ~A" ($name node) ($data node)))

(defun $pfn! (node f) (push f ($fns node)))

(defun node (data &key (name :parameter) link)
  (let ((n (make-instance 'node)))
    (setf ($data n) data)
    (setf ($name n) name)
    (when link (funcall link n))
    n))

(defun $gs! (node &optional gradientv)
  "Set gradient seed value."
  (when ($fns node) (setf ($fns node) nil))
  (setf ($gradientv node) (or gradientv ($one 1))))

(defun accumulate-effects (node)
  (cond (($tensorp ($data node))
         (let ((gv ($zero ($data node))))
           (loop :for f :in ($fns node) :do ($add! gv (funcall f)))
           gv))
        ((numberp ($data node))
         (let ((gv 0D0))
           (loop :for v :in ($fns node) :do (incf gv v))
           gv))))

(defun compute-gradient (node)
  (if ($fns node)
      (let ((gv (accumulate-effects node)))
        (setf ($fns node) nil)
        (setf ($gradientv node) gv))
      ($gs! node))
  ($gradientv node))

(defmethod $gradient ((node node)) (or ($gradientv node) (compute-gradient node)))

(defmethod $cg! ((node node))
  (setf ($fns node) nil
        ($gradientv node) nil))

(defmethod $parameter ((node node)) (node ($data node)))
(defmethod $parameter ((data list)) (node (tensor data)))
(defmethod $parameter ((data t)) (node data))

(defmethod $parameterp ((node node)) T)
(defmethod $parameterp ((object T)) nil)

(defmethod $tensorp ((node node)) ($tensorp ($data node)))

(defmethod $zero ((x node)) (node ($zero ($data x))))
(defmethod $one ((x node)) (node ($one ($data x))))
(defmethod $fill ((x node) value) (node ($fill ($data x) value)))
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

(defmethod $empty ((node node)) ($parameter ($empty ($data node))))
(defmethod $clear ((node node)) ($parameter ($clear ($data node))))

(defmethod $storage ((node node)) ($storage ($data node)))
(defmethod $offset ((node node)) ($offset ($data node)))
(defmethod $size ((node node) &optional dimension) ($size ($data node) dimension))
(defmethod $stride ((node node) &optional dimension) ($stride ($data node) dimension))

(defmethod $attr ((node node) key &optional default)
  (let ((v ($ ($attrs node) key nil)))
    (when (and (null v) default)
      (setf ($ ($attrs node) key) default)
      (setf v default))
    v))

(defmethod (setf $attr) (value (node node) key)
  (setf ($ ($attrs node) key) value)
  value)

(defclass parameters () ((parameters :initform nil :accessor $parameters)))

(defun parameters () (make-instance 'parameters))

(defgeneric $push (parameters parameter) (:documentation "Group the parameter."))

(defmethod $push ((parameters parameters) (node node))
  (push node ($parameters parameters))
  node)

(defmethod $push ((parameters parameters) (data t))
  (let ((v ($parameter data)))
    (push v ($parameters parameters))
    v))

(defmethod $cg! ((parameters list))
  (loop :for p :in parameters :do ($cg! p)))

(defmethod $cg! ((parameters parameters))
  (loop :for p :in ($parameters parameters) :do ($cg! p)))

(defmacro with-node ((self) &body body)
  `(lambda ()
     (let ((dv ($data ,self))
           (gv ($gradient ,self)))
       (declare (ignorable dv gv))
       ,@body)))

(defmacro to (target &body body) `($pfn! ,target (with-node (self) ,@body)))

(defmacro link (&body body) `(lambda (self) ,@body))
