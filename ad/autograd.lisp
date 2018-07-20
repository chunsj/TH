(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defgeneric $bp! (node &optional gradient) (:documentation "Executes backward error propagation."))
(defgeneric $zg! (node) (:documentation "Reset previous gradient values."))

(defgeneric $variable (object) (:documentation "Returns variable node."))
(defgeneric $constant (object) (:documentation "Returns constant node."))

(defgeneric $broadcast (constant matrix))

(defclass node ()
  ((data :initform nil :accessor $data)
   (gradient :initform nil :accessor $gradient)
   (need-gradient-p :initform nil :accessor $gradientp)
   (children :initform nil :accessor $children)
   (backward-function :initform nil :accessor $bpfn)
   (attrs :initform #{} :accessor $attrs)
   (nm :initform nil :accessor $name)
   (bphook :initform nil :accessor $bphookfn)))

(defmethod print-object ((node node) stream)
  (format stream "[~A] " (if (null ($name node))
                            (cond (($gradientp node) "VARIABLE")
                                  (t "CONSTANT"))
                            ($name node)))
  (print-object ($data node) stream))

(defun $c0 (node) ($0 ($children node)))
(defun $c1 (node) ($1 ($children node)))
(defun $c2 (node) ($2 ($children node)))

(defmethod $tensorp ((node node)) ($tensorp ($data node)))

(defun setgradient (node value)
  (if ($gradient node)
      (setf ($gradient node) ($add ($gradient node) value))
      (setf ($gradient node) value)))

(defun default-bpfn (node gradient)
  (setgradient node gradient)
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

(defun runbpfn (node gradient)
  (let ((r (funcall ($bpfn node) node gradient)))
    (when ($bphookfn node) (funcall ($bphookfn node) node))
    r))

(defun bpglobal (node)
  (if ($tensorp node)
      (runbpfn node ($broadcast 1 ($data node)))
      (runbpfn node 1)))

(defun bplocal (node gradient) (runbpfn node gradient))

(defmethod $bp! ((node node) &optional gradient)
  (if (null gradient)
      (bpglobal node)
      (bplocal node gradient)))

(defun $bptt! (nodes &optional gradient)
  (loop :for node :in nodes :do ($bp! node gradient)))

(defun bps! (states)
  (loop :for (s ps) :in (cdr states)
        :for g = ($gradient ps)
        :do ($bp! s g)))

(defun $bpst! (states &rest more-states)
  (loop :for sts :in (cons states more-states) :do (bps! sts)))

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

(defgeneric $state (node))
(defmethod $state ((node node)) ($variable ($clone ($data node))))
