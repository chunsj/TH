(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

;; XXX gradient should be plain tensor or number, not parameter
;; XXX FIX THIS/CHECK THIS

(defclass parameter ()
  ((id :initform (gensym) :accessor $id)
   (data :initform nil :initarg :data :accessor $data)
   (gradient :initform nil :accessor $gradient)
   (references :initform #{} :accessor $references)
   (attrs :initform #{} :accessor $attrs)))

(defclass operation (parameter)
  ((opername :initform nil :accessor $name)
   (creators :initform nil :accessor $creators)
   (bfn :initform nil :accessor $bfn)))

(defmethod print-object ((p parameter) stream) (print-object ($data p) stream))

(defgeneric $parameter (data))

(defmethod $parameter ((data list)) (make-instance 'parameter :data (tensor data)))
(defmethod $parameter ((data tensor)) (make-instance 'parameter :data data))
(defmethod $parameter ((data number)) (make-instance 'parameter :data data))

(defun $operation (data &key creators name bfn id)
  (let ((o (make-instance 'operation)))
    (setf ($data o) data
          ($creators o) creators
          ($name o) name
          ($bfn o) bfn)
    (when id (setf ($id o) id))
    (when creators
      (loop :for creator :in creators
            :do (if ($ ($references creator) ($id o) nil)
                    (incf ($ ($references creator) ($id o)))
                    (setf ($ ($references creator) ($id o)) 1))))
    o))

(defgeneric $bp! (operation &optional gradient origin))

(defun set-gradient! (p &optional gradient origin)
  (when origin
    (if (zerop ($ ($references p) ($id origin)))
        (error "cannot backpropagate more than once")
        (decf ($ ($references p) ($id origin)))))
  (if ($gradient p)
      (setf ($gradient p) ($add ($gradient p) gradient))
      (setf ($gradient p) gradient)))

(defmethod $bp! ((p parameter) &optional gradient origin)
  (set-gradient! p gradient origin))

(defun all-references-visited-p (p)
  (let ((references ($references p)))
    (loop :for id :being :the :hash-keys :of references
          :for cnt = ($ references id)
          :when (not (zerop cnt))
            :return nil)
    t))

(defmethod $bp! ((o operation) &optional gradient origin)
  (let ((gradient (or gradient ($one ($data o)))))
    (set-gradient! o gradient origin)
    (when (and ($creators o) (or (null origin) (all-references-visited-p o)))
      (funcall ($bfn o) o ($gradient o)))
    o))

(defmethod $ndim ((p parameter)) ($ndim ($data p)))
(defmethod $size ((p parameter) &optional dimension) ($size ($data p) dimension))
(defmethod $count ((p parameter)) ($count ($data p)))
