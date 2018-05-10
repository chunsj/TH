(in-package :th)

(defmethod $type ((pointer pointer.byte)) :unsigned-char)
(defmethod $type ((pointer pointer.char)) :char)
(defmethod $type ((pointer pointer.short)) :short)
(defmethod $type ((pointer pointer.int)) :int)
(defmethod $type ((pointer pointer.long)) :long)
(defmethod $type ((pointer pointer.float)) :float)
(defmethod $type ((pointer pointer.double)) :double)

(defmethod $coerce ((pointer pointer.byte) value) (coerce value 'unsigned-byte))
(defmethod $coerce ((pointer pointer.char) value) (coerce value 'signed-byte))
(defmethod $coerce ((pointer pointer.short) value) (coerce value 'integer))
(defmethod $coerce ((pointer pointer.int) value) (coerce value 'integer))
(defmethod $coerce ((pointer pointer.long) value) (coerce value 'integer))
(defmethod $coerce ((pointer pointer.float) value) (coerce value 'single-float))
(defmethod $coerce ((pointer pointer.double) value) (coerce value 'double-float))

(defmethod $ ((pointer pointer) location &rest others-and-default)
  (declare (ignore others-and-default))
  (cffi:mem-aref ($handle pointer) ($type pointer) location))

(defmethod (setf $) (value (pointer pointer.byte) location &rest others)
  (declare (ignore others))
  (setf (cffi:mem-aref ($handle pointer) ($type pointer) location)
        ($coerce pointer value)))

(defmethod allocate-storage ((storage storage.byte) &optional size)
  (let ((handle (if size
                    (th-byte-storage-new-with-size size)
                    (th-byte-storage-new))))
    (setf ($handle storage) handle)
    (sb-ext:finalize storage (lambda () (th-byte-storage-free handle)))))
(defmethod allocate-storage ((storage storage.char) &optional size)
  (let ((handle (if size
                    (th-char-storage-new-with-size size)
                    (th-char-storage-new))))
    (setf ($handle storage) handle)
    (sb-ext:finalize storage (lambda () (th-char-storage-free handle)))))
(defmethod allocate-storage ((storage storage.short) &optional size)
  (let ((handle (if size
                    (th-short-storage-new-with-size size)
                    (th-short-storage-new))))
    (setf ($handle storage) handle)
    (sb-ext:finalize storage (lambda () (th-short-storage-free handle)))))
(defmethod allocate-storage ((storage storage.int) &optional size)
  (let ((handle (if size
                    (th-int-storage-new-with-size size)
                    (th-int-storage-new))))
    (setf ($handle storage) handle)
    (sb-ext:finalize storage (lambda () (th-int-storage-free handle)))))
(defmethod allocate-storage ((storage storage.long) &optional size)
  (let ((handle (if size
                    (th-long-storage-new-with-size size)
                    (th-long-storage-new))))
    (setf ($handle storage) handle)
    (sb-ext:finalize storage (lambda () (th-long-storage-free handle)))))
(defmethod allocate-storage ((storage storage.float) &optional size)
  (let ((handle (if size
                    (th-float-storage-new-with-size size)
                    (th-float-storage-new))))
    (setf ($handle storage) handle)
    (sb-ext:finalize storage (lambda () (th-float-storage-free handle)))))
(defmethod allocate-storage ((storage storage.double) &optional size)
  (let ((handle (if size
                    (th-double-storage-new-with-size size)
                    (th-double-storage-new))))
    (setf ($handle storage) handle)
    (sb-ext:finalize storage (lambda () (th-double-storage-free handle)))))

(defmethod $empty ((storage storage)) (make-storage (type-of storage)))

(defun irange (from below) (loop :for i :from from :below below :collect i))
(defun elmn (storage n)
  (let ((p ($pointer storage)))
    (mapcar (lambda (i) ($ p i)) (loop :for i :from 0 :below n :collect i))))
(defun print-storage-object (storage fmt stream)
  (let ((n ($count storage)))
    (format stream fmt (type-of storage) n (elmn storage (min 5 n)) (if (> n 5) " ...]" "]"))))

(defmethod print-object ((storage storage.integral) stream)
  (print-storage-object storage "#<~A (~A) [~{~4D~^ ~}~A>" stream))
(defmethod print-object ((storage storage.fractional) stream)
  (print-storage-object storage "#<~A ~A [~{~,4E~^ ~}~A>" stream))

(defmethod $type ((storage storage.byte)) :unsigned-char)
(defmethod $type ((storage storage.char)) :char)
(defmethod $type ((storage storage.short)) :short)
(defmethod $type ((storage storage.int)) :int)
(defmethod $type ((storage storage.long)) :long)
(defmethod $type ((storage storage.float)) :float)
(defmethod $type ((storage storage.double)) :double)

(defmethod $coerce ((storage storage.byte) value) (coerce value 'unsigned-byte))
(defmethod $coerce ((storage storage.char) value) (coerce value 'signed-byte))
(defmethod $coerce ((storage storage.short) value) (coerce value 'integer))
(defmethod $coerce ((storage storage.int) value) (coerce value 'integer))
(defmethod $coerce ((storage storage.long) value) (coerce value 'integer))
(defmethod $coerce ((storage storage.float) value) (coerce value 'single-float))
(defmethod $coerce ((storage storage.double) value) (coerce value 'double-float))

(defun make-pointer (cls p)
  (let ((pointer (make-instance cls)))
    (setf ($handle pointer) p)
    pointer))

(defmethod $pointer ((storage storage.byte))
  (make-pointer 'pointer.byte (th-byte-storage-data ($handle storage))))
(defmethod $pointer ((storage storage.char))
  (make-pointer 'pointer.char (th-char-storage-data ($handle storage))))
(defmethod $pointer ((storage storage.short))
  (make-pointer 'pointer.short (th-short-storage-data ($handle storage))))
(defmethod $pointer ((storage storage.int))
  (make-pointer 'pointer.int (th-int-storage-data ($handle storage))))
(defmethod $pointer ((storage storage.long))
  (make-pointer 'pointer.long (th-long-storage-data ($handle storage))))
(defmethod $pointer ((storage storage.float))
  (make-pointer 'pointer.float (th-float-storage-data ($handle storage))))
(defmethod $pointer ((storage storage.double))
  (make-pointer 'pointer.double (th-double-storage-data ($handle storage))))

(defmethod $storagep ((storage storage)) t)
(defmethod $storagep ((storage t)) nil)

(defmethod $size ((storage storage.byte) &optional dim)
  (declare (ignore dim))
  (th-byte-storage-size ($handle storage)))
(defmethod $size ((storage storage.char) &optional dim)
  (declare (ignore dim))
  (th-char-storage-size ($handle storage)))
(defmethod $size ((storage storage.short) &optional dim)
  (declare (ignore dim))
  (th-short-storage-size ($handle storage)))
(defmethod $size ((storage storage.int) &optional dim)
  (declare (ignore dim))
  (th-int-storage-size ($handle storage)))
(defmethod $size ((storage storage.long) &optional dim)
  (declare (ignore dim))
  (th-long-storage-size ($handle storage)))
(defmethod $size ((storage storage.float) &optional dim)
  (declare (ignore dim))
  (th-float-storage-size ($handle storage)))
(defmethod $size ((storage storage.double) &optional dim)
  (declare (ignore dim))
  (th-double-storage-size ($handle storage)))

(defmethod $count ((storage storage)) ($size storage))

(defmethod $ ((storage storage.byte) location &rest others-and-default)
  (declare (ignore others-and-default))
  (th-byte-storage-get ($handle storage) location))
(defmethod $ ((storage storage.char) location &rest others-and-default)
  (declare (ignore others-and-default))
  (th-char-storage-get ($handle storage) location))
(defmethod $ ((storage storage.short) location &rest others-and-default)
  (declare (ignore others-and-default))
  (th-short-storage-get ($handle storage) location))
(defmethod $ ((storage storage.int) location &rest others-and-default)
  (declare (ignore others-and-default))
  (th-int-storage-get ($handle storage) location))
(defmethod $ ((storage storage.long) location &rest others-and-default)
  (declare (ignore others-and-default))
  (th-long-storage-get ($handle storage) location))
(defmethod $ ((storage storage.float) location &rest others-and-default)
  (declare (ignore others-and-default))
  (th-float-storage-get ($handle storage) location))
(defmethod $ ((storage storage.double) location &rest others-and-default)
  (declare (ignore others-and-default))
  (th-double-storage-get ($handle storage) location))

(defmethod (setf $) (value (storage storage.byte) location &rest others)
  (declare (ignore others))
  (let ((v (coerce value 'unsigned-byte)))
    (th-byte-storage-set ($handle storage) location v)
    v))
(defmethod (setf $) (value (storage storage.char) location &rest others)
  (declare (ignore others))
  (let ((v (coerce value 'signed-byte)))
    (th-char-storage-set ($handle storage) location v)
    v))
(defmethod (setf $) (value (storage storage.short) location &rest others)
  (declare (ignore others))
  (let ((v (coerce value 'integer)))
    (th-short-storage-set ($handle storage) location v)
    v))
(defmethod (setf $) (value (storage storage.int) location &rest others)
  (declare (ignore others))
  (let ((v (coerce value 'integer)))
    (th-int-storage-set ($handle storage) location v)
    v))
(defmethod (setf $) (value (storage storage.long) location &rest others)
  (declare (ignore others))
  (let ((v (coerce value 'integer)))
    (th-long-storage-set ($handle storage) location v)
    v))
(defmethod (setf $) (value (storage storage.float) location &rest others)
  (declare (ignore others))
  (let ((v (coerce value 'single-float)))
    (th-float-storage-set ($handle storage) location v)
    v))
(defmethod (setf $) (value (storage storage.double) location &rest others)
  (declare (ignore others))
  (let ((v (coerce value 'double-float)))
    (th-double-storage-set ($handle storage) location v)
    v))

(defmethod $swap! ((storage1 storage.byte) (storage2 storage.byte))
  (th-byte-storage-swap ($handle storage1) ($handle storage2)))
(defmethod $swap! ((storage1 storage.char) (storage2 storage.char))
  (th-char-storage-swap ($handle storage1) ($handle storage2)))
(defmethod $swap! ((storage1 storage.short) (storage2 storage.short))
  (th-short-storage-swap ($handle storage1) ($handle storage2)))
(defmethod $swap! ((storage1 storage.int) (storage2 storage.int))
  (th-int-storage-swap ($handle storage1) ($handle storage2)))
(defmethod $swap! ((storage1 storage.long) (storage2 storage.long))
  (th-long-storage-swap ($handle storage1) ($handle storage2)))
(defmethod $swap! ((storage1 storage.float) (storage2 storage.float))
  (th-float-storage-swap ($handle storage1) ($handle storage2)))
(defmethod $swap! ((storage1 storage.double) (storage2 storage.double))
  (th-double-storage-swap ($handle storage1) ($handle storage2)))

(defmethod $resize! ((storage storage.byte) size &optional stride)
  (declare (ignore stride))
  (th-byte-storage-resize ($handle storage) size)
  storage)
(defmethod $resize! ((storage storage.char) size &optional stride)
  (declare (ignore stride))
  (th-char-storage-resize ($handle storage) size)
  storage)
(defmethod $resize! ((storage storage.short) size &optional stride)
  (declare (ignore stride))
  (th-short-storage-resize ($handle storage) size)
  storage)
(defmethod $resize! ((storage storage.int) size &optional stride)
  (declare (ignore stride))
  (th-int-storage-resize ($handle storage) size)
  storage)
(defmethod $resize! ((storage storage.long) size &optional stride)
  (declare (ignore stride))
  (th-long-storage-resize ($handle storage) size)
  storage)
(defmethod $resize! ((storage storage.float) size &optional stride)
  (declare (ignore stride))
  (th-float-storage-resize ($handle storage) size)
  storage)
(defmethod $resize! ((storage storage.double) size &optional stride)
  (declare (ignore stride))
  (th-double-storage-resize ($handle storage) size)
  storage)

(defmethod $fill ((storage storage.byte) value)
  (th-byte-storage-fill ($handle storage) ($coerce storage value))
  storage)
(defmethod $fill ((storage storage.char) value)
  (th-char-storage-fill ($handle storage) ($coerce storage value))
  storage)
(defmethod $fill ((storage storage.short) value)
  (th-short-storage-fill ($handle storage) ($coerce storage value))
  storage)
(defmethod $fill ((storage storage.int) value)
  (th-int-storage-fill ($handle storage) ($coerce storage value))
  storage)
(defmethod $fill ((storage storage.long) value)
  (th-long-storage-fill ($handle storage) ($coerce storage value))
  storage)
(defmethod $fill ((storage storage.float) value)
  (th-float-storage-fill ($handle storage) ($coerce storage value))
  storage)
(defmethod $fill ((storage storage.double) value)
  (th-double-storage-fill ($handle storage) ($coerce storage value))
  storage)

(defmethod $copy! ((storage storage.byte) (src storage.byte))
  (th-byte-storage-copy ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.byte) (src storage.char))
  (th-byte-storage-copy-char ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.byte) (src storage.short))
  (th-byte-storage-copy-short ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.byte) (src storage.int))
  (th-byte-storage-copy-int ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.byte) (src storage.long))
  (th-byte-storage-copy-long ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.byte) (src storage.float))
  (th-byte-storage-copy-float ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.byte) (src storage.double))
  (th-byte-storage-copy-double ($handle storage) ($handle src))
  storage)

(defmethod $copy! ((storage storage.char) (src storage.byte))
  (th-char-storage-copy-byte ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.char) (src storage.char))
  (th-char-storage-copy ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.char) (src storage.short))
  (th-char-storage-copy-short ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.char) (src storage.int))
  (th-char-storage-copy-int ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.char) (src storage.long))
  (th-char-storage-copy-long ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.char) (src storage.float))
  (th-char-storage-copy-float ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.char) (src storage.double))
  (th-char-storage-copy-double ($handle storage) ($handle src))
  storage)

(defmethod $copy! ((storage storage.short) (src storage.byte))
  (th-short-storage-copy-byte ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.short) (src storage.char))
  (th-short-storage-copy-char ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.short) (src storage.short))
  (th-short-storage-copy ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.short) (src storage.int))
  (th-short-storage-copy-int ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.short) (src storage.long))
  (th-short-storage-copy-long ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.short) (src storage.float))
  (th-short-storage-copy-float ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.short) (src storage.double))
  (th-short-storage-copy-double ($handle storage) ($handle src))
  storage)

(defmethod $copy! ((storage storage.int) (src storage.byte))
  (th-int-storage-copy-byte ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.int) (src storage.char))
  (th-int-storage-copy-char ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.int) (src storage.short))
  (th-int-storage-copy-short ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.int) (src storage.int))
  (th-int-storage-copy ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.int) (src storage.long))
  (th-int-storage-copy-long ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.int) (src storage.float))
  (th-int-storage-copy-float ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.int) (src storage.double))
  (th-int-storage-copy-double ($handle storage) ($handle src))
  storage)

(defmethod $copy! ((storage storage.long) (src storage.byte))
  (th-long-storage-copy-byte ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.long) (src storage.char))
  (th-long-storage-copy-char ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.long) (src storage.short))
  (th-long-storage-copy-short ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.long) (src storage.int))
  (th-long-storage-copy-int ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.long) (src storage.long))
  (th-long-storage-copy ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.long) (src storage.float))
  (th-long-storage-copy-float ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.long) (src storage.double))
  (th-long-storage-copy-double ($handle storage) ($handle src))
  storage)

(defmethod $copy! ((storage storage.float) (src storage.byte))
  (th-float-storage-copy-byte ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.float) (src storage.char))
  (th-float-storage-copy-char ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.float) (src storage.short))
  (th-float-storage-copy-short ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.float) (src storage.int))
  (th-float-storage-copy-int ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.float) (src storage.long))
  (th-float-storage-copy-long ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.float) (src storage.float))
  (th-float-storage-copy ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.float) (src storage.double))
  (th-float-storage-copy-double ($handle storage) ($handle src))
  storage)

(defmethod $copy! ((storage storage.double) (src storage.byte))
  (th-double-storage-copy-byte ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.double) (src storage.char))
  (th-double-storage-copy-char ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.double) (src storage.short))
  (th-double-storage-copy-short ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.double) (src storage.int))
  (th-double-storage-copy-int ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.double) (src storage.long))
  (th-double-storage-copy-long ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.double) (src storage.float))
  (th-double-storage-copy-float ($handle storage) ($handle src))
  storage)
(defmethod $copy! ((storage storage.double) (src storage.double))
  (th-double-storage-copy ($handle storage) ($handle src))
  storage)

(defmethod $list ((storage storage))
  (loop :for i :from 0 :below ($count storage) :collect ($ storage i)))
(defmethod $list ((storage list)) storage)
