(in-package :th)

(defclass th.object () ((handle :initform nil :accessor $handle)))
(defmethod $handle (null) +nil+)

(defclass storage (th.object) ())
(defclass storage.integral (storage) ())
(defclass storage.fractional (storage) ())

(defclass storage.byte (storage.integral) ())
(defclass storage.char (storage.integral) ())
(defclass storage.short (storage.integral) ())
(defclass storage.int (storage.integral) ())
(defclass storage.long (storage.integral) ())
(defclass storage.float (storage.fractional) ())
(defclass storage.double (storage.fractional) ())

(defclass pointer (th.object) ())
(defclass pointer.integral (pointer) ())
(defclass pointer.fractional (pointer) ())

(defclass pointer.byte (pointer.integral) ())
(defclass pointer.char (pointer.integral) ())
(defclass pointer.short (pointer.integral) ())
(defclass pointer.int (pointer.integral) ())
(defclass pointer.long (pointer.integral) ())
(defclass pointer.float (pointer.fractional) ())
(defclass pointer.double (pointer.fractional) ())
