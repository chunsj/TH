(in-package :th)

(defclass tensor (th.object) ())
(defclass tensor.integral (tensor) ())
(defclass tensor.fractional (tensor) ())

(defclass tensor.byte (tensor.integral) ())
(defclass tensor.char (tensor.integral) ())
(defclass tensor.short (tensor.integral) ())
(defclass tensor.int (tensor.integral) ())
(defclass tensor.long (tensor.integral) ())
(defclass tensor.float (tensor.fractional) ())
(defclass tensor.double (tensor.fractional) ())
