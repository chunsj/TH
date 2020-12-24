(in-package :th.pp)

(defclass r/trace (r/variable)
  ((collection :initform nil)
   (mval :initform nil :accessor r/mapval)
   (burn-ins :initform 0)
   (thin :initform 0)))
