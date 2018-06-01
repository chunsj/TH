(in-package :th)

(defgeneric $xwpb (x w b &optional ones))

(defmethod $xwpb ((x tensor) (w tensor) (b tensor) &optional ones)
  (let ((o (or ones (ones (if (eq 1 ($ndim x)) 1 ($size x 0))))))
    ($add! ($mm x w) ($vv o b))))

(defmethod $xwpb ((x node) (w node) (b node) &optional ones)
  (let ((o (or ones ($constant (ones (if (eq 1 ($ndim x)) 1 ($size x 0)))))))
    ($add ($mm x w) ($vv o b))))
