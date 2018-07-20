(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defgeneric $xwpb (x w b &optional ones))

(defmethod $xwpb ((x tensor) (w tensor) (b tensor) &optional ones)
  (let ((o (or ones (ones (if (eq 1 ($ndim x)) 1 ($size x 0))))))
    ($add! ($mm x w) ($vv o b))))

(defmethod $xwpb ((x node) (w node) (b node) &optional ones)
  (let ((o (or ones ($constant (ones (if (eq 1 ($ndim x)) 1 ($size x 0)))))))
    ($add ($mm x w) ($vv o b))))

(defgeneric $wimb (xwi w))

(defmethod $wimb ((xwi list) (w tensor)) ($sum ($index w 0 xwi) 0))
(defmethod $wimb ((xwi tensor.int) (w tensor)) ($sum ($index w 0 xwi) 0))
(defmethod $wimb ((xwi tensor.long) (w tensor)) ($sum ($index w 0 xwi) 0))

(defmethod $wimb ((xwi list) (w node)) ($sum ($index w 0 xwi) 0))
(defmethod $wimb ((xwi tensor.int) (w node)) ($sum ($index w 0 xwi) 0))
(defmethod $wimb ((xwi tensor.long) (w node)) ($sum ($index w 0 xwi) 0))
