(in-package :th)

(defun $xwpb (x w b &optional ones)
  (let ((o (or ones ($constant (ones (if (eq 1 ($ndim x)) 1 ($size x 0)))))))
    ($add ($mm x w) ($vv o b))))
