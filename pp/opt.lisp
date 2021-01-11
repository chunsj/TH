(in-package :th.pp)

(defun objective-function (posterior initial-values)
  (lambda (va)
    (let ((p (apply posterior
                    (map 'list (lambda (v v0)
                                 (if (integerp v0)
                                     (round v)
                                     v))
                         va initial-values))))
      (if p
          ($neg p)
          most-positive-double-float))))

(defun map/fit (posterior initial-values)
  (let ((vs (grnm/minimize (objective-function posterior initial-values) initial-values)))
    (coerce vs 'list)))
