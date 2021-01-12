(in-package :th.pp)

(defun objective-function (posterior initial-values)
  (lambda (&rest args)
    (let ((p (apply posterior
                    (map 'list (lambda (v v0)
                                 (if (integerp v0)
                                     (round v)
                                     v))
                         args initial-values))))
      (if p
          ($neg p)
          most-positive-single-float))))

(defun map/fit (posterior initial-values)
  (multiple-value-bind (vs fv)
      (nelder-mead (objective-function posterior initial-values) initial-values)
    (values (loop :for k :from 0 :below ($count vs)
                  :for v = ($ vs k)
                  :for i :in initial-values
                  :collect (if (integerp i)
                               (round v)
                               v))
            fv)))
