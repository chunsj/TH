(in-package :th)

(defclass generator (th.object) ())

(defun generator (&optional seed)
  (let ((gen (make-instance 'generator))
        (h (th-generator-new)))
    (setf ($handle gen) h)
    (sb-ext:finalize gen (lambda () (th-generator-free h)))
    (when seed
      (th-random-manual-seed ($handle gen) (coerce seed 'integer)))
    gen))

(defparameter *generator* (generator))
