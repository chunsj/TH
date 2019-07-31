(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defclass generator (th.object) ())

#+ccl
(defmethod ccl:terminate ((g generator))
  (when (has-valid-handle-p g)
    (th-generator-free ($handle g))
    (reset-handle g)))


(defun generator (&optional seed)
  (let ((gen (make-instance 'generator))
        (h (th-generator-new)))
    (setf ($handle gen) h)
    #+sbcl (sb-ext:finalize gen (lambda () (th-generator-free h)))
    (when seed
      (th-random-manual-seed ($handle gen) (coerce seed 'integer)))
    gen))

(defparameter *generator* (generator))
