(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defclass generator (th.object) ())

#+ccl
(defmethod ccl:terminate ((g generator))
  (let ((h ($handle g)))
    (th-generator-free h)
    (setf ($handle g) nil)))


(defun generator (&optional seed)
  (let ((gen (make-instance 'generator))
        (h (th-generator-new)))
    (setf ($handle gen) h)
    #+sbcl (sb-ext:finalize gen (lambda () (th-generator-free h)))
    (when seed
      (th-random-manual-seed ($handle gen) (coerce seed 'integer)))
    gen))

(defparameter *generator* (generator))
