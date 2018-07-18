(declaim (optimize (speed 3) (debug 0) (safety 0)))

(in-package :th)

(defclass file (th.object) ())
(defclass file.disk (file) ())
(defclass file.pipe (file.disk) ())
(defclass file.memory (file) ())

(defun file.disk (name mode &optional quietp)
  (let ((n (make-instance 'file.disk))
        (h (th-disk-file-new name mode (if quietp 1 0))))
    (setf ($handle n) h)
    (sb-ext:finalize n (lambda () (th-file-free h)))
    n))

(defun file.pipe (name mode &optional quietp)
  (let ((n (make-instance 'file.pipe))
        (h (th-pipe-file-new name mode (if quietp 1 0))))
    (setf ($handle n) h)
    (sb-ext:finalize n (lambda () (th-file-free h)))
    n))

(defun file.memory (mode &optional storage)
  (let ((n (make-instance 'file.memory))
        (h (if storage
               (th-memory-file-new-with-storage ($handle storage) mode)
               (th-memory-file-new mode))))
    (setf ($handle n) h)
    (sb-ext:finalize n (lambda () (th-file-free h)))
    n))
