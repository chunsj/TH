(declaim (optimize (speed 3) (debug 0) (safety 0)))

(in-package :th)

(cffi:defcallback thgc :void ((data :pointer))
  (declare (ignore data))
  (gc))

(cffi:defcfun ("THSetGCHandler" th-set-gc-handler) :void
  (fn :pointer)
  (data :pointer))

(th-set-gc-handler (cffi:callback thgc) +nil+)

#+sbcl
(defun current-gc-configs ()
  (list (sb-ext:bytes-consed-between-gcs)
        (sb-ext:generation-bytes-consed-between-gcs 0)
        (sb-ext:generation-bytes-consed-between-gcs 1)
        (sb-ext:generation-bytes-consed-between-gcs 2)
        (sb-ext:generation-bytes-consed-between-gcs 3)
        (sb-ext:generation-bytes-consed-between-gcs 4)
        (sb-ext:generation-bytes-consed-between-gcs 5)
        (sb-ext:generation-bytes-consed-between-gcs 6)))
#-sbcl
(defun current-gc-configs ())

(defparameter *original-gc-configs* (current-gc-configs))

#+sbcl
(defun limit-memory ()
  (gcf)
  (setf (sb-ext:bytes-consed-between-gcs) (* 8 1024)
        (sb-ext:generation-bytes-consed-between-gcs 0) (* 64 1024)
        (sb-ext:generation-bytes-consed-between-gcs 1) (* 64 1024)
        (sb-ext:generation-bytes-consed-between-gcs 2) (* 64 1024)
        (sb-ext:generation-bytes-consed-between-gcs 3) (* 64 1024)
        (sb-ext:generation-bytes-consed-between-gcs 4) (* 64 1024)
        (sb-ext:generation-bytes-consed-between-gcs 5) (* 64 1024)
        (sb-ext:generation-bytes-consed-between-gcs 6) (* 64 1024))
  (sb-ext:gc)
  (gcf))
#-sbcl
(defun limit-memory () (gcf))

#+sbcl
(defun restore-config ()
  (gcf)
  (setf (sb-ext:bytes-consed-between-gcs) ($ *original-gc-configs* 0)
        (sb-ext:generation-bytes-consed-between-gcs 0) ($ *original-gc-configs* 1)
        (sb-ext:generation-bytes-consed-between-gcs 1) ($ *original-gc-configs* 2)
        (sb-ext:generation-bytes-consed-between-gcs 2) ($ *original-gc-configs* 3)
        (sb-ext:generation-bytes-consed-between-gcs 3) ($ *original-gc-configs* 4)
        (sb-ext:generation-bytes-consed-between-gcs 4) ($ *original-gc-configs* 5)
        (sb-ext:generation-bytes-consed-between-gcs 5) ($ *original-gc-configs* 6)
        (sb-ext:generation-bytes-consed-between-gcs 6) ($ *original-gc-configs* 7))
  (sb-ext:gc)
  (gcf))
#-sbcl
(defun restore-config () (gcf))

(defmacro with-foreign-memory-limit (() &body body)
  (limit-memory)
  `(unwind-protect (progn ,@body)
     (restore-config)))
