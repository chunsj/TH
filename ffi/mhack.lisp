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

(defparameter *original-gc-configs* (current-gc-configs))

#+sbcl
(defun limit-memory ()
  (setf (sb-ext:bytes-consed-between-gcs) (* 32 1024)
        (sb-ext:generation-bytes-consed-between-gcs 0) (* 32 1024)
        (sb-ext:generation-bytes-consed-between-gcs 1) (* 32 1024)
        (sb-ext:generation-bytes-consed-between-gcs 2) (* 32 1024)
        (sb-ext:generation-bytes-consed-between-gcs 3) (* 32 1024)
        (sb-ext:generation-bytes-consed-between-gcs 4) (* 32 1024)
        (sb-ext:generation-bytes-consed-between-gcs 5) (* 32 1024)
        (sb-ext:generation-bytes-consed-between-gcs 6) (* 32 1024))
  (sb-ext:gc)
  (gcf))

#+sbcl
(defun restore-config ()
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

(defmacro with-mhack (&body body)
  #+sbcl
  (limit-memory)
  `(let ((___r___ ,@body))
     #+sbcl
     (restore-config)
     ___r___))
