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
(setf (sb-ext:bytes-consed-between-gcs) (* 16 1024 1024))
