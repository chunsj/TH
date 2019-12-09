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
(setf (sb-ext:bytes-consed-between-gcs) (* 128 1024))
#+sbcl
(setf (sb-ext:generation-bytes-consed-between-gcs 0) (* 128 1024))
#+sbcl
(setf (sb-ext:generation-bytes-consed-between-gcs 1) (* 128 1024))
#+sbcl
(setf (sb-ext:generation-bytes-consed-between-gcs 2) (* 128 1024))
#+sbcl
(setf (sb-ext:generation-bytes-consed-between-gcs 3) (* 256 1024))
#+sbcl
(setf (sb-ext:generation-bytes-consed-between-gcs 4) (* 256 1024))
#+sbcl
(setf (sb-ext:generation-bytes-consed-between-gcs 5) (* 256 1024))
#+sbcl
(setf (sb-ext:generation-bytes-consed-between-gcs 6) (* 512 1024))
#+sbcl
(sb-ext:gc)
