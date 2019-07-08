(declaim (optimize (speed 3) (debug 0) (safety 0)))

(in-package :th)

(defvar *mhack-foreign-memory-threshold-default* (* 2 1024 1024 1024))

(cffi:defcallback thgc :void ((data :pointer))
  (declare (ignore data))
  (sb-ext:gc :full T))

(cffi:defcfun ("THSetGCHandler" th-set-gc-handler) :void
  (fn :pointer)
  (data :pointer))

(cffi:defcfun ("THSetGCHardMax" th-set-gc-hard-max) :void
  (hm :long-long))

(cffi:defcfun ("THGetHeapSize" th-get-heap-size) :long-long)
(cffi:defcfun ("THGetHeapDelta" th-get-heap-delta) :long-long)
(cffi:defcfun ("THGetHeapSoftmax" th-get-heap-softmax) :long-long)

(th-set-gc-hard-max *mhack-foreign-memory-threshold-default*)
(th-set-gc-handler (cffi:callback thgc) +nil+)

(defun set-gc-threshold (sz)
  (when (> sz (* 300 1024 1024))
    (th-set-gc-hard-max sz)))
