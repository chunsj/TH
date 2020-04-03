(declaim (optimize (speed 3) (debug 0) (safety 0)))

(in-package :th)

;; libTHTensor functions
(cffi:defcfun ("THSetLispGCManager" th-set-lisp-gc-manager) :void
  (fn :pointer))
(cffi:defcfun ("THGetCurrentHeapSize" th-get-current-heap-size) :long-long)

;; for management
(defvar *maximum-allowed-heap-size* (* 1 1024 1024 1024))
(defun th-set-maximum-allowed-heap-size (sz-in-mb)
  (setf *maximum-allowed-heap-size* (* sz-in-mb (* 1024 1024))))
(defun th-get-maximum-allowed-heap-size () *maximum-allowed-heap-size*)
(cffi:defcallback check-and-gc :void ()
  (when (> (th-get-current-heap-size) *maximum-allowed-heap-size*)
    (gcf)))

;; set gc manager for memory management
(th-set-lisp-gc-manager (cffi:callback check-and-gc))
