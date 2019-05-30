(declaim (optimize (speed 3) (debug 0) (safety 0)))

(in-package :th)

(defvar *mhack-foreign-memory-allocated* nil)
(defvar *mhack-foreign-memory-threshold* nil)
(defvar *mhack-foreign-memory-threshold-default* 2048)

(cffi:defcfun ("get_current_memory_usage" mh-get-current-memory-usage) :long)
(defun current-memory-usage () (round (/ (mh-get-current-memory-usage) 1024 1024)))

(cffi:defcvar (*th-default-allocator* "THDefaultAllocator") (:struct th-allocator))

(defvar *original-malloc* (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                                   '(:struct th-allocator) 'malloc))
(defvar *original-free* (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                                 '(:struct th-allocator) 'free))
(defvar *original-realloc* (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                                    '(:struct th-allocator) 'realloc))

(defun use-default-allocator ()
  (setf (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                 '(:struct th-allocator) 'malloc)
        *original-malloc*)
  (setf (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                 '(:struct th-allocator) 'free)
        *original-free*)
  (setf (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                 '(:struct th-allocator) 'realloc)
        *original-realloc*))

(defun manage-foreign-memory (size)
  (when (and *mhack-foreign-memory-allocated* (> size 0))
    (sb-ext:atomic-update *mhack-foreign-memory-allocated* (lambda (x) (incf x size)))
    (when (>= *mhack-foreign-memory-allocated* *mhack-foreign-memory-threshold*)
      (sb-ext:atomic-update *mhack-foreign-memory-allocated* (lambda (x) (setf x 0)))
      (sb-ext:gc :full T))))

(defun manage-current-memory-size (size)
  (when (and *mhack-foreign-memory-threshold* (> size 0))
    (when (> (current-memory-usage) *mhack-foreign-memory-threshold*)
      (prn "called")
      (sb-ext:gc :full T))))

(cffi:defcallback malloc (:pointer :void) ((ctx :pointer) (size :long-long))
  (declare (ignore ctx))
  ;;(manage-foreign-memory size)
  (manage-current-memory-size size)
  (cffi:foreign-alloc :char :count size))

(cffi:defcallback free :void ((ctx :pointer) (ptr :pointer))
  (declare (ignore ctx))
  (cffi:foreign-free ptr))

(defun use-mhack-allocator ()
  (setf (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                 '(:struct th-allocator) 'malloc)
        (cffi:callback malloc))
  (setf (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                 '(:struct th-allocator) 'free)
        (cffi:callback free))
  (setf (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                 '(:struct th-allocator) 'realloc)
        +nil+))

(use-mhack-allocator)

(defmacro with-foreign-memory-limit* (size-mb &body body)
  `(let ((*mhack-foreign-memory-allocated* 0)
         (*mhack-foreign-memory-threshold* (* ,size-mb 1024 1024)))
     ,@body))

(defmacro with-foreign-memory-limit (&body body)
  `(with-foreign-memory-limit* *mhack-foreign-memory-threshold-default*
     ,@body))
