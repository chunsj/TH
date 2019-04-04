(in-package :th)

(defvar *mhack-foreign-memory-allocated* nil)
(defvar *mhack-foreign-memory-threshold* (round (/ (sb-ext:dynamic-space-size) 8)))
(defvar *mhack-foreign-allocation-count* 0)

(defun manage-foreign-memory (size)
  (when (and *mhack-foreign-memory-allocated* (> size 0))
    (incf *mhack-foreign-memory-allocated* size)
    (when (>= *mhack-foreign-memory-allocated* *mhack-foreign-memory-threshold*)
      (setf *mhack-foreign-memory-allocated* 0)
      (gc))))

(cffi:defcallback malloc (:pointer :void) ((ctx :pointer) (size :long-long))
  (declare (ignore ctx))
  (let ((ptr (cffi:foreign-alloc :char :count size)))
    (incf *mhack-foreign-allocation-count*)
    (manage-foreign-memory size)
    ptr))

(cffi:defcallback free :void ((ctx :pointer) (ptr :pointer))
  (declare (ignore ctx))
  (decf *mhack-foreign-allocation-count*)
  (cffi:foreign-free ptr))

(cffi:defcstruct th-allocator
  (malloc :pointer)
  (realloc :pointer)
  (free :pointer))

(cffi:defcvar (*th-default-allocator* "THDefaultAllocator") (:struct th-allocator))

(setf (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                               '(:struct th-allocator) 'malloc)
      (cffi:callback malloc))
(setf (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                               '(:struct th-allocator) 'free)
      (cffi:callback free))
(setf (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                               '(:struct th-allocator) 'realloc)
      +nil+)

(defmacro with-foreign-memory-limit* (size-mb &body body)
  `(let ((*mhack-foreign-memory-allocated* 0)
         (*mhack-foreign-memory-threshold* (max *mhack-foreign-memory-threshold*
                                                (* ,size-mb 1024 1024))))
     ,@body))

(defmacro with-foreign-memory-limit (&body body)
  `(with-foreign-memory-limit* 0
     ,@body))
