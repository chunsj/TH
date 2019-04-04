(in-package :th)

(defvar *mhack-foreign-memory-allocated* nil)
(defvar *mhack-foreign-memory-threshold* (round (/ (sb-ext:dynamic-space-size) 8)))
(defvar *mhack-foreign-allocation-count* (cons 0 nil))

(defun manage-foreign-memory (size)
  (when (and *mhack-foreign-memory-allocated* (> size 0))
    (sb-ext:atomic-incf (car *mhack-foreign-memory-allocated*) size)
    (when (>= (car *mhack-foreign-memory-allocated*) *mhack-foreign-memory-threshold*)
      (let ((x *mhack-foreign-memory-allocated*))
        (sb-ext:atomic-decf (car *mhack-foreign-memory-allocated*) x))
      (gc))))

(cffi:defcallback malloc (:pointer :void) ((ctx :pointer) (size :long-long))
  (declare (ignore ctx))
  (sb-ext:atomic-incf (car *mhack-foreign-allocation-count*) 1)
  (manage-foreign-memory size)
  (cffi:foreign-alloc :char :count size))

(cffi:defcallback free :void ((ctx :pointer) (ptr :pointer))
  (declare (ignore ctx))
  (sb-ext:atomic-decf (car *mhack-foreign-allocation-count*) 1)
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
  `(let ((*mhack-foreign-memory-allocated* (cons 0 nil))
         (*mhack-foreign-memory-threshold* (max *mhack-foreign-memory-threshold*
                                                (* ,size-mb 1024 1024))))
     ,@body))

(defmacro with-foreign-memory-limit (&body body)
  `(with-foreign-memory-limit* 0
     ,@body))
