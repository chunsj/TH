(in-package :th)

(defvar *mhack-foreign-memory-size* nil)
(defvar *mhack-threshold* (round (/ (sb-ext:dynamic-space-size) (* 1024 1024) 8)))

(defun hack-gc ()
  (when (and *mhack-foreign-memory-size*
             (>= *mhack-foreign-memory-size* (* *mhack-threshold* 1024 1024)))
    ;;(format t "***** HACK GC! ~A *****~%" *mhack-foreign-memory-size*)
    (setf *mhack-foreign-memory-size* 0)
    (gc)))

(defun mhack (sz)
  (when *mhack-foreign-memory-size*
    (incf *mhack-foreign-memory-size* sz)
    (hack-gc)))

(cffi:defcallback malloc (:pointer :void) ((ctx :pointer) (size :long-long))
  (declare (ignore ctx))
  (mhack size)
  (cffi:foreign-alloc :char :count size))

(cffi:defcallback free :void ((ctx :pointer) (ptr :pointer))
  (declare (ignore ctx))
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

(defmacro with-foreign-memory-hack* (size-mb &body body)
  `(let ((*mhack-foreign-memory-size* 0)
         (*mhack-threshold* (max *mhack-threshold* ,size-mb)))
     ,@body))

(defmacro with-foreign-memory-hack (&body body)
  `(with-foreign-memory-hack* *mhack-threshold*
     ,@body))
