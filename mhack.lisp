(in-package :th)

(defvar *mhack-foreign-memory-allocated* nil)
(defvar *mhack-foreign-memory-threshold* (round (/ (sb-ext:dynamic-space-size) 8)))
(defvar *mhack-foreign-allocation-count* 0)

(defun manage-foreign-memory (size)
  (when (and *mhack-foreign-memory-allocated* (> size 0))
    (sb-ext:atomic-update *mhack-foreign-memory-allocated* (lambda (x) (incf x size)))
    (when (>= *mhack-foreign-memory-allocated* *mhack-foreign-memory-threshold*)
      (sb-ext:atomic-update *mhack-foreign-memory-allocated* (lambda (x) (setf x 0)))
      (gc))))

(defun increase-allocation-count ()
  (when *mhack-foreign-allocation-count*
    (sb-ext:atomic-update *mhack-foreign-allocation-count* (lambda (x) (incf x)))))

(defun decrease-allocation-count ()
  (when *mhack-foreign-allocation-count*
    (sb-ext:atomic-update *mhack-foreign-allocation-count* (lambda (x) (decf x)))))

(cffi:defcallback counting-malloc (:pointer :void) ((ctx :pointer) (size :long-long))
  (declare (ignore ctx))
  (increase-allocation-count)
  (manage-foreign-memory size)
  (cffi:foreign-alloc :char :count size))

(cffi:defcallback counting-free :void ((ctx :pointer) (ptr :pointer))
  (declare (ignore ctx))
  (decrease-allocation-count)
  (cffi:foreign-free ptr))

(cffi:defcallback malloc (:pointer :void) ((ctx :pointer) (size :long-long))
  (declare (ignore ctx))
  (manage-foreign-memory size)
  (cffi:foreign-alloc :char :count size))

(cffi:defcallback free :void ((ctx :pointer) (ptr :pointer))
  (declare (ignore ctx))
  (cffi:foreign-free ptr))

(cffi:defcstruct th-allocator
  (malloc :pointer)
  (realloc :pointer)
  (free :pointer))

(cffi:defcvar (*th-default-allocator* "THDefaultAllocator") (:struct th-allocator))

(defvar *original-malloc* (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                                   '(:struct th-allocator) 'malloc))
(defvar *original-free* (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                                 '(:struct th-allocator) 'free))
(defvar *original-realloc* (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                                    '(:struct th-allocator) 'realloc))

(defun set-original-allocator ()
  (setf (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                 '(:struct th-allocator) 'malloc)
        *original-malloc*)
  (setf (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                 '(:struct th-allocator) 'free)
        *original-free*)
  (setf (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                 '(:struct th-allocator) 'realloc)
        *original-realloc*))

(defun set-mhack-counting-allocator ()
  (setf (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                 '(:struct th-allocator) 'malloc)
        (cffi:callback counting-malloc))
  (setf (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                 '(:struct th-allocator) 'free)
        (cffi:callback counting-free))
  (setf (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                 '(:struct th-allocator) 'realloc)
        +nil+))

(defun set-mhack-allocator ()
  (setf (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                 '(:struct th-allocator) 'malloc)
        (cffi:callback malloc))
  (setf (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                 '(:struct th-allocator) 'free)
        (cffi:callback free))
  (setf (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                 '(:struct th-allocator) 'realloc)
        +nil+))

(set-mhack-allocator)

(defmacro with-foreign-memory-limit* (size-mb &body body)
  `(let ((*mhack-foreign-memory-allocated* 0)
         (*mhack-foreign-memory-threshold* (min *mhack-foreign-memory-threshold*
                                                (* ,size-mb 1024 1024))))
     ,@body))

(defmacro with-foreign-memory-limit (&body body)
  `(with-foreign-memory-limit* 4096
     ,@body))
