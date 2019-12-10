(declaim (optimize (speed 3) (debug 0) (safety 0)))

(in-package :th)

(cffi:defcallback thgc :void ((data :pointer))
  (declare (ignore data))
  (gc))

(cffi:defcfun ("THSetGCHandler" th-set-gc-handler) :void
  (fn :pointer)
  (data :pointer))

(th-set-gc-handler (cffi:callback thgc) +nil+)

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

(defvar *custom-alloc-counts* 0)
(defvar *custom-alloc-slots* (make-hash-table :synchronized t))

(defun allocated-foreign-memory-size ()
  (loop :for k :in (hash-table-keys *custom-alloc-slots*)
        :summing ($ *custom-alloc-slots* k)))

(defun tensor-data-address (tensor)
  (cffi:pointer-address ($handle ($pointer tensor))))

(defun tensor-data-size (tensor)
  ($ *custom-alloc-slots* (tensor-data-address tensor)))

#+sbcl
(cffi:defcallback malloc (:pointer :void) ((ctx :pointer) (size :long-long))
  (declare (ignore ctx))
  (let ((p (cffi:foreign-alloc :char :count size)))
    (sb-ext:atomic-update *custom-alloc-counts* (lambda (x) (incf x)))
    (setf ($ *custom-alloc-slots* (cffi:pointer-address p)) size)
    p))

#+sbcl
(cffi:defcallback free :void ((ctx :pointer) (ptr :pointer))
  (declare (ignore ctx))
  (remhash (cffi:pointer-address ptr) *custom-alloc-slots*)
  (sb-ext:atomic-update *custom-alloc-counts* (lambda (x) (decf x)))
  (cffi:foreign-free ptr))

#+sbcl
(defun use-custom-allocator ()
  (setf (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                 '(:struct th-allocator) 'malloc)
        (cffi:callback malloc))
  (setf (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                 '(:struct th-allocator) 'free)
        (cffi:callback free))
  (setf (cffi:foreign-slot-value (cffi:get-var-pointer '*th-default-allocator*)
                                 '(:struct th-allocator) 'realloc)
        +nil+))

#+sbcl
(use-custom-allocator)

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
#-sbcl
(defun current-gc-configs ())

(defparameter *original-gc-configs* (current-gc-configs))

#+sbcl
(defun limit-memory ()
  (sb-ext:gc :full t)
  (setf (sb-ext:bytes-consed-between-gcs) (* 8 1024)
        (sb-ext:generation-bytes-consed-between-gcs 0) (* 64 1024)
        (sb-ext:generation-bytes-consed-between-gcs 1) (* 64 1024)
        (sb-ext:generation-bytes-consed-between-gcs 2) (* 64 1024)
        (sb-ext:generation-bytes-consed-between-gcs 3) (* 64 1024)
        (sb-ext:generation-bytes-consed-between-gcs 4) (* 64 1024)
        (sb-ext:generation-bytes-consed-between-gcs 5) (* 64 1024)
        (sb-ext:generation-bytes-consed-between-gcs 6) (* 64 1024))
  (sb-ext:gc :full t))
#-sbcl
(defun limit-memory () (gcf))

#+sbcl
(defun restore-config ()
  (sb-ext:gc :full t)
  (setf (sb-ext:bytes-consed-between-gcs) ($ *original-gc-configs* 0)
        (sb-ext:generation-bytes-consed-between-gcs 0) ($ *original-gc-configs* 1)
        (sb-ext:generation-bytes-consed-between-gcs 1) ($ *original-gc-configs* 2)
        (sb-ext:generation-bytes-consed-between-gcs 2) ($ *original-gc-configs* 3)
        (sb-ext:generation-bytes-consed-between-gcs 3) ($ *original-gc-configs* 4)
        (sb-ext:generation-bytes-consed-between-gcs 4) ($ *original-gc-configs* 5)
        (sb-ext:generation-bytes-consed-between-gcs 5) ($ *original-gc-configs* 6)
        (sb-ext:generation-bytes-consed-between-gcs 6) ($ *original-gc-configs* 7))
  (sb-ext:gc :full t))
#-sbcl
(defun restore-config () (gcf))

(defmacro with-foreign-memory-limit (() &body body)
  (limit-memory)
  `(unwind-protect (progn ,@body)
     (restore-config)))
