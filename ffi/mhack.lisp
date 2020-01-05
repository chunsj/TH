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

#+sbcl
(defvar *custom-alloc-slots* (make-hash-table :synchronized t))
#+ccl
(defvar *custom-alloc-slots* (make-hash-table))
#+ccl
(defvar *custom-alloc-vectors* (make-hash-table))

#+sbcl
(defun allocated-slot-counts () ($count (hash-table-keys *custom-alloc-slots*)))
#+ccl
(defun allocated-slot-counts () ($count (hash-table-keys *custom-alloc-slots*)))

#+sbcl
(defun allocated-foreign-memory-size ()
  (loop :for k :in (hash-table-keys *custom-alloc-slots*)
        :for sz = (or ($ *custom-alloc-slots* k) 0)
        :summing sz))
#+ccl
(defun allocated-foreign-memory-size ()
  (loop :for k :in (hash-table-keys *custom-alloc-slots*)
        :for sz = (or ($ *custom-alloc-slots* k) 0)
        :summing sz))

#+sbcl
(defun foreign-memory-stats ()
  (let ((scount (allocated-slot-counts))
        (amsz (allocated-foreign-memory-size)))
    (list amsz scount)))

#+ccl
(defun foreign-memory-stats ()
  (let ((scount (allocated-slot-counts))
        (amsz (allocated-foreign-memory-size)))
    (list amsz scount)))

#+sbcl
(defun gen-stats (&optional (str t))
  (format str "~%~%GEN      ALLOCED    GCS       MAGE~%")
  (loop :for i :from 0 :to sb-vm:+pseudo-static-generation+
        do (format str "~3D ~12D ~6D  ~,4E~%" i
                   (sb-ext:generation-bytes-allocated i)
                   (sb-ext:generation-number-of-gcs i)
                   (sb-ext:generation-average-age i))))

#+sbcl
(defun report-foreign-memory-allocation ()
  (let ((scount (allocated-slot-counts))
        (amsz (allocated-foreign-memory-size)))
    (prn "* ALLOCATED:" (round (/ amsz (* 1024 1024D0))) "MB" "/" "COUNT:" scount)
    (gen-stats)))
#+ccl
(defun report-foreign-memory-allocation ()
  (let ((scount (allocated-slot-counts))
        (amsz (allocated-foreign-memory-size)))
    (prn "* ALLOCATED:" (round (/ amsz (* 1024 1024D0))) "MB" "/" "COUNT:" scount)))

(defun tensor-data-address (tensor)
  (cffi:pointer-address ($handle ($pointer tensor))))

#+sbcl
(defun tensor-data-size (tensor)
  ($ *custom-alloc-slots* (tensor-data-address tensor)))
#+ccl
(defun tensor-data-size (tensor)
  ($ *custom-alloc-slots* (tensor-data-address tensor)))

#+sbcl
(cffi:defcallback malloc (:pointer :void) ((ctx :pointer) (size :long-long))
  (declare (ignore ctx))
  (let ((p (cffi:foreign-alloc :char :count size)))
    (setf ($ *custom-alloc-slots* (cffi:pointer-address p)) size)
    p))
#+ccl
(cffi:defcallback malloc (:pointer :void) ((ctx :pointer) (size :long-long))
  (declare (ignore ctx))
  (multiple-value-bind (a ap) (ccl:make-heap-ivector size '(unsigned-byte 8))
    (setf ($ *custom-alloc-slots* (cffi:pointer-address ap)) size)
    (setf ($ *custom-alloc-vectors* (cffi:pointer-address ap)) a)
    ap))

#+sbcl
(cffi:defcallback free :void ((ctx :pointer) (ptr :pointer))
  (declare (ignore ctx))
  (remhash (cffi:pointer-address ptr) *custom-alloc-slots*)
  (cffi:foreign-free ptr))
#+ccl
(cffi:defcallback free :void ((ctx :pointer) (ptr :pointer))
  (declare (ignore ctx))
  (remhash (cffi:pointer-address ptr) *custom-alloc-slots*)
  (let ((a ($ *custom-alloc-vectors* (cffi:pointer-address ptr))))
    (remhash (cffi:pointer-address ptr) *custom-alloc-vectors*)
    (ccl:dispose-heap-ivector a)))

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
#+ccl
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

#+ccl
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

#+sbcl
(defparameter *original-gc-configs* (current-gc-configs))

#+sbcl
(defun limit-memory (&optional (l1 64))
  (sb-ext:gc :full t)
  (setf (sb-ext:bytes-consed-between-gcs) (* l1 1024) ;; was 8
        (sb-ext:generation-bytes-consed-between-gcs 0) (* 1 1024)
        (sb-ext:generation-bytes-consed-between-gcs 1) (* 1 1024)
        (sb-ext:generation-bytes-consed-between-gcs 2) (* 1 1024)
        (sb-ext:generation-bytes-consed-between-gcs 3) (* 1 1024)
        (sb-ext:generation-bytes-consed-between-gcs 4) (* 1 1024)
        (sb-ext:generation-bytes-consed-between-gcs 5) (* 1 1024)
        (sb-ext:generation-bytes-consed-between-gcs 6) (* 1 1024))
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

(defmacro with-foreign-memory-limit ((&optional (l1 64)) &body body)
  (limit-memory l1)
  `(unwind-protect (progn ,@body)
     (restore-config)))
