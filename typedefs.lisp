(in-package :th)

(defvar +nil+ (cffi:null-pointer))

(cffi:defcstruct th-byte-storage
  (data (:pointer :unsigned-char))
  (size :long-long)
  (ref-count :int)
  (flag :char)
  (allocator (:pointer :void))
  (allocator-context (:pointer :void))
  (view (:pointer (:struct th-byte-storage))))

(cffi:defctype th-byte-storage-ptr (:pointer (:struct th-byte-storage)))

(cffi:defcstruct th-byte-tensor
  (size (:pointer :long))
  (stride (:pointer :long))
  (n-dimension :int)
  (storage (:pointer (:struct th-byte-storage)))
  (storage-offset :long-long)
  (ref-count :int)
  (flag :char))

(cffi:defctype th-byte-tensor-ptr (:pointer (:struct th-byte-tensor)))

(cffi:defcstruct th-char-storage
  (data (:pointer :char))
  (size :long-long)
  (ref-count :int)
  (flag :char)
  (allocator (:pointer :void))
  (allocator-context (:pointer :void))
  (view (:pointer (:struct th-char-storage))))

(cffi:defctype th-char-storage-ptr (:pointer (:struct th-char-storage)))

(cffi:defcstruct th-char-tensor
  (size (:pointer :long))
  (stride (:pointer :long))
  (n-dimension :int)
  (storage (:pointer (:struct th-char-storage)))
  (storage-offset :long-long)
  (ref-count :int)
  (flag :char))

(cffi:defctype th-char-tensor-ptr (:pointer (:struct th-char-tensor)))

(cffi:defcstruct th-short-storage
  (data (:pointer :short))
  (size :long-long)
  (ref-count :int)
  (flag :char)
  (allocator (:pointer :void))
  (allocator-context (:pointer :void))
  (view (:pointer (:struct th-short-storage))))

(cffi:defctype th-short-storage-ptr (:pointer (:struct th-short-storage)))

(cffi:defcstruct th-short-tensor
  (size (:pointer :long))
  (stride (:pointer :long))
  (n-dimension :int)
  (storage (:pointer (:struct th-short-storage)))
  (storage-offset :long-long)
  (ref-count :int)
  (flag :char))

(cffi:defctype th-short-tensor-ptr (:pointer (:struct th-short-tensor)))

(cffi:defcstruct th-int-storage
  (data (:pointer :int))
  (size :long-long)
  (ref-count :int)
  (flag :char)
  (allocator (:pointer :void))
  (allocator-context (:pointer :void))
  (view (:pointer (:struct th-int-storage))))

(cffi:defctype th-int-storage-ptr (:pointer (:struct th-int-storage)))

(cffi:defcstruct th-int-tensor
  (size (:pointer :long))
  (stride (:pointer :long))
  (n-dimension :int)
  (storage (:pointer (:struct th-int-storage)))
  (storage-offset :long-long)
  (ref-count :int)
  (flag :char))

(cffi:defctype th-int-tensor-ptr (:pointer (:struct th-int-tensor)))

(cffi:defcstruct th-long-storage
  (data (:pointer :long))
  (size :long-long)
  (ref-count :int)
  (flag :char)
  (allocator (:pointer :void))
  (allocator-context (:pointer :void))
  (view (:pointer (:struct th-long-storage))))

(cffi:defctype th-long-storage-ptr (:pointer (:struct th-long-storage)))

(cffi:defcstruct th-long-tensor
  (size (:pointer :long))
  (stride (:pointer :long))
  (n-dimension :int)
  (storage (:pointer (:struct th-long-storage)))
  (storage-offset :long-long)
  (ref-count :int)
  (flag :char))

(cffi:defctype th-long-tensor-ptr (:pointer (:struct th-long-tensor)))

(cffi:defcstruct th-float-storage
  (data (:pointer :float))
  (size :long-long)
  (ref-count :int)
  (flag :char)
  (allocator (:pointer :void))
  (allocator-context (:pointer :void))
  (view (:pointer (:struct th-float-storage))))

(cffi:defctype th-float-storage-ptr (:pointer (:struct th-float-storage)))

(cffi:defcstruct th-float-tensor
  (size (:pointer :long))
  (stride (:pointer :long))
  (n-dimension :int)
  (storage (:pointer (:struct th-float-storage)))
  (storage-offset :long-long)
  (ref-count :int)
  (flag :char))

(cffi:defctype th-float-tensor-ptr (:pointer (:struct th-float-tensor)))

(cffi:defcstruct th-double-storage
  (data (:pointer :double))
  (size :long-long)
  (ref-count :int)
  (flag :char)
  (allocator (:pointer :void))
  (allocator-context (:pointer :void))
  (view (:pointer (:struct th-double-storage))))

(cffi:defctype th-double-storage-ptr (:pointer (:struct th-double-storage)))

(cffi:defcstruct th-double-tensor
  (size (:pointer :long))
  (stride (:pointer :long))
  (n-dimension :int)
  (storage (:pointer (:struct th-double-storage)))
  (storage-offset :long-long)
  (ref-count :int)
  (flag :char))

(cffi:defctype th-double-tensor-ptr (:pointer (:struct th-double-tensor)))

(cffi:defctype th-generator-ptr (:pointer :void))
(cffi:defctype th-generator-state-ptr (:pointer :void))

(cffi:defcallback error-handler :void ((msg :string) (data :pointer))
  (declare (ignore data))
  (handler-case
      (progn (error "error: ~A" msg))
    (error () (format t "OOPS IN GENERIC ERROR HANDLER~%"))))

;; static void defaultArgErrorHandlerFunction(int argNumber, const char *msg, void *data)
(cffi:defcallback arg-error-handler :void ((arg-number :int) (msg :string) (data :pointer))
  (declare (ignore data))
  (handler-case
      (progn (error "argerr[~A]: ~A" arg-number msg))
    (error () (format t "OOPS IN ARGUMENT ERROR HANDLER~%"))))

(cffi:defcfun ("THSetDefaultArgErrorHandler" th-set-default-arg-error-handler) :void
  (fn :pointer)
  (data :pointer))

(cffi:defcfun ("THSetDefaultErrorHandler" th-set-default-error-handler) :void
  (fn :pointer)
  (data :pointer))

(th-set-default-error-handler (cffi:callback error-handler) (cffi:null-pointer))
(th-set-default-arg-error-handler (cffi:callback arg-error-handler) (cffi:null-pointer))

(cffi:defctype th-file-ptr (:pointer :void))

(cffi:defcfun ("THSetNumThreads" th-set-num-threads) :void (n :int))
(cffi:defcfun ("THGetNumThreads" th-get-num-threads) :int)

(cffi:defcfun ("omp_get_num_threads" omp-get-num-threads) :int)
(cffi:defcfun ("omp_set_num_threads" omp-set-num-threads) :void (n :int))

(th-set-num-threads 4)
(omp-set-num-threads 4)
