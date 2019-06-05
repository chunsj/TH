(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defvar +nil+ (cffi:null-pointer))

(cffi:defcstruct th-allocator
  (malloc :pointer)
  (realloc :pointer)
  (free :pointer))

(cffi:defctype th-generator-ptr (:pointer :void))
(cffi:defctype th-generator-state-ptr (:pointer :void))

(cffi:defcallback error-handler :void ((msg :string) (data :pointer))
  (declare (ignore data))
  (error "THERR: ~A" msg))

;; static void defaultArgErrorHandlerFunction(int argNumber, const char *msg, void *data)
(cffi:defcallback arg-error-handler :void ((arg-number :int) (msg :string) (data :pointer))
  (declare (ignore data))
  (error "THARGERR[~A]: ~A" arg-number msg))

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

;;(cffi:defcfun ("omp_get_num_threads" omp-get-num-threads) :int)
;;(cffi:defcfun ("omp_set_num_threads" omp-set-num-threads) :void (n :int))

;;(th-set-num-threads 4)
;;(omp-set-num-threads 4)

;; macbook 12 2017
(th-set-num-threads 2)

(defparameter *th-type-infos* '(("byte" :unsigned-char "Byte" :long)
                                ("char" :char "Char" :long)
                                ("short" :short "Short" :long)
                                ("int" :int "Int" :long)
                                ("long" :long "Long" :long)
                                ("float" :float "Float" :double)
                                ("double" :double "Double" :double)))

(loop :for td :in *th-type-infos*
      :for typename = (car td)
      :for datatype = (cadr td)
      :do (eval `(define-structs ,typename ,datatype)))
