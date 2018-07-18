(declaim (optimize (speed 3) (debug 0) (safety 0)))

(in-package :th)

;; real* THStorage_(data)(const THStorage*);
(cffi:defcfun ("THFloatStorage_data" th-float-storage-data) (:pointer :float)
  (storage th-float-storage-ptr))
;; ptrdiff_t THStorage_(size)(const THStorage*);
(cffi:defcfun ("THFloatStorage_size" th-float-storage-size) :long-long
  (storage th-float-storage-ptr))
;; size_t THStorage_(elementSize)(void);
(cffi:defcfun ("THFloatStorage_elementSize" th-float-storage-element-size) :long)

;; /* slow access -- checks everything */
;; void THStorage_(set)(THStorage*, ptrdiff_t, real);
(cffi:defcfun ("THFloatStorage_set" th-float-storage-set) :void
  (storage th-float-storage-ptr)
  (loc :long-long)
  (value :float))
;; real THStorage_(get)(const THStorage*, ptrdiff_t);
(cffi:defcfun ("THFloatStorage_get" th-float-storage-get) :float
  (storage th-float-storage-ptr)
  (loc :long-long))

;; THStorage* THStorage_(new)(void);
(cffi:defcfun ("THFloatStorage_new" th-float-storage-new) th-float-storage-ptr)
;; THStorage* THStorage_(newWithSize)(ptrdiff_t size);
(cffi:defcfun ("THFloatStorage_newWithSize" th-float-storage-new-with-size) th-float-storage-ptr
  (size :long-long))
;; THStorage* THStorage_(newWithSize1)(real);
(cffi:defcfun ("THFloatStorage_newWithSize1" th-float-storage-new-with-size1) th-float-storage-ptr
  (size :float))
;; THStorage* THStorage_(newWithSize2)(real, real);
(cffi:defcfun ("THFloatStorage_newWithSize2" th-float-storage-new-with-size2) th-float-storage-ptr
  (size1 :float)
  (size2 :float))
;; THStorage* THStorage_(newWithSize3)(real, real, real);
(cffi:defcfun ("THFloatStorage_newWithSize3" th-float-storage-new-with-size3) th-float-storage-ptr
  (size1 :float)
  (size2 :float)
  (size3 :float))
;; THStorage* THStorage_(newWithSize4)(real, real, real, real);
(cffi:defcfun ("THFloatStorage_newWithSize4" th-float-storage-new-with-size4) th-float-storage-ptr
  (size1 :float)
  (size2 :float)
  (size3 :float)
  (size4 :float))
;; THStorage* THStorage_(newWithMapping)(const char *filename, ptrdiff_t size, int flags);
(cffi:defcfun ("THFloatStorage_newWithMapping" th-float-storage-new-with-mapping)
    th-float-storage-ptr
  (filename :string)
  (size :long-long)
  (flags :int))

;; /* takes ownership of data */
;; THStorage* THStorage_(newWithData)(real *data, ptrdiff_t size);
(cffi:defcfun ("THFloatStorage_newWithData" th-float-storage-new-with-data) th-float-storage-ptr
  (data (:pointer :float))
  (size :long-long))

;; THStorage* THStorage_(newWithAllocator)(ptrdiff_t size,
;;                                         THAllocator* allocator,
;;                                         void *allocatorContext);
;; THStorage* THStorage_(newWithDataAndAllocator)(
;;                       real* data, ptrdiff_t size, THAllocator* allocator, void *allocatorContext);

;; /* should not differ with API */
;; void THStorage_(setFlag)(THStorage *storage, const char flag);
(cffi:defcfun ("THFloatStorage_setFlag" th-float-storage-set-flag) :void
  (storage th-float-storage-ptr)
  (flag :char))
;; void THStorage_(clearFlag)(THStorage *storage, const char flag);
(cffi:defcfun ("THFloatStorage_clearFlag" th-float-storage-clear-flag) :void
  (storage th-float-storage-ptr)
  (flag :char))
;; void THStorage_(retain)(THStorage *storage);
(cffi:defcfun ("THFloatStorage_retain" th-float-storage-retain) :void
  (storage th-float-storage-ptr))
;; void THStorage_(swap)(THStorage *storage1, THStorage *storage2);
(cffi:defcfun ("THFloatStorage_swap" th-float-storage-swap) :void
  (storage1 th-float-storage-ptr)
  (storage2 th-float-storage-ptr))

;; /* might differ with other API (like CUDA) */
;; void THStorage_(free)(THStorage *storage);
(cffi:defcfun ("THFloatStorage_free" th-float-storage-free) :void
  (storage th-float-storage-ptr))
;; void THStorage_(resize)(THStorage *storage, ptrdiff_t size);
(cffi:defcfun ("THFloatStorage_resize" th-float-storage-resize) :void
  (storage th-float-storage-ptr)
  (size :long-long))
;; void THStorage_(fill)(THStorage *storage, real value);
(cffi:defcfun ("THFloatStorage_fill" th-float-storage-fill) :void
  (storage th-float-storage-ptr)
  (value :float))

;; void THStorage_(rawCopy)(THStorage *storage, real *src);
(cffi:defcfun ("THFloatStorage_rawCopy" th-float-storage-raw-copy) :void
  (storage th-float-storage-ptr)
  (src (:pointer :float)))
;; void THStorage_(copy)(THStorage *storage, THStorage *src);
(cffi:defcfun ("THFloatStorage_copy" th-float-storage-copy) :void
  (storage th-float-storage-ptr)
  (src th-float-storage-ptr))
;; void THStorage_(copyByte)(THStorage *storage, struct THByteStorage *src);
(cffi:defcfun ("THFloatStorage_copyByte" th-float-storage-copy-byte) :void
  (storage th-float-storage-ptr)
  (src th-byte-storage-ptr))
;; void THStorage_(copyChar)(THStorage *storage, struct THCharStorage *src);
(cffi:defcfun ("THFloatStorage_copyChar" th-float-storage-copy-char) :void
  (storage th-float-storage-ptr)
  (src th-char-storage-ptr))
;; void THStorage_(copyShort)(THStorage *storage, struct THShortStorage *src);
(cffi:defcfun ("THFloatStorage_copyShort" th-float-storage-copy-short) :void
  (storage th-float-storage-ptr)
  (src th-short-storage-ptr))
;; void THStorage_(copyInt)(THStorage *storage, struct THIntStorage *src);
(cffi:defcfun ("THFloatStorage_copyInt" th-float-storage-copy-int) :void
  (storage th-float-storage-ptr)
  (src th-int-storage-ptr))
;; void THStorage_(copyLong)(THStorage *storage, struct THLongStorage *src);
(cffi:defcfun ("THFloatStorage_copyLong" th-float-storage-copy-long) :void
  (storage th-float-storage-ptr)
  (src th-long-storage-ptr))
;; void THStorage_(copyFloat)(THStorage *storage, struct THFloatStorage *src);
(cffi:defcfun ("THFloatStorage_copyFloat" th-float-storage-copy-float) :void
  (storage th-float-storage-ptr)
  (src th-float-storage-ptr))
;; void THStorage_(copyDouble)(THStorage *storage, struct THDoubleStorage *src);
(cffi:defcfun ("THFloatStorage_copyDouble" th-float-storage-copy-double) :void
  (storage th-float-storage-ptr)
  (src th-double-storage-ptr))
