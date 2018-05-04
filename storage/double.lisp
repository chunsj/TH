(in-package :th)

;; real* THStorage_(data)(const THStorage*);
(cffi:defcfun ("THDoubleStorage_data" th-double-storage-data) (:pointer :double)
  (storage th-double-storage-ptr))
;; ptrdiff_t THStorage_(size)(const THStorage*);
(cffi:defcfun ("THDoubleStorage_size" th-double-storage-size) :long-long
  (storage th-double-storage-ptr))
;; size_t THStorage_(elementSize)(void);
(cffi:defcfun ("THDoubleStorage_elementSize" th-double-storage-element-size) :long)

;; /* slow access -- checks everything */
;; void THStorage_(set)(THStorage*, ptrdiff_t, real);
(cffi:defcfun ("THDoubleStorage_set" th-double-storage-set) :void
  (storage th-double-storage-ptr)
  (loc :long-long)
  (value :double))
;; real THStorage_(get)(const THStorage*, ptrdiff_t);
(cffi:defcfun ("THDoubleStorage_get" th-double-storage-get) :double
  (storage th-double-storage-ptr)
  (loc :long-long))

;; THStorage* THStorage_(new)(void);
(cffi:defcfun ("THDoubleStorage_new" th-double-storage-new) th-double-storage-ptr)
;; THStorage* THStorage_(newWithSize)(ptrdiff_t size);
(cffi:defcfun ("THDoubleStorage_newWithSize" th-double-storage-new-with-size) th-double-storage-ptr
  (size :long-long))
;; THStorage* THStorage_(newWithSize1)(real);
(cffi:defcfun ("THDoubleStorage_newWithSize1" th-double-storage-new-with-size1) th-double-storage-ptr
  (size :double))
;; THStorage* THStorage_(newWithSize2)(real, real);
(cffi:defcfun ("THDoubleStorage_newWithSize2" th-double-storage-new-with-size2) th-double-storage-ptr
  (size1 :double)
  (size2 :double))
;; THStorage* THStorage_(newWithSize3)(real, real, real);
(cffi:defcfun ("THDoubleStorage_newWithSize3" th-double-storage-new-with-size3) th-double-storage-ptr
  (size1 :double)
  (size2 :double)
  (size3 :double))
;; THStorage* THStorage_(newWithSize4)(real, real, real, real);
(cffi:defcfun ("THDoubleStorage_newWithSize4" th-double-storage-new-with-size4) th-double-storage-ptr
  (size1 :double)
  (size2 :double)
  (size3 :double)
  (size4 :double))
;; THStorage* THStorage_(newWithMapping)(const char *filename, ptrdiff_t size, int flags);
(cffi:defcfun ("THDoubleStorage_newWithMapping" th-double-storage-new-with-mapping)
    th-double-storage-ptr
  (filename :string)
  (size :long-long)
  (flags :int))

;; /* takes ownership of data */
;; THStorage* THStorage_(newWithData)(real *data, ptrdiff_t size);
(cffi:defcfun ("THDoubleStorage_newWithData" th-double-storage-new-with-data) th-double-storage-ptr
  (data (:pointer :double))
  (size :long-long))

;; THStorage* THStorage_(newWithAllocator)(ptrdiff_t size,
;;                                         THAllocator* allocator,
;;                                         void *allocatorContext);
;; THStorage* THStorage_(newWithDataAndAllocator)(
;;                       real* data, ptrdiff_t size, THAllocator* allocator, void *allocatorContext);

;; /* should not differ with API */
;; void THStorage_(setFlag)(THStorage *storage, const char flag);
(cffi:defcfun ("THDoubleStorage_setFlag" th-double-storage-set-flag) :void
  (storage th-double-storage-ptr)
  (flag :char))
;; void THStorage_(clearFlag)(THStorage *storage, const char flag);
(cffi:defcfun ("THDoubleStorage_clearFlag" th-double-storage-clear-flag) :void
  (storage th-double-storage-ptr)
  (flag :char))
;; void THStorage_(retain)(THStorage *storage);
(cffi:defcfun ("THDoubleStorage_retain" th-double-storage-retain) :void
  (storage th-double-storage-ptr))
;; void THStorage_(swap)(THStorage *storage1, THStorage *storage2);
(cffi:defcfun ("THDoubleStorage_swap" th-double-storage-swap) :void
  (storage1 th-double-storage-ptr)
  (storage2 th-double-storage-ptr))

;; /* might differ with other API (like CUDA) */
;; void THStorage_(free)(THStorage *storage);
(cffi:defcfun ("THDoubleStorage_free" th-double-storage-free) :void
  (storage th-double-storage-ptr))
;; void THStorage_(resize)(THStorage *storage, ptrdiff_t size);
(cffi:defcfun ("THDoubleStorage_resize" th-double-storage-resize) :void
  (storage th-double-storage-ptr)
  (size :long-long))
;; void THStorage_(fill)(THStorage *storage, real value);
(cffi:defcfun ("THDoubleStorage_fill" th-double-storage-fill) :void
  (storage th-double-storage-ptr)
  (value :double))

;; void THStorage_(rawCopy)(THStorage *storage, real *src);
(cffi:defcfun ("THDoubleStorage_rawCopy" th-double-storage-raw-copy) :void
  (storage th-double-storage-ptr)
  (src (:pointer :double)))
;; void THStorage_(copy)(THStorage *storage, THStorage *src);
(cffi:defcfun ("THDoubleStorage_copy" th-double-storage-copy) :void
  (storage th-double-storage-ptr)
  (src th-double-storage-ptr))
;; void THStorage_(copyByte)(THStorage *storage, struct THByteStorage *src);
(cffi:defcfun ("THDoubleStorage_copyByte" th-double-storage-copy-byte) :void
  (storage th-double-storage-ptr)
  (src th-byte-storage-ptr))
;; void THStorage_(copyChar)(THStorage *storage, struct THCharStorage *src);
(cffi:defcfun ("THDoubleStorage_copyChar" th-double-storage-copy-char) :void
  (storage th-double-storage-ptr)
  (src th-char-storage-ptr))
;; void THStorage_(copyShort)(THStorage *storage, struct THShortStorage *src);
(cffi:defcfun ("THDoubleStorage_copyShort" th-double-storage-copy-short) :void
  (storage th-double-storage-ptr)
  (src th-short-storage-ptr))
;; void THStorage_(copyInt)(THStorage *storage, struct THIntStorage *src);
(cffi:defcfun ("THDoubleStorage_copyInt" th-double-storage-copy-int) :void
  (storage th-double-storage-ptr)
  (src th-int-storage-ptr))
;; void THStorage_(copyLong)(THStorage *storage, struct THLongStorage *src);
(cffi:defcfun ("THDoubleStorage_copyLong" th-double-storage-copy-long) :void
  (storage th-double-storage-ptr)
  (src th-long-storage-ptr))
;; void THStorage_(copyFloat)(THStorage *storage, struct THFloatStorage *src);
(cffi:defcfun ("THDoubleStorage_copyFloat" th-double-storage-copy-float) :void
  (storage th-double-storage-ptr)
  (src th-float-storage-ptr))
;; void THStorage_(copyDouble)(THStorage *storage, struct THDoubleStorage *src);
(cffi:defcfun ("THDoubleStorage_copyDouble" th-double-storage-copy-double) :void
  (storage th-double-storage-ptr)
  (src th-double-storage-ptr))
