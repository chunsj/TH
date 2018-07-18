(declaim (optimize (speed 3) (debug 0) (safety 0)))

(in-package :th)

;; real* THStorage_(data)(const THStorage*);
(cffi:defcfun ("THShortStorage_data" th-short-storage-data) (:pointer :short)
  (storage th-short-storage-ptr))
;; ptrdiff_t THStorage_(size)(const THStorage*);
(cffi:defcfun ("THShortStorage_size" th-short-storage-size) :long-long
  (storage th-short-storage-ptr))
;; size_t THStorage_(elementSize)(void);
(cffi:defcfun ("THShortStorage_elementSize" th-short-storage-element-size) :long)

;; /* slow access -- checks everything */
;; void THStorage_(set)(THStorage*, ptrdiff_t, real);
(cffi:defcfun ("THShortStorage_set" th-short-storage-set) :void
  (storage th-short-storage-ptr)
  (loc :long-long)
  (value :short))
;; real THStorage_(get)(const THStorage*, ptrdiff_t);
(cffi:defcfun ("THShortStorage_get" th-short-storage-get) :short
  (storage th-short-storage-ptr)
  (loc :long-long))

;; THStorage* THStorage_(new)(void);
(cffi:defcfun ("THShortStorage_new" th-short-storage-new) th-short-storage-ptr)
;; THStorage* THStorage_(newWithSize)(ptrdiff_t size);
(cffi:defcfun ("THShortStorage_newWithSize" th-short-storage-new-with-size) th-short-storage-ptr
  (size :long-long))
;; THStorage* THStorage_(newWithSize1)(real);
(cffi:defcfun ("THShortStorage_newWithSize1" th-short-storage-new-with-size1) th-short-storage-ptr
  (size :short))
;; THStorage* THStorage_(newWithSize2)(real, real);
(cffi:defcfun ("THShortStorage_newWithSize2" th-short-storage-new-with-size2) th-short-storage-ptr
  (size1 :short)
  (size2 :short))
;; THStorage* THStorage_(newWithSize3)(real, real, real);
(cffi:defcfun ("THShortStorage_newWithSize3" th-short-storage-new-with-size3) th-short-storage-ptr
  (size1 :short)
  (size2 :short)
  (size3 :short))
;; THStorage* THStorage_(newWithSize4)(real, real, real, real);
(cffi:defcfun ("THShortStorage_newWithSize4" th-short-storage-new-with-size4) th-short-storage-ptr
  (size1 :short)
  (size2 :short)
  (size3 :short)
  (size4 :short))
;; THStorage* THStorage_(newWithMapping)(const char *filename, ptrdiff_t size, int flags);
(cffi:defcfun ("THShortStorage_newWithMapping" th-short-storage-new-with-mapping)
    th-short-storage-ptr
  (filename :string)
  (size :long-long)
  (flags :int))

;; /* takes ownership of data */
;; THStorage* THStorage_(newWithData)(real *data, ptrdiff_t size);
(cffi:defcfun ("THShortStorage_newWithData" th-short-storage-new-with-data) th-short-storage-ptr
  (data (:pointer :short))
  (size :long-long))

;; THStorage* THStorage_(newWithAllocator)(ptrdiff_t size,
;;                                         THAllocator* allocator,
;;                                         void *allocatorContext);
;; THStorage* THStorage_(newWithDataAndAllocator)(
;;                       real* data, ptrdiff_t size, THAllocator* allocator, void *allocatorContext);

;; /* should not differ with API */
;; void THStorage_(setFlag)(THStorage *storage, const char flag);
(cffi:defcfun ("THShortStorage_setFlag" th-short-storage-set-flag) :void
  (storage th-short-storage-ptr)
  (flag :char))
;; void THStorage_(clearFlag)(THStorage *storage, const char flag);
(cffi:defcfun ("THShortStorage_clearFlag" th-short-storage-clear-flag) :void
  (storage th-short-storage-ptr)
  (flag :char))
;; void THStorage_(retain)(THStorage *storage);
(cffi:defcfun ("THShortStorage_retain" th-short-storage-retain) :void
  (storage th-short-storage-ptr))
;; void THStorage_(swap)(THStorage *storage1, THStorage *storage2);
(cffi:defcfun ("THShortStorage_swap" th-short-storage-swap) :void
  (storage1 th-short-storage-ptr)
  (storage2 th-short-storage-ptr))

;; /* might differ with other API (like CUDA) */
;; void THStorage_(free)(THStorage *storage);
(cffi:defcfun ("THShortStorage_free" th-short-storage-free) :void
  (storage th-short-storage-ptr))
;; void THStorage_(resize)(THStorage *storage, ptrdiff_t size);
(cffi:defcfun ("THShortStorage_resize" th-short-storage-resize) :void
  (storage th-short-storage-ptr)
  (size :long-long))
;; void THStorage_(fill)(THStorage *storage, real value);
(cffi:defcfun ("THShortStorage_fill" th-short-storage-fill) :void
  (storage th-short-storage-ptr)
  (value :short))

;; void THStorage_(rawCopy)(THStorage *storage, real *src);
(cffi:defcfun ("THShortStorage_rawCopy" th-short-storage-raw-copy) :void
  (storage th-short-storage-ptr)
  (src (:pointer :short)))
;; void THStorage_(copy)(THStorage *storage, THStorage *src);
(cffi:defcfun ("THShortStorage_copy" th-short-storage-copy) :void
  (storage th-short-storage-ptr)
  (src th-short-storage-ptr))
;; void THStorage_(copyByte)(THStorage *storage, struct THByteStorage *src);
(cffi:defcfun ("THShortStorage_copyByte" th-short-storage-copy-byte) :void
  (storage th-short-storage-ptr)
  (src th-byte-storage-ptr))
;; void THStorage_(copyChar)(THStorage *storage, struct THCharStorage *src);
(cffi:defcfun ("THShortStorage_copyChar" th-short-storage-copy-char) :void
  (storage th-short-storage-ptr)
  (src th-char-storage-ptr))
;; void THStorage_(copyShort)(THStorage *storage, struct THShortStorage *src);
(cffi:defcfun ("THShortStorage_copyShort" th-short-storage-copy-short) :void
  (storage th-short-storage-ptr)
  (src th-short-storage-ptr))
;; void THStorage_(copyInt)(THStorage *storage, struct THIntStorage *src);
(cffi:defcfun ("THShortStorage_copyInt" th-short-storage-copy-int) :void
  (storage th-short-storage-ptr)
  (src th-int-storage-ptr))
;; void THStorage_(copyLong)(THStorage *storage, struct THLongStorage *src);
(cffi:defcfun ("THShortStorage_copyLong" th-short-storage-copy-long) :void
  (storage th-short-storage-ptr)
  (src th-long-storage-ptr))
;; void THStorage_(copyFloat)(THStorage *storage, struct THFloatStorage *src);
(cffi:defcfun ("THShortStorage_copyFloat" th-short-storage-copy-float) :void
  (storage th-short-storage-ptr)
  (src th-float-storage-ptr))
;; void THStorage_(copyDouble)(THStorage *storage, struct THDoubleStorage *src);
(cffi:defcfun ("THShortStorage_copyDouble" th-short-storage-copy-double) :void
  (storage th-short-storage-ptr)
  (src th-double-storage-ptr))
