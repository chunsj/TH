(in-package :th)

;; real* THStorage_(data)(const THStorage*);
(cffi:defcfun ("THByteStorage_data" th-byte-storage-data) (:pointer :unsigned-char)
  (storage th-byte-storage-ptr))
;; ptrdiff_t THStorage_(size)(const THStorage*);
(cffi:defcfun ("THByteStorage_size" th-byte-storage-size) :long-long
  (storage th-byte-storage-ptr))
;; size_t THStorage_(elementSize)(void);
(cffi:defcfun ("THByteStorage_elementSize" th-byte-storage-element-size) :long)

;; /* slow access -- checks everything */
;; void THStorage_(set)(THStorage*, ptrdiff_t, real);
(cffi:defcfun ("THByteStorage_set" th-byte-storage-set) :void
  (storage th-byte-storage-ptr)
  (loc :long-long)
  (value :unsigned-char))
;; real THStorage_(get)(const THStorage*, ptrdiff_t);
(cffi:defcfun ("THByteStorage_get" th-byte-storage-get) :unsigned-char
  (storage th-byte-storage-ptr)
  (loc :long-long))

;; THStorage* THStorage_(new)(void);
(cffi:defcfun ("THByteStorage_new" th-byte-storage-new) th-byte-storage-ptr)
;; THStorage* THStorage_(newWithSize)(ptrdiff_t size);
(cffi:defcfun ("THByteStorage_newWithSize" th-byte-storage-new-with-size) th-byte-storage-ptr
  (size :long-long))
;; THStorage* THStorage_(newWithSize1)(real);
(cffi:defcfun ("THByteStorage_newWithSize1" th-byte-storage-new-with-size1) th-byte-storage-ptr
  (size :unsigned-char))
;; THStorage* THStorage_(newWithSize2)(real, real);
(cffi:defcfun ("THByteStorage_newWithSize2" th-byte-storage-new-with-size2) th-byte-storage-ptr
  (size1 :unsigned-char)
  (size2 :unsigned-char))
;; THStorage* THStorage_(newWithSize3)(real, real, real);
(cffi:defcfun ("THByteStorage_newWithSize3" th-byte-storage-new-with-size3) th-byte-storage-ptr
  (size1 :unsigned-char)
  (size2 :unsigned-char)
  (size3 :unsigned-char))
;; THStorage* THStorage_(newWithSize4)(real, real, real, real);
(cffi:defcfun ("THByteStorage_newWithSize4" th-byte-storage-new-with-size4) th-byte-storage-ptr
  (size1 :unsigned-char)
  (size2 :unsigned-char)
  (size3 :unsigned-char)
  (size4 :unsigned-char))
;; THStorage* THStorage_(newWithMapping)(const char *filename, ptrdiff_t size, int flags);
(cffi:defcfun ("THByteStorage_newWithMapping" th-byte-storage-new-with-mapping)
    th-byte-storage-ptr
  (filename :string)
  (size :long-long)
  (flags :int))

;; /* takes ownership of data */
;; THStorage* THStorage_(newWithData)(real *data, ptrdiff_t size);
(cffi:defcfun ("THByteStorage_newWithData" th-byte-storage-new-with-data) th-byte-storage-ptr
  (data (:pointer :unsigned-char))
  (size :long-long))

;; THStorage* THStorage_(newWithAllocator)(ptrdiff_t size,
;;                                         THAllocator* allocator,
;;                                         void *allocatorContext);
;; THStorage* THStorage_(newWithDataAndAllocator)(
;;                       real* data, ptrdiff_t size, THAllocator* allocator, void *allocatorContext);

;; /* should not differ with API */
;; void THStorage_(setFlag)(THStorage *storage, const char flag);
(cffi:defcfun ("THByteStorage_setFlag" th-byte-storage-set-flag) :void
  (storage th-byte-storage-ptr)
  (flag :char))
;; void THStorage_(clearFlag)(THStorage *storage, const char flag);
(cffi:defcfun ("THByteStorage_clearFlag" th-byte-storage-clear-flag) :void
  (storage th-byte-storage-ptr)
  (flag :char))
;; void THStorage_(retain)(THStorage *storage);
(cffi:defcfun ("THByteStorage_retain" th-byte-storage-retain) :void
  (storage th-byte-storage-ptr))
;; void THStorage_(swap)(THStorage *storage1, THStorage *storage2);
(cffi:defcfun ("THByteStorage_swap" th-byte-storage-swap) :void
  (storage1 th-byte-storage-ptr)
  (storage2 th-byte-storage-ptr))

;; /* might differ with other API (like CUDA) */
;; void THStorage_(free)(THStorage *storage);
(cffi:defcfun ("THByteStorage_free" th-byte-storage-free) :void
  (storage th-byte-storage-ptr))
;; void THStorage_(resize)(THStorage *storage, ptrdiff_t size);
(cffi:defcfun ("THByteStorage_resize" th-byte-storage-resize) :void
  (storage th-byte-storage-ptr)
  (size :long-long))
;; void THStorage_(fill)(THStorage *storage, real value);
(cffi:defcfun ("THByteStorage_fill" th-byte-storage-fill) :void
  (storage th-byte-storage-ptr)
  (value :unsigned-char))

;; void THStorage_(rawCopy)(THStorage *storage, real *src);
(cffi:defcfun ("THByteStorage_rawCopy" th-byte-storage-raw-copy) :void
  (storage th-byte-storage-ptr)
  (src (:pointer :unsigned-char)))
;; void THStorage_(copy)(THStorage *storage, THStorage *src);
(cffi:defcfun ("THByteStorage_copy" th-byte-storage-copy) :void
  (storage th-byte-storage-ptr)
  (src th-byte-storage-ptr))
;; void THStorage_(copyByte)(THStorage *storage, struct THByteStorage *src);
(cffi:defcfun ("THByteStorage_copyByte" th-byte-storage-copy-byte) :void
  (storage th-byte-storage-ptr)
  (src th-byte-storage-ptr))
;; void THStorage_(copyChar)(THStorage *storage, struct THCharStorage *src);
(cffi:defcfun ("THByteStorage_copyChar" th-byte-storage-copy-char) :void
  (storage th-byte-storage-ptr)
  (src th-char-storage-ptr))
;; void THStorage_(copyShort)(THStorage *storage, struct THShortStorage *src);
(cffi:defcfun ("THByteStorage_copyShort" th-byte-storage-copy-short) :void
  (storage th-byte-storage-ptr)
  (src th-short-storage-ptr))
;; void THStorage_(copyInt)(THStorage *storage, struct THIntStorage *src);
(cffi:defcfun ("THByteStorage_copyInt" th-byte-storage-copy-int) :void
  (storage th-byte-storage-ptr)
  (src th-int-storage-ptr))
;; void THStorage_(copyLong)(THStorage *storage, struct THLongStorage *src);
(cffi:defcfun ("THByteStorage_copyLong" th-byte-storage-copy-long) :void
  (storage th-byte-storage-ptr)
  (src th-long-storage-ptr))
;; void THStorage_(copyFloat)(THStorage *storage, struct THFloatStorage *src);
(cffi:defcfun ("THByteStorage_copyFloat" th-byte-storage-copy-float) :void
  (storage th-byte-storage-ptr)
  (src th-float-storage-ptr))
;; void THStorage_(copyDouble)(THStorage *storage, struct THDoubleStorage *src);
(cffi:defcfun ("THByteStorage_copyDouble" th-byte-storage-copy-double) :void
  (storage th-byte-storage-ptr)
  (src th-double-storage-ptr))
