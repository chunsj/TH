(in-package :th)

;; real* THStorage_(data)(const THStorage*);
(cffi:defcfun ("THCharStorage_data" th-char-storage-data) (:pointer :char)
  (storage th-char-storage-ptr))
;; ptrdiff_t THStorage_(size)(const THStorage*);
(cffi:defcfun ("THCharStorage_size" th-char-storage-size) :long-long
  (storage th-char-storage-ptr))
;; size_t THStorage_(elementSize)(void);
(cffi:defcfun ("THCharStorage_elementSize" th-char-storage-element-size) :long)

;; /* slow access -- checks everything */
;; void THStorage_(set)(THStorage*, ptrdiff_t, real);
(cffi:defcfun ("THCharStorage_set" th-char-storage-set) :void
  (storage th-char-storage-ptr)
  (loc :long-long)
  (value :char))
;; real THStorage_(get)(const THStorage*, ptrdiff_t);
(cffi:defcfun ("THCharStorage_get" th-char-storage-get) :char
  (storage th-char-storage-ptr)
  (loc :long-long))

;; THStorage* THStorage_(new)(void);
(cffi:defcfun ("THCharStorage_new" th-char-storage-new) th-char-storage-ptr)
;; THStorage* THStorage_(newWithSize)(ptrdiff_t size);
(cffi:defcfun ("THCharStorage_newWithSize" th-char-storage-new-with-size) th-char-storage-ptr
  (size :long-long))
;; THStorage* THStorage_(newWithSize1)(real);
(cffi:defcfun ("THCharStorage_newWithSize1" th-char-storage-new-with-size1) th-char-storage-ptr
  (size :char))
;; THStorage* THStorage_(newWithSize2)(real, real);
(cffi:defcfun ("THCharStorage_newWithSize2" th-char-storage-new-with-size2) th-char-storage-ptr
  (size1 :char)
  (size2 :char))
;; THStorage* THStorage_(newWithSize3)(real, real, real);
(cffi:defcfun ("THCharStorage_newWithSize3" th-char-storage-new-with-size3) th-char-storage-ptr
  (size1 :char)
  (size2 :char)
  (size3 :char))
;; THStorage* THStorage_(newWithSize4)(real, real, real, real);
(cffi:defcfun ("THCharStorage_newWithSize4" th-char-storage-new-with-size4) th-char-storage-ptr
  (size1 :char)
  (size2 :char)
  (size3 :char)
  (size4 :char))
;; THStorage* THStorage_(newWithMapping)(const char *filename, ptrdiff_t size, int flags);
(cffi:defcfun ("THCharStorage_newWithMapping" th-char-storage-new-with-mapping)
    th-char-storage-ptr
  (filename :string)
  (size :long-long)
  (flags :int))

;; /* takes ownership of data */
;; THStorage* THStorage_(newWithData)(real *data, ptrdiff_t size);
(cffi:defcfun ("THCharStorage_newWithData" th-char-storage-new-with-data) th-char-storage-ptr
  (data (:pointer :char))
  (size :long-long))

;; THStorage* THStorage_(newWithAllocator)(ptrdiff_t size,
;;                                         THAllocator* allocator,
;;                                         void *allocatorContext);
;; THStorage* THStorage_(newWithDataAndAllocator)(
;;                       real* data, ptrdiff_t size, THAllocator* allocator, void *allocatorContext);

;; /* should not differ with API */
;; void THStorage_(setFlag)(THStorage *storage, const char flag);
(cffi:defcfun ("THCharStorage_setFlag" th-char-storage-set-flag) :void
  (storage th-char-storage-ptr)
  (flag :char))
;; void THStorage_(clearFlag)(THStorage *storage, const char flag);
(cffi:defcfun ("THCharStorage_clearFlag" th-char-storage-clear-flag) :void
  (storage th-char-storage-ptr)
  (flag :char))
;; void THStorage_(retain)(THStorage *storage);
(cffi:defcfun ("THCharStorage_retain" th-char-storage-retain) :void
  (storage th-char-storage-ptr))
;; void THStorage_(swap)(THStorage *storage1, THStorage *storage2);
(cffi:defcfun ("THCharStorage_swap" th-char-storage-swap) :void
  (storage1 th-char-storage-ptr)
  (storage2 th-char-storage-ptr))

;; /* might differ with other API (like CUDA) */
;; void THStorage_(free)(THStorage *storage);
(cffi:defcfun ("THCharStorage_free" th-char-storage-free) :void
  (storage th-char-storage-ptr))
;; void THStorage_(resize)(THStorage *storage, ptrdiff_t size);
(cffi:defcfun ("THCharStorage_resize" th-char-storage-resize) :void
  (storage th-char-storage-ptr)
  (size :long-long))
;; void THStorage_(fill)(THStorage *storage, real value);
(cffi:defcfun ("THCharStorage_fill" th-char-storage-fill) :void
  (storage th-char-storage-ptr)
  (value :char))

;; void THStorage_(rawCopy)(THStorage *storage, real *src);
(cffi:defcfun ("THCharStorage_rawCopy" th-char-storage-raw-copy) :void
  (storage th-char-storage-ptr)
  (src (:pointer :char)))
;; void THStorage_(copy)(THStorage *storage, THStorage *src);
(cffi:defcfun ("THCharStorage_copy" th-char-storage-copy) :void
  (storage th-char-storage-ptr)
  (src th-char-storage-ptr))
;; void THStorage_(copyByte)(THStorage *storage, struct THByteStorage *src);
(cffi:defcfun ("THCharStorage_copyByte" th-char-storage-copy-byte) :void
  (storage th-char-storage-ptr)
  (src th-byte-storage-ptr))
;; void THStorage_(copyChar)(THStorage *storage, struct THCharStorage *src);
(cffi:defcfun ("THCharStorage_copyChar" th-char-storage-copy-char) :void
  (storage th-char-storage-ptr)
  (src th-char-storage-ptr))
;; void THStorage_(copyShort)(THStorage *storage, struct THShortStorage *src);
(cffi:defcfun ("THCharStorage_copyShort" th-char-storage-copy-short) :void
  (storage th-char-storage-ptr)
  (src th-short-storage-ptr))
;; void THStorage_(copyInt)(THStorage *storage, struct THIntStorage *src);
(cffi:defcfun ("THCharStorage_copyInt" th-char-storage-copy-int) :void
  (storage th-char-storage-ptr)
  (src th-int-storage-ptr))
;; void THStorage_(copyLong)(THStorage *storage, struct THLongStorage *src);
(cffi:defcfun ("THCharStorage_copyLong" th-char-storage-copy-long) :void
  (storage th-char-storage-ptr)
  (src th-long-storage-ptr))
;; void THStorage_(copyFloat)(THStorage *storage, struct THFloatStorage *src);
(cffi:defcfun ("THCharStorage_copyFloat" th-char-storage-copy-float) :void
  (storage th-char-storage-ptr)
  (src th-float-storage-ptr))
;; void THStorage_(copyDouble)(THStorage *storage, struct THDoubleStorage *src);
(cffi:defcfun ("THCharStorage_copyDouble" th-char-storage-copy-double) :void
  (storage th-char-storage-ptr)
  (src th-double-storage-ptr))
