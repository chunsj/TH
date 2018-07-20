(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

;; real* THStorage_(data)(const THStorage*);
(cffi:defcfun ("THLongStorage_data" th-long-storage-data) (:pointer :long)
  (storage th-long-storage-ptr))
;; ptrdiff_t THStorage_(size)(const THStorage*);
(cffi:defcfun ("THLongStorage_size" th-long-storage-size) :long-long
  (storage th-long-storage-ptr))
;; size_t THStorage_(elementSize)(void);
(cffi:defcfun ("THLongStorage_elementSize" th-long-storage-element-size) :long)

;; /* slow access -- checks everything */
;; void THStorage_(set)(THStorage*, ptrdiff_t, real);
(cffi:defcfun ("THLongStorage_set" th-long-storage-set) :void
  (storage th-long-storage-ptr)
  (loc :long-long)
  (value :long))
;; real THStorage_(get)(const THStorage*, ptrdiff_t);
(cffi:defcfun ("THLongStorage_get" th-long-storage-get) :long
  (storage th-long-storage-ptr)
  (loc :long-long))

;; THStorage* THStorage_(new)(void);
(cffi:defcfun ("THLongStorage_new" th-long-storage-new) th-long-storage-ptr)
;; THStorage* THStorage_(newWithSize)(ptrdiff_t size);
(cffi:defcfun ("THLongStorage_newWithSize" th-long-storage-new-with-size) th-long-storage-ptr
  (size :long-long))
;; THStorage* THStorage_(newWithSize1)(real);
(cffi:defcfun ("THLongStorage_newWithSize1" th-long-storage-new-with-size1) th-long-storage-ptr
  (size :long))
;; THStorage* THStorage_(newWithSize2)(real, real);
(cffi:defcfun ("THLongStorage_newWithSize2" th-long-storage-new-with-size2) th-long-storage-ptr
  (size1 :long)
  (size2 :long))
;; THStorage* THStorage_(newWithSize3)(real, real, real);
(cffi:defcfun ("THLongStorage_newWithSize3" th-long-storage-new-with-size3) th-long-storage-ptr
  (size1 :long)
  (size2 :long)
  (size3 :long))
;; THStorage* THStorage_(newWithSize4)(real, real, real, real);
(cffi:defcfun ("THLongStorage_newWithSize4" th-long-storage-new-with-size4) th-long-storage-ptr
  (size1 :long)
  (size2 :long)
  (size3 :long)
  (size4 :long))
;; THStorage* THStorage_(newWithMapping)(const char *filename, ptrdiff_t size, int flags);
(cffi:defcfun ("THLongStorage_newWithMapping" th-long-storage-new-with-mapping)
    th-long-storage-ptr
  (filename :string)
  (size :long-long)
  (flags :int))

;; /* takes ownership of data */
;; THStorage* THStorage_(newWithData)(real *data, ptrdiff_t size);
(cffi:defcfun ("THLongStorage_newWithData" th-long-storage-new-with-data) th-long-storage-ptr
  (data (:pointer :long))
  (size :long-long))

;; THStorage* THStorage_(newWithAllocator)(ptrdiff_t size,
;;                                         THAllocator* allocator,
;;                                         void *allocatorContext);
;; THStorage* THStorage_(newWithDataAndAllocator)(
;;                       real* data, ptrdiff_t size, THAllocator* allocator, void *allocatorContext);

;; /* should not differ with API */
;; void THStorage_(setFlag)(THStorage *storage, const char flag);
(cffi:defcfun ("THLongStorage_setFlag" th-long-storage-set-flag) :void
  (storage th-long-storage-ptr)
  (flag :char))
;; void THStorage_(clearFlag)(THStorage *storage, const char flag);
(cffi:defcfun ("THLongStorage_clearFlag" th-long-storage-clear-flag) :void
  (storage th-long-storage-ptr)
  (flag :char))
;; void THStorage_(retain)(THStorage *storage);
(cffi:defcfun ("THLongStorage_retain" th-long-storage-retain) :void
  (storage th-long-storage-ptr))
;; void THStorage_(swap)(THStorage *storage1, THStorage *storage2);
(cffi:defcfun ("THLongStorage_swap" th-long-storage-swap) :void
  (storage1 th-long-storage-ptr)
  (storage2 th-long-storage-ptr))

;; /* might differ with other API (like CUDA) */
;; void THStorage_(free)(THStorage *storage);
(cffi:defcfun ("THLongStorage_free" th-long-storage-free) :void
  (storage th-long-storage-ptr))
;; void THStorage_(resize)(THStorage *storage, ptrdiff_t size);
(cffi:defcfun ("THLongStorage_resize" th-long-storage-resize) :void
  (storage th-long-storage-ptr)
  (size :long-long))
;; void THStorage_(fill)(THStorage *storage, real value);
(cffi:defcfun ("THLongStorage_fill" th-long-storage-fill) :void
  (storage th-long-storage-ptr)
  (value :long))

;; void THStorage_(rawCopy)(THStorage *storage, real *src);
(cffi:defcfun ("THLongStorage_rawCopy" th-long-storage-raw-copy) :void
  (storage th-long-storage-ptr)
  (src (:pointer :long)))
;; void THStorage_(copy)(THStorage *storage, THStorage *src);
(cffi:defcfun ("THLongStorage_copy" th-long-storage-copy) :void
  (storage th-long-storage-ptr)
  (src th-long-storage-ptr))
;; void THStorage_(copyByte)(THStorage *storage, struct THByteStorage *src);
(cffi:defcfun ("THLongStorage_copyByte" th-long-storage-copy-byte) :void
  (storage th-long-storage-ptr)
  (src th-byte-storage-ptr))
;; void THStorage_(copyChar)(THStorage *storage, struct THCharStorage *src);
(cffi:defcfun ("THLongStorage_copyChar" th-long-storage-copy-char) :void
  (storage th-long-storage-ptr)
  (src th-char-storage-ptr))
;; void THStorage_(copyShort)(THStorage *storage, struct THShortStorage *src);
(cffi:defcfun ("THLongStorage_copyShort" th-long-storage-copy-short) :void
  (storage th-long-storage-ptr)
  (src th-short-storage-ptr))
;; void THStorage_(copyInt)(THStorage *storage, struct THIntStorage *src);
(cffi:defcfun ("THLongStorage_copyInt" th-long-storage-copy-int) :void
  (storage th-long-storage-ptr)
  (src th-int-storage-ptr))
;; void THStorage_(copyLong)(THStorage *storage, struct THLongStorage *src);
(cffi:defcfun ("THLongStorage_copyLong" th-long-storage-copy-long) :void
  (storage th-long-storage-ptr)
  (src th-long-storage-ptr))
;; void THStorage_(copyFloat)(THStorage *storage, struct THFloatStorage *src);
(cffi:defcfun ("THLongStorage_copyFloat" th-long-storage-copy-float) :void
  (storage th-long-storage-ptr)
  (src th-float-storage-ptr))
;; void THStorage_(copyDouble)(THStorage *storage, struct THLongStorage *src);
(cffi:defcfun ("THLongStorage_copyDouble" th-long-storage-copy-double) :void
  (storage th-long-storage-ptr)
  (src th-double-storage-ptr))
