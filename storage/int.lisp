(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

;; real* THStorage_(data)(const THStorage*);
(cffi:defcfun ("THIntStorage_data" th-int-storage-data) (:pointer :int)
  (storage th-int-storage-ptr))
;; ptrdiff_t THStorage_(size)(const THStorage*);
(cffi:defcfun ("THIntStorage_size" th-int-storage-size) :long-long
  (storage th-int-storage-ptr))
;; size_t THStorage_(elementSize)(void);
(cffi:defcfun ("THIntStorage_elementSize" th-int-storage-element-size) :long)

;; /* slow access -- checks everything */
;; void THStorage_(set)(THStorage*, ptrdiff_t, real);
(cffi:defcfun ("THIntStorage_set" th-int-storage-set) :void
  (storage th-int-storage-ptr)
  (loc :long-long)
  (value :int))
;; real THStorage_(get)(const THStorage*, ptrdiff_t);
(cffi:defcfun ("THIntStorage_get" th-int-storage-get) :int
  (storage th-int-storage-ptr)
  (loc :long-long))

;; THStorage* THStorage_(new)(void);
(cffi:defcfun ("THIntStorage_new" th-int-storage-new) th-int-storage-ptr)
;; THStorage* THStorage_(newWithSize)(ptrdiff_t size);
(cffi:defcfun ("THIntStorage_newWithSize" th-int-storage-new-with-size) th-int-storage-ptr
  (size :long-long))
;; THStorage* THStorage_(newWithSize1)(real);
(cffi:defcfun ("THIntStorage_newWithSize1" th-int-storage-new-with-size1) th-int-storage-ptr
  (size :int))
;; THStorage* THStorage_(newWithSize2)(real, real);
(cffi:defcfun ("THIntStorage_newWithSize2" th-int-storage-new-with-size2) th-int-storage-ptr
  (size1 :int)
  (size2 :int))
;; THStorage* THStorage_(newWithSize3)(real, real, real);
(cffi:defcfun ("THIntStorage_newWithSize3" th-int-storage-new-with-size3) th-int-storage-ptr
  (size1 :int)
  (size2 :int)
  (size3 :int))
;; THStorage* THStorage_(newWithSize4)(real, real, real, real);
(cffi:defcfun ("THIntStorage_newWithSize4" th-int-storage-new-with-size4) th-int-storage-ptr
  (size1 :int)
  (size2 :int)
  (size3 :int)
  (size4 :int))
;; THStorage* THStorage_(newWithMapping)(const char *filename, ptrdiff_t size, int flags);
(cffi:defcfun ("THIntStorage_newWithMapping" th-int-storage-new-with-mapping)
    th-int-storage-ptr
  (filename :string)
  (size :long-long)
  (flags :int))

;; /* takes ownership of data */
;; THStorage* THStorage_(newWithData)(real *data, ptrdiff_t size);
(cffi:defcfun ("THIntStorage_newWithData" th-int-storage-new-with-data) th-int-storage-ptr
  (data (:pointer :int))
  (size :long-long))

;; THStorage* THStorage_(newWithAllocator)(ptrdiff_t size,
;;                                         THAllocator* allocator,
;;                                         void *allocatorContext);
;; THStorage* THStorage_(newWithDataAndAllocator)(
;;                       real* data, ptrdiff_t size, THAllocator* allocator, void *allocatorContext);

;; /* should not differ with API */
;; void THStorage_(setFlag)(THStorage *storage, const char flag);
(cffi:defcfun ("THIntStorage_setFlag" th-int-storage-set-flag) :void
  (storage th-int-storage-ptr)
  (flag :char))
;; void THStorage_(clearFlag)(THStorage *storage, const char flag);
(cffi:defcfun ("THIntStorage_clearFlag" th-int-storage-clear-flag) :void
  (storage th-int-storage-ptr)
  (flag :char))
;; void THStorage_(retain)(THStorage *storage);
(cffi:defcfun ("THIntStorage_retain" th-int-storage-retain) :void
  (storage th-int-storage-ptr))
;; void THStorage_(swap)(THStorage *storage1, THStorage *storage2);
(cffi:defcfun ("THIntStorage_swap" th-int-storage-swap) :void
  (storage1 th-int-storage-ptr)
  (storage2 th-int-storage-ptr))

;; /* might differ with other API (like CUDA) */
;; void THStorage_(free)(THStorage *storage);
(cffi:defcfun ("THIntStorage_free" th-int-storage-free) :void
  (storage th-int-storage-ptr))
;; void THStorage_(resize)(THStorage *storage, ptrdiff_t size);
(cffi:defcfun ("THIntStorage_resize" th-int-storage-resize) :void
  (storage th-int-storage-ptr)
  (size :long-long))
;; void THStorage_(fill)(THStorage *storage, real value);
(cffi:defcfun ("THIntStorage_fill" th-int-storage-fill) :void
  (storage th-int-storage-ptr)
  (value :int))

;; void THStorage_(rawCopy)(THStorage *storage, real *src);
(cffi:defcfun ("THIntStorage_rawCopy" th-int-storage-raw-copy) :void
  (storage th-int-storage-ptr)
  (src (:pointer :int)))
;; void THStorage_(copy)(THStorage *storage, THStorage *src);
(cffi:defcfun ("THIntStorage_copy" th-int-storage-copy) :void
  (storage th-int-storage-ptr)
  (src th-int-storage-ptr))
;; void THStorage_(copyByte)(THStorage *storage, struct THByteStorage *src);
(cffi:defcfun ("THIntStorage_copyByte" th-int-storage-copy-byte) :void
  (storage th-int-storage-ptr)
  (src th-byte-storage-ptr))
;; void THStorage_(copyChar)(THStorage *storage, struct THCharStorage *src);
(cffi:defcfun ("THIntStorage_copyChar" th-int-storage-copy-char) :void
  (storage th-int-storage-ptr)
  (src th-char-storage-ptr))
;; void THStorage_(copyShort)(THStorage *storage, struct THShortStorage *src);
(cffi:defcfun ("THIntStorage_copyShort" th-int-storage-copy-short) :void
  (storage th-int-storage-ptr)
  (src th-short-storage-ptr))
;; void THStorage_(copyInt)(THStorage *storage, struct THIntStorage *src);
(cffi:defcfun ("THIntStorage_copyInt" th-int-storage-copy-int) :void
  (storage th-int-storage-ptr)
  (src th-int-storage-ptr))
;; void THStorage_(copyLong)(THStorage *storage, struct THLongStorage *src);
(cffi:defcfun ("THIntStorage_copyLong" th-int-storage-copy-long) :void
  (storage th-int-storage-ptr)
  (src th-long-storage-ptr))
;; void THStorage_(copyFloat)(THStorage *storage, struct THFloatStorage *src);
(cffi:defcfun ("THIntStorage_copyFloat" th-int-storage-copy-float) :void
  (storage th-int-storage-ptr)
  (src th-float-storage-ptr))
;; void THStorage_(copyDouble)(THStorage *storage, struct THDoubleStorage *src);
(cffi:defcfun ("THIntStorage_copyDouble" th-int-storage-copy-double) :void
  (storage th-int-storage-ptr)
  (src th-double-storage-ptr))
