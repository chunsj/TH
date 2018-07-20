(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

;; ACCESS METHODS
;; THStorage* THTensor_(storage)(const THTensor *self)
(cffi:defcfun ("THIntTensor_storage" th-int-tensor-storage) th-int-storage-ptr
  (tensor th-int-tensor-ptr))
;; ptrdiff_t THTensor_(storageOffset)(const THTensor *self)
(cffi:defcfun ("THIntTensor_storageOffset" th-int-tensor-storage-offset) :long-long
  (tensor th-int-tensor-ptr))
;; int THTensor_(nDimension)(const THTensor *self)
(cffi:defcfun ("THIntTensor_nDimension" th-int-tensor-n-dimension) :int
  (tensor th-int-tensor-ptr))
;; long THTensor_(size)(const THTensor *self, int dim)
(cffi:defcfun ("THIntTensor_size" th-int-tensor-size) :long
  (tensor th-int-tensor-ptr)
  (dim :int))
;; long THTensor_(stride)(const THTensor *self, int dim)
(cffi:defcfun ("THIntTensor_stride" th-int-tensor-stride) :long
  (tensor th-int-tensor-ptr)
  (dim :int))
;; THLongStorage *THTensor_(newSizeOf)(THTensor *self)
(cffi:defcfun ("THIntTensor_newSizeOf" th-int-tensor-new-size-of) th-long-storage-ptr
  (tensor th-int-tensor-ptr))
;; THLongStorage *THTensor_(newStrideOf)(THTensor *self)
(cffi:defcfun ("THIntTensor_newStrideOf" th-int-tensor-new-stride-of) th-long-storage-ptr
  (tensor th-int-tensor-ptr))
;; real *THTensor_(data)(const THTensor *self)
(cffi:defcfun ("THIntTensor_data" th-int-tensor-data) (:pointer :int)
  (tensor th-int-tensor-ptr))

;; void THTensor_(setFlag)(THTensor *self, const char flag)
(cffi:defcfun ("THIntTensor_setFlag" th-int-tensor-set-flag) :void
  (tensor th-int-tensor-ptr)
  (flag :char))
;; void THTensor_(clearFlag)(THTensor *self, const char flag)
(cffi:defcfun ("THIntTensor_clearFlag" th-int-tensor-clear-flag) :void
  (tensor th-int-tensor-ptr)
  (flag :char))

;; CREATION METHODS
;; THTensor *THTensor_(new)(void)
(cffi:defcfun ("THIntTensor_new" th-int-tensor-new) th-int-tensor-ptr)
;; THTensor *THTensor_(newWithTensor)(THTensor *tensor)
(cffi:defcfun ("THIntTensor_newWithTensor" th-int-tensor-new-with-tensor) th-int-tensor-ptr
  (tensor th-int-tensor-ptr))
;; stride might be NULL
;; THTensor *THTensor_(newWithStorage)(THStorage *storage_, ptrdiff_t storageOffset_,
;;                                            THLongStorage *size_, THLongStorage *stride_)
(cffi:defcfun ("THIntTensor_newWithStorage" th-int-tensor-new-with-storage)
    th-int-tensor-ptr
  (storage th-int-storage-ptr)
  (storage-offset :long-long)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
;; THTensor *THTensor_(newWithStorage1d)(THStorage *storage_, ptrdiff_t storageOffset_,
;;                                              long size0_, long stride0_);
(cffi:defcfun ("THIntTensor_newWithStorage1d" th-int-tensor-new-with-storage-1d)
    th-int-tensor-ptr
  (storage th-int-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long))
(cffi:defcfun ("THIntTensor_newWithStorage2d" th-int-tensor-new-with-storage-2d)
    th-int-tensor-ptr
  (storage th-int-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long))
(cffi:defcfun ("THIntTensor_newWithStorage3d" th-int-tensor-new-with-storage-3d)
    th-int-tensor-ptr
  (storage th-int-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long)
  (size2 :long)
  (stride2 :long))
(cffi:defcfun ("THIntTensor_newWithStorage4d" th-int-tensor-new-with-storage-4d)
    th-int-tensor-ptr
  (storage th-int-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long)
  (size2 :long)
  (stride2 :long)
  (size3 :long)
  (stride3 :long))

;; stride might be NULL
;; THTensor *THTensor_(newWithSize)(THLongStorage *size_, THLongStorage *stride_)
(cffi:defcfun ("THIntTensor_newWithSize" th-int-tensor-new-with-size) th-int-tensor-ptr
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
;; THTensor *THTensor_(newWithSize1d)(long size0_);
(cffi:defcfun ("THIntTensor_newWithSize1d" th-int-tensor-new-with-size-1d)
    th-int-tensor-ptr
  (size0 :long))
(cffi:defcfun ("THIntTensor_newWithSize2d" th-int-tensor-new-with-size-2d)
    th-int-tensor-ptr
  (size0 :long)
  (size1 :long))
(cffi:defcfun ("THIntTensor_newWithSize3d" th-int-tensor-new-with-size-3d)
    th-int-tensor-ptr
  (size0 :long)
  (size1 :long)
  (size2 :long))
(cffi:defcfun ("THIntTensor_newWithSize4d" th-int-tensor-new-with-size-4d)
    th-int-tensor-ptr
  (size0 :long)
  (size1 :long)
  (size2 :long)
  (size3 :long))

;; THTensor *THTensor_(newClone)(THTensor *self)
(cffi:defcfun ("THIntTensor_newClone" th-int-tensor-new-clone) th-int-tensor-ptr
  (tensor th-int-tensor-ptr))
(cffi:defcfun ("THIntTensor_newContiguous" th-int-tensor-new-contiguous) th-int-tensor-ptr
  (tensor th-int-tensor-ptr))
(cffi:defcfun ("THIntTensor_newSelect" th-int-tensor-new-select) th-int-tensor-ptr
  (tensor th-int-tensor-ptr)
  (dimension :int)
  (slice-index :long))
(cffi:defcfun ("THIntTensor_newNarrow" th-int-tensor-new-narrow) th-int-tensor-ptr
  (tensor th-int-tensor-ptr)
  (dimension :int)
  (first-index :long)
  (size :long))

(cffi:defcfun ("THIntTensor_newTranspose" th-int-tensor-new-transpose) th-int-tensor-ptr
  (tensor th-int-tensor-ptr)
  (dimension1 :int)
  (dimension2 :int))
(cffi:defcfun ("THIntTensor_newUnfold" th-int-tensor-new-unfold) th-int-tensor-ptr
  (tensor th-int-tensor-ptr)
  (dimension :int)
  (size :long)
  (step :long))
(cffi:defcfun ("THIntTensor_newView" th-int-tensor-new-view) th-int-tensor-ptr
  (tensor th-int-tensor-ptr)
  (size th-long-storage-ptr))

(cffi:defcfun ("THIntTensor_expand" th-int-tensor-expand) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (size th-long-storage-ptr))

(cffi:defcfun ("THIntTensor_resize" th-int-tensor-resize) :void
  (tensor th-int-tensor-ptr)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
(cffi:defcfun ("THIntTensor_resizeAs" th-int-tensor-resize-as) :void
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr))
(cffi:defcfun ("THIntTensor_resizeNd" th-int-tensor-resize-nd) :void
  (tensor th-int-tensor-ptr)
  (dimension :int)
  (size (:pointer :long))
  (stride (:pointer :long)))
(cffi:defcfun ("THIntTensor_resize1d" th-int-tensor-resize-1d) :void
  (tensor th-int-tensor-ptr)
  (size0 :long))
(cffi:defcfun ("THIntTensor_resize2d" th-int-tensor-resize-2d) :void
  (tensor th-int-tensor-ptr)
  (size0 :long)
  (size1 :long))
(cffi:defcfun ("THIntTensor_resize3d" th-int-tensor-resize-3d) :void
  (tensor th-int-tensor-ptr)
  (size0 :long)
  (size1 :long)
  (size2 :long))
(cffi:defcfun ("THIntTensor_resize4d" th-int-tensor-resize-4d) :void
  (tensor th-int-tensor-ptr)
  (size0 :long)
  (size1 :long)
  (size2 :long)
  (size3 :long))
(cffi:defcfun ("THIntTensor_resize5d" th-int-tensor-resize-5d) :void
  (tensor th-int-tensor-ptr)
  (size0 :long)
  (size1 :long)
  (size2 :long)
  (size3 :long)
  (size4 :long))

(cffi:defcfun ("THIntTensor_set" th-int-tensor-set) :void
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr))
(cffi:defcfun ("THIntTensor_setStorage" th-int-tensor-set-storage) :void
  (tensor th-int-tensor-ptr)
  (storage th-int-storage-ptr)
  (storage-offset :long-long)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
(cffi:defcfun ("THIntTensor_setStorageNd" th-int-tensor-set-storage-nd) :void
  (tensor th-int-tensor-ptr)
  (storage th-int-storage-ptr)
  (storage-offset :long-long)
  (dimension :int)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
(cffi:defcfun ("THIntTensor_setStorage1d" th-int-tensor-set-storage-1d) :void
  (tensor th-int-tensor-ptr)
  (storage th-int-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long))
(cffi:defcfun ("THIntTensor_setStorage2d" th-int-tensor-set-storage-2d) :void
  (tensor th-int-tensor-ptr)
  (storage th-int-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long))
(cffi:defcfun ("THIntTensor_setStorage3d" th-int-tensor-set-storage-3d) :void
  (tensor th-int-tensor-ptr)
  (storage th-int-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long)
  (size2 :long)
  (stride2 :long))
(cffi:defcfun ("THIntTensor_setStorage4d" th-int-tensor-set-storage-4d) :void
  (tensor th-int-tensor-ptr)
  (storage th-int-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long)
  (size2 :long)
  (stride2 :long)
  (size3 :long)
  (stride3 :long))

(cffi:defcfun ("THIntTensor_narrow" th-int-tensor-narrow) :void
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr)
  (dimension :int)
  (first-index :long)
  (size :long))
(cffi:defcfun ("THIntTensor_select" th-int-tensor-select) :void
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr)
  (dimension :int)
  (slice-index :long))
(cffi:defcfun ("THIntTensor_transpose" th-int-tensor-transpose) :void
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr)
  (dimension1 :int)
  (dimension2 :int))
(cffi:defcfun ("THIntTensor_unfold" th-int-tensor-unfold) :void
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr)
  (dimension :int)
  (size :long)
  (step :long))
(cffi:defcfun ("THIntTensor_squeeze" th-int-tensor-squeeze) :void
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr))
(cffi:defcfun ("THIntTensor_squeeze1d" th-int-tensor-squeeze-1d) :void
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr)
  (dimension :int))
(cffi:defcfun ("THIntTensor_unsqueeze1d" th-int-tensor-unsqueeze-1d) :void
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr)
  (dimension :int))

(cffi:defcfun ("THIntTensor_isContiguous" th-int-tensor-is-contiguous) :int
  (tensor th-int-tensor-ptr))
(cffi:defcfun ("THIntTensor_isSameSizeAs" th-int-tensor-is-same-size-as) :int
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr))
(cffi:defcfun ("THIntTensor_isSetTo" th-int-tensor-is-set-to) :int
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr))
(cffi:defcfun ("THIntTensor_isSize" th-int-tensor-is-size) :int
  (tensor th-int-tensor-ptr)
  (dims th-long-storage-ptr))
(cffi:defcfun ("THIntTensor_nElement" th-int-tensor-n-element) :long-long
  (tensor th-int-tensor-ptr))

(cffi:defcfun ("THIntTensor_retain" th-int-tensor-retain) :void
  (tensor th-int-tensor-ptr))
(cffi:defcfun ("THIntTensor_free" th-int-tensor-free) :void
  (tensor th-int-tensor-ptr))
(cffi:defcfun ("THIntTensor_freeCopyTo" th-int-tensor-free-copy-to) :void
  (source th-int-tensor-ptr)
  (target th-int-tensor-ptr))

;; slow access methods [check everything]
;; void THTensor_(set1d)(THTensor *tensor, long x0, real value);
(cffi:defcfun ("THIntTensor_set1d" th-int-tensor-set-1d) :void
  (tensor th-int-tensor-ptr)
  (index0 :long)
  (value :int))
;; void THTensor_(set2d)(THTensor *tensor, long x0, long x1, real value);
(cffi:defcfun ("THIntTensor_set2d" th-int-tensor-set-2d) :void
  (tensor th-int-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (value :int))
;; void THTensor_(set3d)(THTensor *tensor, long x0, long x1, long x2, real value);
(cffi:defcfun ("THIntTensor_set3d" th-int-tensor-set-3d) :void
  (tensor th-int-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long)
  (value :int))
;; void THTensor_(set4d)(THTensor *tensor, long x0, long x1, long x2, long x3, real value);
(cffi:defcfun ("THIntTensor_set4d" th-int-tensor-set-4d) :void
  (tensor th-int-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long)
  (index3 :long)
  (value :int))

(cffi:defcfun ("THIntTensor_get1d" th-int-tensor-get-1d) :int
  (tensor th-int-tensor-ptr)
  (index0 :long))
(cffi:defcfun ("THIntTensor_get2d" th-int-tensor-get-2d) :int
  (tensor th-int-tensor-ptr)
  (index0 :long)
  (index1 :long))
(cffi:defcfun ("THIntTensor_get3d" th-int-tensor-get-3d) :int
  (tensor th-int-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long))
(cffi:defcfun ("THIntTensor_get4d" th-int-tensor-get-4d) :int
  (tensor th-int-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long)
  (index3 :long))

;; support for copy betweeb different tensor types
;; void THTensor_(copy)(THTensor *tensor, THTensor *src);
(cffi:defcfun ("THIntTensor_copy" th-int-tensor-copy) :void
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(copyByte)(THTensor *tensor, struct THByteTensor *src);
(cffi:defcfun ("THIntTensor_copyByte" th-int-tensor-copy-byte) :void
  (tensor th-int-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(copyChar)(THTensor *tensor, struct THCharTensor *src);
(cffi:defcfun ("THIntTensor_copyChar" th-int-tensor-copy-char) :void
  (tensor th-int-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(copyShort)(THTensor *tensor, struct THShortTensor *src);
(cffi:defcfun ("THIntTensor_copyShort" th-int-tensor-copy-short) :void
  (tensor th-int-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(copyInt)(THTensor *tensor, struct THIntTensor *src);
(cffi:defcfun ("THIntTensor_copyInt" th-int-tensor-copy-int) :void
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(copyLong)(THTensor *tensor, struct THLongTensor *src);
(cffi:defcfun ("THIntTensor_copyLong" th-int-tensor-copy-long) :void
  (tensor th-int-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(copyFloat)(THTensor *tensor, struct THFloatTensor *src);
(cffi:defcfun ("THIntTensor_copyFloat" th-int-tensor-copy-float) :void
  (tensor th-int-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(copyDouble)(THTensor *tensor, struct THDoubleTensor *src);
(cffi:defcfun ("THIntTensor_copyDouble" th-int-tensor-copy-double) :void
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr))

;; fast method to access to tensor data - how to implement this in lisp? XXX
;; #define THTensor_fastGet1d(self, x0)                                    \
;; (((self)->storage->data+(self)->storageOffset)[(x0)*(self)->stride[0]])

;; #define THTensor_fastGet2d(self, x0, x1)                                \
;; (((self)->storage->data+(self)->storageOffset)[(x0)*(self)->stride[0]+(x1)*(self)->stride[1]])

;; #define THTensor_fastGet3d(self, x0, x1, x2)                            \
;; (((self)->storage->data+(self)->storageOffset)[(x0)*(self)->stride[0]+(x1)*(self)->stride[1]+(x2)*(self)->stride[2]])

;; #define THTensor_fastGet4d(self, x0, x1, x2, x3)                        \
;; (((self)->storage->data+(self)->storageOffset)[(x0)*(self)->stride[0]+(x1)*(self)->stride[1]+(x2)*(self)->stride[2]+(x3)*(self)->stride[3]])

;; #define THTensor_fastSet1d(self, x0, value)                             \
;; (((self)->storage->data+(self)->storageOffset)[(x0)*(self)->stride[0]] = value)

;; #define THTensor_fastSet2d(self, x0, x1, value)                         \
;; (((self)->storage->data+(self)->storageOffset)[(x0)*(self)->stride[0]+(x1)*(self)->stride[1]] = value)

;; #define THTensor_fastSet3d(self, x0, x1, x2, value)                     \
;; (((self)->storage->data+(self)->storageOffset)[(x0)*(self)->stride[0]+(x1)*(self)->stride[1]+(x2)*(self)->stride[2]] = value)

;; #define THTensor_fastSet4d(self, x0, x1, x2, x3, value)                 \
;; (((self)->storage->data+(self)->storageOffset)[(x0)*(self)->stride[0]+(x1)*(self)->stride[1]+(x2)*(self)->stride[2]+(x3)*(self)->stride[3]] = value)

;; RANDOM
;; void THTensor_(random)(THTensor *self, THGenerator *_generator);
(cffi:defcfun ("THIntTensor_random" th-int-tensor-random) :void
  (tensor th-int-tensor-ptr)
  (generator th-generator-ptr))
;; void THTensor_(clampedRandom)(THTensor *self, THGenerator *_generator, long min, long max)
(cffi:defcfun ("THIntTensor_clampedRandom" th-int-tensor-clamped-random) :void
  (tensor th-int-tensor-ptr)
  (genrator th-generator-ptr)
  (min :long)
  (max :long))
;; void THTensor_(cappedRandom)(THTensor *self, THGenerator *_generator, long max);
(cffi:defcfun ("THIntTensor_cappedRandom" th-int-tensor-capped-random) :void
  (tensor th-int-tensor-ptr)
  (generator th-generator-ptr)
  (max :long))
;; void THTensor_(geometric)(THTensor *self, THGenerator *_generator, double p);
(cffi:defcfun ("THIntTensor_geometric" th-int-tensor-geometric) :void
  (tensor th-int-tensor-ptr)
  (generator th-generator-ptr)
  (p :double))
;; void THTensor_(bernoulli)(THTensor *self, THGenerator *_generator, double p);
(cffi:defcfun ("THIntTensor_bernoulli" th-int-tensor-bernoulli) :void
  (tensor th-int-tensor-ptr)
  (generator th-generator-ptr)
  (p :double))
;; void THTensor_(bernoulli_FloatTensor)(THTensor *self, THGenerator *_generator, THFloatTensor *p);
(cffi:defcfun ("THIntTensor_bernoulli_FloatTensor" th-int-tensor-bernoulli-float-tensor) :void
  (tensor th-int-tensor-ptr)
  (generator th-generator-ptr)
  (p th-float-tensor-ptr))
;; void THTensor_(bernoulli_DoubleTensor)(THTensor *self, THGenerator *_generator, THDoubleTensor *p);
(cffi:defcfun ("THIntTensor_bernoulli_DoubleTensor" th-int-tensor-bernoulli-double-tensor)
    :void
  (tensor th-int-tensor-ptr)
  (generator th-generator-ptr)
  (p th-double-tensor-ptr))

;; #if defined(TH_REAL_IS_BYTE)
;; void THTensor_(getRNGState)(THGenerator *_generator, THTensor *self);
;; void THTensor_(setRNGState)(THGenerator *_generator, THTensor *self);
;; #endif

;; void THTensor_(fill)(THTensor *r_, real value);
(cffi:defcfun ("THIntTensor_fill" th-int-tensor-fill) :void
  (tensor th-int-tensor-ptr)
  (value :int))
;; void THTensor_(zero)(THTensor *r_);
(cffi:defcfun ("THIntTensor_zero" th-int-tensor-zero) :void
  (tensor th-int-tensor-ptr))

;; void THTensor_(maskedFill)(THTensor *tensor, THByteTensor *mask, real value);
(cffi:defcfun ("THIntTensor_maskedFill" th-int-tensor-masked-fill) :void
  (tensor th-int-tensor-ptr)
  (mask th-byte-tensor-ptr)
  (value :int))
;; void THTensor_(maskedCopy)(THTensor *tensor, THByteTensor *mask, THTensor* src);
(cffi:defcfun ("THIntTensor_maskedCopy" th-int-tensor-masked-copy) :void
  (tensor th-int-tensor-ptr)
  (mask th-byte-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(maskedSelect)(THTensor *tensor, THTensor* src, THByteTensor *mask);
(cffi:defcfun ("THIntTensor_maskedSelect" th-int-tensor-masked-select) :void
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr)
  (mask (th-byte-tensor-ptr)))

;; void THTensor_(nonzero)(THLongTensor *subscript, THTensor *tensor);
(cffi:defcfun ("THIntTensor_nonzero" th-int-tensor-nonzero) :void
  (subscript th-long-tensor-ptr)
  (tensor th-int-tensor-ptr))

;; void THTensor_(indexSelect)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index)
(cffi:defcfun ("THIntTensor_indexSelect" th-int-tensor-index-select) :void
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr))
;; void THTensor_(indexCopy)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THIntTensor_indexCopy" th-int-tensor-index-copy) :void
  (tensor th-int-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(indexAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THIntTensor_indexAdd" th-int-tensor-index-add) :void
  (tensor th-int-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(indexFill)(THTensor *tensor, int dim, THLongTensor *index, real val);
(cffi:defcfun ("THIntTensor_indexFill" th-int-tensor-index-fill) :void
  (tensor th-int-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (value :int))

;; void THTensor_(gather)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index);
(cffi:defcfun ("THIntTensor_gather" th-int-tensor-gather) :void
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr))
;; void THTensor_(scatter)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THIntTensor_scatter" th-int-tensor-scatter) :void
  (tensor th-int-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(scatterAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THIntTensor_scatterAdd" th-int-tensor-scatter-add) :void
  (tensor th-int-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(scatterFill)(THTensor *tensor, int dim, THLongTensor *index, real val);
(cffi:defcfun ("THIntTensor_scatterFill" th-int-tensor-scatter-fill) :void
  (tensor th-int-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (value :int))

;; accreal THTensor_(dot)(THTensor *t, THTensor *src);
(cffi:defcfun ("THIntTensor_dot" th-int-tensor-dot) :long
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr))

;; real THTensor_(minall)(THTensor *t);
(cffi:defcfun ("THIntTensor_minall" th-int-tensor-min-all) :int
  (tensor th-int-tensor-ptr))
;; real THTensor_(maxall)(THTensor *t);
(cffi:defcfun ("THIntTensor_maxall" th-int-tensor-max-all) :int
  (tensor th-int-tensor-ptr))
;; real THTensor_(medianall)(THTensor *t);
(cffi:defcfun ("THIntTensor_medianall" th-int-tensor-median-all) :int
  (tensor th-int-tensor-ptr))
;; accreal THTensor_(sumall)(THTensor *t);
(cffi:defcfun ("THIntTensor_sumall" th-int-tensor-sum-all) :long
  (tensor th-int-tensor-ptr))
;; accreal THTensor_(prodall)(THTensor *t);
(cffi:defcfun ("THIntTensor_prodall" th-int-tensor-prod-all) :long
  (tensor th-int-tensor-ptr))

;; void THTensor_(neg)(THTensor *self, THTensor *src);
(cffi:defcfun ("THIntTensor_neg" th-int-tensor-neg) :void
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr))

;; void THTensor_(add)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THIntTensor_add" th-int-tensor-add) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))
;; void THTensor_(sub)(THTensor *self, THTensor *src, real value);
(cffi:defcfun ("THIntTensor_sub" th-int-tensor-sub) :void
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr)
  (value :int))
;; void THTensor_(mul)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THIntTensor_mul" th-int-tensor-mul) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))
;; void THTensor_(div)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THIntTensor_div" th-int-tensor-div) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))
;; void THTensor_(lshift)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THIntTensor_lshift" th-int-tensor-lshift) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))
;; void THTensor_(rshift)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THIntTensor_rshift" th-int-tensor-rshift) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))
;; void THTensor_(fmod)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THIntTensor_fmod" th-int-tensor-fmod) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))
;; void THTensor_(remainder)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THIntTensor_remainder" th-int-tensor-remainder) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))
;; void THTensor_(clamp)(THTensor *r_, THTensor *t, real min_value, real max_value);
(cffi:defcfun ("THIntTensor_clamp" th-int-tensor-clamp) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (min-value :int)
  (max-value :int))
;; void THTensor_(bitand)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THIntTensor_bitand" th-int-tensor-bitand) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))
;; void THTensor_(bitor)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THIntTensor_bitor" th-int-tensor-bitor) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))
;; void THTensor_(bitxor)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THIntTensor_bitxor" th-int-tensor-bitxor) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))

;; void THTensor_(cadd)(THTensor *r_, THTensor *t, real value, THTensor *src);
(cffi:defcfun ("THIntTensor_cadd" th-int-tensor-cadd) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int)
  (src th-int-tensor-ptr))
;; void THTensor_(csub)(THTensor *self, THTensor *src1, real value, THTensor *src2);
(cffi:defcfun ("THIntTensor_csub" th-int-tensor-csub) :void
  (tensor th-int-tensor-ptr)
  (src1 th-int-tensor-ptr)
  (value :int)
  (src2 th-int-tensor-ptr))
;; void THTensor_(cmul)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THIntTensor_cmul" th-int-tensor-cmul) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(cpow)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THIntTensor_cpow" th-int-tensor-cpow) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(cdiv)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THIntTensor_cdiv" th-int-tensor-cdiv) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(clshift)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THIntTensor_clshift" th-int-tensor-clshift) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(crshift)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THIntTensor_crshift" th-int-tensor-crshift) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(cfmod)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THIntTensor_cfmod" th-int-tensor-cfmod) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(cremainder)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THIntTensor_cremainder" th-int-tensor-cremainder) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(cbitand)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THIntTensor_cbitand" th-int-tensor-cbitand) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(cbitor)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THIntTensor_cbitor" th-int-tensor-cbitor) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(cbitxor)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THIntTensor_cbitxor" th-int-tensor-cbitxor) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr))

;; void THTensor_(addcmul)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2);
(cffi:defcfun ("THIntTensor_addcmul" th-int-tensor-add-cmul) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int)
  (src1 th-int-tensor-ptr)
  (src2 th-int-tensor-ptr))
;; void THTensor_(addcdiv)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2);
(cffi:defcfun ("THIntTensor_addcdiv" th-int-tensor-add-cdiv) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int)
  (src1 th-int-tensor-ptr)
  (src2 th-int-tensor-ptr))
;; void THTensor_(addmv)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat,  THTensor *vec);
(cffi:defcfun ("THIntTensor_addmv" th-int-tensor-add-mv) :void
  (result th-int-tensor-ptr)
  (beta :int)
  (tensor th-int-tensor-ptr)
  (alpha :int)
  (matrix th-int-tensor-ptr)
  (vector th-int-tensor-ptr))
;; void THTensor_(addmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat1, THTensor *mat2);
(cffi:defcfun ("THIntTensor_addmm" th-int-tensor-add-mm) :void
  (result th-int-tensor-ptr)
  (beta :int)
  (tensor th-int-tensor-ptr)
  (alpha :int)
  (matrix1 th-int-tensor-ptr)
  (matrix2 th-int-tensor-ptr))
;; void THTensor_(addr)(THTensor *r_,  real beta, THTensor *t, real alpha, THTensor *vec1, THTensor *vec2);
(cffi:defcfun ("THIntTensor_addr" th-int-tensor-add-r) :void
  (result th-int-tensor-ptr)
  (beta :int)
  (tensor th-int-tensor-ptr)
  (alpha :int)
  (vector1 th-int-tensor-ptr)
  (vector2 th-int-tensor-ptr))
;; void THTensor_(addbmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *batch1, THTensor *batch2);
(cffi:defcfun ("THIntTensor_addbmm" th-int-tensor-add-bmm) :void
  (result th-int-tensor-ptr)
  (beta :int)
  (tensor th-int-tensor-ptr)
  (alpha :int)
  (batch1 th-int-tensor-ptr)
  (batch2 th-int-tensor-ptr))
;; void THTensor_(baddbmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *batch1, THTensor *batch2);
(cffi:defcfun ("THIntTensor_baddbmm" th-int-tensor-badd-bmm) :void
  (result th-int-tensor-ptr)
  (beta :int)
  (tensor th-int-tensor-ptr)
  (alpha :int)
  (batch1 th-int-tensor-ptr)
  (batch2 th-int-tensor-ptr))

;; void THTensor_(match)(THTensor *r_, THTensor *m1, THTensor *m2, real gain);
(cffi:defcfun ("THIntTensor_match" th-int-tensor-match) :void
  (result th-int-tensor-ptr)
  (m1 th-int-tensor-ptr)
  (m2 th-int-tensor-ptr)
  (gain :int))

;; ptrdiff_t THTensor_(numel)(THTensor *t);
(cffi:defcfun ("THIntTensor_numel" th-int-tensor-numel) :long-long
  (tensor th-int-tensor-ptr))
;; void THTensor_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THIntTensor_max" th-int-tensor-max) :void
  (values th-int-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THIntTensor_min" th-int-tensor-min) :void
  (values th-int-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(kthvalue)(THTensor *values_, THLongTensor *indices_, THTensor *t, long k, int dimension, int keepdim);
(cffi:defcfun ("THIntTensor_kthvalue" th-int-tensor-kth-value) :void
  (values th-int-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (k :long)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(mode)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THIntTensor_mode" th-int-tensor-mode) :void
  (values th-int-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(median)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THIntTensor_median" th-int-tensor-median) :void
  (values th-int-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(sum)(THTensor *r_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THIntTensor_sum" th-int-tensor-sum) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(prod)(THTensor *r_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THIntTensor_prod" th-int-tensor-prod) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(cumsum)(THTensor *r_, THTensor *t, int dimension);
(cffi:defcfun ("THIntTensor_cumsum" th-int-tensor-cum-sum) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (dimension :int))
;; void THTensor_(cumprod)(THTensor *r_, THTensor *t, int dimension);
(cffi:defcfun ("THIntTensor_cumprod" th-int-tensor-cum-prod) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (dimension :int))
;; void THTensor_(sign)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THIntTensor_sign" th-int-tensor-sign) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr))
;; accreal THTensor_(trace)(THTensor *t);
(cffi:defcfun ("THIntTensor_trace" th-int-tensor-trace) :long
  (tensor th-int-tensor-ptr))
;; void THTensor_(cross)(THTensor *r_, THTensor *a, THTensor *b, int dimension);
(cffi:defcfun ("THIntTensor_cross" th-int-tensor-cross) :void
  (result th-int-tensor-ptr)
  (a th-int-tensor-ptr)
  (b th-int-tensor-ptr)
  (dimension :int))

;; void THTensor_(cmax)(THTensor *r, THTensor *t, THTensor *src);
(cffi:defcfun ("THIntTensor_cmax" th-int-tensor-cmax) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(cmin)(THTensor *r, THTensor *t, THTensor *src);
(cffi:defcfun ("THIntTensor_cmin" th-int-tensor-cmin) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(cmaxValue)(THTensor *r, THTensor *t, real value);
(cffi:defcfun ("THIntTensor_cmaxValue" th-int-tensor-cmax-value) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))
;; void THTensor_(cminValue)(THTensor *r, THTensor *t, real value);
(cffi:defcfun ("THIntTensor_cminValue" th-int-tensor-cmin-value) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))

;; void THTensor_(zeros)(THTensor *r_, THLongStorage *size);
(cffi:defcfun ("THIntTensor_zeros" th-int-tensor-zeros) :void
  (result th-int-tensor-ptr)
  (size th-long-storage-ptr))
;; void THTensor_(zerosLike)(THTensor *r_, THTensor *input);
(cffi:defcfun ("THIntTensor_zerosLike" th-int-tensor-zero-like) :void
  (result th-int-tensor-ptr)
  (input th-int-tensor-ptr))
;; void THTensor_(ones)(THTensor *r_, THLongStorage *size);
(cffi:defcfun ("THIntTensor_ones" th-int-tensor-ones) :void
  (result th-int-tensor-ptr)
  (size th-long-storage-ptr))
;; void THTensor_(onesLike)(THTensor *r_, THTensor *input);
(cffi:defcfun ("THIntTensor_onesLike" th-int-tensor-one-like) :void
  (result th-int-tensor-ptr)
  (input th-int-tensor-ptr))
;; void THTensor_(diag)(THTensor *r_, THTensor *t, int k);
(cffi:defcfun ("THIntTensor_diag" th-int-tensor-diag) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (k :int))
;; void THTensor_(eye)(THTensor *r_, long n, long m);
(cffi:defcfun ("THIntTensor_eye" th-int-tensor-eye) :void
  (result th-int-tensor-ptr)
  (n :long)
  (m :long))
;; void THTensor_(arange)(THTensor *r_, accreal xmin, accreal xmax, accreal step);
(cffi:defcfun ("THIntTensor_arange" th-int-tensor-arange) :void
  (result th-int-tensor-ptr)
  (xmin :long)
  (xmax :long)
  (step :long))
;; void THTensor_(range)(THTensor *r_, accreal xmin, accreal xmax, accreal step);
(cffi:defcfun ("THIntTensor_range" th-int-tensor-range) :void
  (result th-int-tensor-ptr)
  (xmin :long)
  (xmax :long)
  (step :long))
;; void THTensor_(randperm)(THTensor *r_, THGenerator *_generator, long n);
(cffi:defcfun ("THIntTensor_randperm" th-int-tensor-rand-perm) :void
  (result th-int-tensor-ptr)
  (generator th-generator-ptr)
  (n :long))

;; void THTensor_(reshape)(THTensor *r_, THTensor *t, THLongStorage *size);
(cffi:defcfun ("THIntTensor_reshape" th-int-tensor-reshape) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (size th-long-storage-ptr))
;; void THTensor_(sort)(THTensor *rt_, THLongTensor *ri_, THTensor *t, int dimension, int descendingOrder);
(cffi:defcfun ("THIntTensor_sort" th-int-tensor-sort) :void
  (result-tensor th-int-tensor-ptr)
  (result-indices th-long-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (dimension :int)
  (discending-order :int))
;; void THTensor_(topk)(THTensor *rt_, THLongTensor *ri_, THTensor *t, long k, int dim, int dir, int sorted);
(cffi:defcfun ("THIntTensor_topk" th-int-tensor-topk) :void
  (result-tensor th-int-tensor-ptr)
  (result-indices th-long-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (k :long)
  (dim :int)
  (dir :int)
  (sorted :int))
;; void THTensor_(tril)(THTensor *r_, THTensor *t, long k);
(cffi:defcfun ("THIntTensor_tril" th-int-tensor-tril) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (k :long))
;; void THTensor_(triu)(THTensor *r_, THTensor *t, long k);
(cffi:defcfun ("THIntTensor_triu" th-int-tensor-triu) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (k :long))
;; void THTensor_(cat)(THTensor *r_, THTensor *ta, THTensor *tb, int dimension);
(cffi:defcfun ("THIntTensor_cat" th-int-tensor-cat) :void
  (result th-int-tensor-ptr)
  (tensora th-int-tensor-ptr)
  (tensorb th-int-tensor-ptr)
  (dimension :int))
;; void THTensor_(catArray)(THTensor *result, THTensor **inputs, int numInputs, int dimension);
(cffi:defcfun ("THIntTensor_catArray" th-int-tensor-cat-array) :void
  (result th-int-tensor-ptr)
  (inputs (:pointer th-int-tensor-ptr))
  (num-inputs :int)
  (dimension :int))

;; int THTensor_(equal)(THTensor *ta, THTensor *tb);
(cffi:defcfun ("THIntTensor_equal" th-int-tensor-equal) :int
  (tensora th-int-tensor-ptr)
  (tensorb th-int-tensor-ptr))

;; void THTensor_(ltValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THIntTensor_ltValue" th-int-tensor-lt-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))
;; void THTensor_(leValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THIntTensor_leValue" th-int-tensor-le-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))
;; void THTensor_(gtValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THIntTensor_gtValue" th-int-tensor-gt-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))
;; void THTensor_(geValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THIntTensor_geValue" th-int-tensor-ge-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))
;; void THTensor_(neValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THIntTensor_neValue" th-int-tensor-ne-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))
;; void THTensor_(eqValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THIntTensor_eqValue" th-int-tensor-eq-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))

;; void THTensor_(ltValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THIntTensor_ltValueT" th-int-tensor-lt-value-t) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))
;; void THTensor_(leValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THIntTensor_leValueT" th-int-tensor-le-value-t) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))
;; void THTensor_(gtValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THIntTensor_gtValueT" th-int-tensor-gt-value-t) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))
;; void THTensor_(geValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THIntTensor_geValueT" th-int-tensor-ge-value-t) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))
;; void THTensor_(neValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THIntTensor_neValueT" th-int-tensor-ne-value-t) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))
;; void THTensor_(eqValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THIntTensor_eqValueT" th-int-tensor-eq-value-t) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr)
  (value :int))

;; void THTensor_(ltTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THIntTensor_ltTensor" th-int-tensor-lt-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-int-tensor-ptr)
  (tensorb th-int-tensor-ptr))
;; void THTensor_(leTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THIntTensor_leTensor" th-int-tensor-le-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-int-tensor-ptr)
  (tensorb th-int-tensor-ptr))
;; void THTensor_(gtTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THIntTensor_gtTensor" th-int-tensor-gt-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-int-tensor-ptr)
  (tensorb th-int-tensor-ptr))
;; void THTensor_(geTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THIntTensor_geTensor" th-int-tensor-ge-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-int-tensor-ptr)
  (tensorb th-int-tensor-ptr))
;; void THTensor_(neTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THIntTensor_neTensor" th-int-tensor-ne-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-int-tensor-ptr)
  (tensorb th-int-tensor-ptr))
;; void THTensor_(eqTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THIntTensor_eqTensor" th-int-tensor-eq-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-int-tensor-ptr)
  (tensorb th-int-tensor-ptr))

;; void THTensor_(ltTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THIntTensor_ltTensorT" th-int-tensor-lt-tensor-t) :void
  (result th-int-tensor-ptr)
  (tensora th-int-tensor-ptr)
  (tensorb th-int-tensor-ptr))
;; void THTensor_(leTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THIntTensor_leTensorT" th-int-tensor-le-tensor-t) :void
  (result th-int-tensor-ptr)
  (tensora th-int-tensor-ptr)
  (tensorb th-int-tensor-ptr))
;; void THTensor_(gtTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THIntTensor_gtTensorT" th-int-tensor-gt-tensor-t) :void
  (result th-int-tensor-ptr)
  (tensora th-int-tensor-ptr)
  (tensorb th-int-tensor-ptr))
;; void THTensor_(geTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THIntTensor_geTensorT" th-int-tensor-ge-tensor-t) :void
  (result th-int-tensor-ptr)
  (tensora th-int-tensor-ptr)
  (tensorb th-int-tensor-ptr))
;; void THTensor_(neTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THIntTensor_neTensorT" th-int-tensor-ne-tensor-t) :void
  (result th-int-tensor-ptr)
  (tensora th-int-tensor-ptr)
  (tensorb th-int-tensor-ptr))
;; void THTensor_(eqTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THIntTensor_eqTensorT" th-int-tensor-eq-tensor-t) :void
  (result th-int-tensor-ptr)
  (tensora th-int-tensor-ptr)
  (tensorb th-int-tensor-ptr))

;; void THTensor_(abs)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THIntTensor_abs" th-int-tensor-abs) :void
  (result th-int-tensor-ptr)
  (tensor th-int-tensor-ptr))

;; #if defined(TH_REAL_IS_BYTE)

;; int THTensor_(logicalall)(THTensor *self);
;; (cffi:defcfun ("THIntTensor_logicalall" th-int-tensor-logical-all) :int
;;   (tensor (:pointer :void)))
;; int THTensor_(logicalany)(THTensor *self);
;; (cffi:defcfun ("THIntTensor_logicalany" th-int-tensor-logical-any) :int
;;   (tensor (:pointer :void)))

;; #endif /* TH_REAL_IS_BYTE */


;; void THTensor_(validXCorr2Dptr)(real *r_,
;;                                 real alpha,
;;                                 real *t_, long ir, long ic,
;;                                 real *k_, long kr, long kc,
;;                                 long sr, long sc);
(cffi:defcfun ("THIntTensor_validXCorr2Dptr" th-int-tensor-valid-x-corr-2d-ptr) :void
  (res (:pointer :int))
  (alpha :int)
  (ten (:pointer :int))
  (ir :long)
  (ic :long)
  (k (:pointer :int))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(validConv2Dptr)(real *r_,
;;                                real alpha,
;;                                real *t_, long ir, long ic,
;;                                real *k_, long kr, long kc,
;;                                long sr, long sc);
(cffi:defcfun ("THIntTensor_validConv2Dptr" th-int-tensor-valid-conv-2d-ptr) :void
  (res (:pointer :int))
  (alpha :int)
  (ten (:pointer :int))
  (ir :long)
  (ic :long)
  (k (:pointer :int))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(fullXCorr2Dptr)(real *r_,
;;                                real alpha,
;;                                real *t_, long ir, long ic,
;;                                real *k_, long kr, long kc,
;;                                long sr, long sc);
(cffi:defcfun ("THIntTensor_fullXCorr2Dptr" th-int-tensor-full-x-corr-2d-ptr) :void
  (res (:pointer :int))
  (alpha :int)
  (ten (:pointer :int))
  (ir :long)
  (ic :long)
  (k (:pointer :int))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(fullConv2Dptr)(real *r_,
;;                               real alpha,
;;                               real *t_, long ir, long ic,
;;                               real *k_, long kr, long kc,
;;                               long sr, long sc);
(cffi:defcfun ("THIntTensor_fullConv2Dptr" th-int-tensor-full-conv-2d-ptr) :void
  (res (:pointer :int))
  (alpha :int)
  (ten (:pointer :int))
  (ir :long)
  (ic :long)
  (k (:pointer :int))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(validXCorr2DRevptr)(real *r_,
;;                                    real alpha,
;;                                    real *t_, long ir, long ic,
;;                                    real *k_, long kr, long kc,
;;                                    long sr, long sc);
(cffi:defcfun ("THIntTensor_validXCorr2DRevptr" th-int-tensor-valid-x-corr-2d-rev-ptr) :void
  (res (:pointer :int))
  (alpha :int)
  (ten (:pointer :int))
  (ir :long)
  (ic :long)
  (k (:pointer :int))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(conv2DRevger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol);
(cffi:defcfun ("THIntTensor_conv2DRevger" th-int-tensor-conv-2d-rev-ger) :void
  (result th-int-tensor-ptr)
  (beta :int)
  (alpha :int)
  (tensor th-int-tensor-ptr)
  (k th-int-tensor-ptr)
  (srow :long)
  (scol :long))
;; void THTensor_(conv2DRevgerm)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol);
(cffi:defcfun ("THIntTensor_conv2DRevgerm" th-int-tensor-conv-2d-rev-germ) :void
  (result th-int-tensor-ptr)
  (beta :int)
  (alpha :int)
  (tensor th-int-tensor-ptr)
  (k th-int-tensor-ptr)
  (srow :long)
  (scol :long))
;; void THTensor_(conv2Dger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THIntTensor_conv2Dger" th-int-tensor-conv-2d-ger) :void
  (result th-int-tensor-ptr)
  (beta :int)
  (alpha :int)
  (tensor th-int-tensor-ptr)
  (k th-int-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv2Dmv)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THIntTensor_conv2Dmv" th-int-tensor-conv-2d-mv) :void
  (result th-int-tensor-ptr)
  (beta :int)
  (alpha :int)
  (tensor th-int-tensor-ptr)
  (k th-int-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv2Dmm)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THIntTensor_conv2Dmm" th-int-tensor-conv-2d-mm) :void
  (result th-int-tensor-ptr)
  (beta :int)
  (alpha :int)
  (tensor th-int-tensor-ptr)
  (k th-int-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv2Dmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THIntTensor_conv2Dmul" th-int-tensor-conv-2d-mul) :void
  (result th-int-tensor-ptr)
  (beta :int)
  (alpha :int)
  (tensor th-int-tensor-ptr)
  (k th-int-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv2Dcmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THIntTensor_conv2Dcmul" th-int-tensor-conv-2d-cmul) :void
  (result th-int-tensor-ptr)
  (beta :int)
  (alpha :int)
  (tensor th-int-tensor-ptr)
  (k th-int-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))

;; void THTensor_(validXCorr3Dptr)(real *r_,
;;                                 real alpha,
;;                                 real *t_, long it, long ir, long ic,
;;                                 real *k_, long kt, long kr, long kc,
;;                                 long st, long sr, long sc);
(cffi:defcfun ("THIntTensor_validXCorr3Dptr" th-int-tensor-valid-x-corr-3d-ptr) :void
  (res (:pointer :int))
  (alpha :int)
  (ten (:pointer :int))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :int))
  (kt :long)
  (kr :long)
  (kc :long)
  (st :long)
  (sr :long)
  (sc :long))

;; void THTensor_(validConv3Dptr)(real *r_,
;;                                real alpha,
;;                                real *t_, long it, long ir, long ic,
;;                                real *k_, long kt, long kr, long kc,
;;                                long st, long sr, long sc);
(cffi:defcfun ("THIntTensor_validConv3Dptr" th-int-tensor-valid-conv-3d-ptr) :void
  (res (:pointer :int))
  (alpha :int)
  (ten (:pointer :int))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :int))
  (kt :long)
  (kr :long)
  (kc :long)
  (st :long)
  (sr :long)
  (sc :long))

;; void THTensor_(fullXCorr3Dptr)(real *r_,
;;                                real alpha,
;;                                real *t_, long it, long ir, long ic,
;;                                real *k_, long kt, long kr, long kc,
;;                                long st, long sr, long sc);
(cffi:defcfun ("THIntTensor_fullXCorr3Dptr" th-int-tensor-full-x-corr-3d-ptr) :void
  (res (:pointer :int))
  (alpha :int)
  (ten (:pointer :int))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :int))
  (kt :long)
  (kr :long)
  (kc :long)
  (st :long)
  (sr :long)
  (sc :long))

;; void THTensor_(fullConv3Dptr)(real *r_,
;;                               real alpha,
;;                               real *t_, long it, long ir, long ic,
;;                               real *k_, long kt, long kr, long kc,
;;                               long st, long sr, long sc);
(cffi:defcfun ("THIntTensor_fullConv3Dptr" th-int-tensor-full-conv-3d-ptr) :void
  (res (:pointer :int))
  (alpha :int)
  (ten (:pointer :int))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :int))
  (kt :long)
  (kr :long)
  (kc :long)
  (st :long)
  (sr :long)
  (sc :long))

;; void THTensor_(validXCorr3DRevptr)(real *r_,
;;                                    real alpha,
;;                                    real *t_, long it, long ir, long ic,
;;                                    real *k_, long kt, long kr, long kc,
;;                                    long st, long sr, long sc);
(cffi:defcfun ("THIntTensor_validXCorr3DRevptr" th-int-tensor-valid-x-corr-3d-rev-ptr) :void
  (res (:pointer :int))
  (alpha :int)
  (ten (:pointer :int))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :int))
  (kt :long)
  (kr :long)
  (kc :long)
  (st :long)
  (sr :long)
  (sc :long))

;; void THTensor_(conv3DRevger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol);
(cffi:defcfun ("THIntTensor_conv3DRevger" th-int-tensor-conv-3d-rev-ger) :void
  (result th-int-tensor-ptr)
  (beta :int)
  (alpha :int)
  (tensor th-int-tensor-ptr)
  (k th-int-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long))
;; void THTensor_(conv3Dger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THIntTensor_conv3Dger" th-int-tensor-conv-3d-ger) :void
  (result th-int-tensor-ptr)
  (beta :int)
  (alpha :int)
  (tensor th-int-tensor-ptr)
  (k th-int-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv3Dmv)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THIntTensor_conv3Dmv" th-int-tensor-conv-3d-mv) :void
  (result th-int-tensor-ptr)
  (beta :int)
  (alpha :int)
  (tensor th-int-tensor-ptr)
  (k th-int-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv3Dmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THIntTensor_conv3Dmul" th-int-tensor-conv-3d-mul) :void
  (result th-int-tensor-ptr)
  (beta :int)
  (alpha :int)
  (tensor th-int-tensor-ptr)
  (k th-int-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv3Dcmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THIntTensor_conv3Dcmul" th-int-tensor-conv-3d-cmul) :void
  (result th-int-tensor-ptr)
  (beta :int)
  (alpha :int)
  (tensor th-int-tensor-ptr)
  (k th-int-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
