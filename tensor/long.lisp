(declaim (optimize (speed 3) (debug 0) (safety 0)))

(in-package :th)

;; ACCESS METHODS
;; THStorage* THTensor_(storage)(const THTensor *self)
(cffi:defcfun ("THLongTensor_storage" th-long-tensor-storage) th-long-storage-ptr
  (tensor th-long-tensor-ptr))
;; ptrdiff_t THTensor_(storageOffset)(const THTensor *self)
(cffi:defcfun ("THLongTensor_storageOffset" th-long-tensor-storage-offset) :long-long
  (tensor th-long-tensor-ptr))
;; int THTensor_(nDimension)(const THTensor *self)
(cffi:defcfun ("THLongTensor_nDimension" th-long-tensor-n-dimension) :int
  (tensor th-long-tensor-ptr))
;; long THTensor_(size)(const THTensor *self, int dim)
(cffi:defcfun ("THLongTensor_size" th-long-tensor-size) :long
  (tensor th-long-tensor-ptr)
  (dim :int))
;; long THTensor_(stride)(const THTensor *self, int dim)
(cffi:defcfun ("THLongTensor_stride" th-long-tensor-stride) :long
  (tensor th-long-tensor-ptr)
  (dim :int))
;; THLongStorage *THTensor_(newSizeOf)(THTensor *self)
(cffi:defcfun ("THLongTensor_newSizeOf" th-long-tensor-new-size-of) th-long-storage-ptr
  (tensor th-long-tensor-ptr))
;; THLongStorage *THTensor_(newStrideOf)(THTensor *self)
(cffi:defcfun ("THLongTensor_newStrideOf" th-long-tensor-new-stride-of) th-long-storage-ptr
  (tensor th-long-tensor-ptr))
;; real *THTensor_(data)(const THTensor *self)
(cffi:defcfun ("THLongTensor_data" th-long-tensor-data) (:pointer :long)
  (tensor th-long-tensor-ptr))

;; void THTensor_(setFlag)(THTensor *self, const char flag)
(cffi:defcfun ("THLongTensor_setFlag" th-long-tensor-set-flag) :void
  (tensor th-long-tensor-ptr)
  (flag :char))
;; void THTensor_(clearFlag)(THTensor *self, const char flag)
(cffi:defcfun ("THLongTensor_clearFlag" th-long-tensor-clear-flag) :void
  (tensor th-long-tensor-ptr)
  (flag :char))

;; CREATION METHODS
;; THTensor *THTensor_(new)(void)
(cffi:defcfun ("THLongTensor_new" th-long-tensor-new) th-long-tensor-ptr)
;; THTensor *THTensor_(newWithTensor)(THTensor *tensor)
(cffi:defcfun ("THLongTensor_newWithTensor" th-long-tensor-new-with-tensor) th-long-tensor-ptr
  (tensor th-long-tensor-ptr))
;; stride might be NULL
;; THTensor *THTensor_(newWithStorage)(THStorage *storage_, ptrdiff_t storageOffset_,
;;                                            THLongStorage *size_, THLongStorage *stride_)
(cffi:defcfun ("THLongTensor_newWithStorage" th-long-tensor-new-with-storage)
    th-long-tensor-ptr
  (storage th-long-storage-ptr)
  (storage-offset :long-long)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
;; THTensor *THTensor_(newWithStorage1d)(THStorage *storage_, ptrdiff_t storageOffset_,
;;                                              long size0_, long stride0_);
(cffi:defcfun ("THLongTensor_newWithStorage1d" th-long-tensor-new-with-storage-1d)
    th-long-tensor-ptr
  (storage th-long-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long))
(cffi:defcfun ("THLongTensor_newWithStorage2d" th-long-tensor-new-with-storage-2d)
    th-long-tensor-ptr
  (storage th-long-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long))
(cffi:defcfun ("THLongTensor_newWithStorage3d" th-long-tensor-new-with-storage-3d)
    th-long-tensor-ptr
  (storage th-long-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long)
  (size2 :long)
  (stride2 :long))
(cffi:defcfun ("THLongTensor_newWithStorage4d" th-long-tensor-new-with-storage-4d)
    th-long-tensor-ptr
  (storage th-long-storage-ptr)
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
(cffi:defcfun ("THLongTensor_newWithSize" th-long-tensor-new-with-size) th-long-tensor-ptr
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
;; THTensor *THTensor_(newWithSize1d)(long size0_);
(cffi:defcfun ("THLongTensor_newWithSize1d" th-long-tensor-new-with-size-1d)
    th-long-tensor-ptr
  (size0 :long))
(cffi:defcfun ("THLongTensor_newWithSize2d" th-long-tensor-new-with-size-2d)
    th-long-tensor-ptr
  (size0 :long)
  (size1 :long))
(cffi:defcfun ("THLongTensor_newWithSize3d" th-long-tensor-new-with-size-3d)
    th-long-tensor-ptr
  (size0 :long)
  (size1 :long)
  (size2 :long))
(cffi:defcfun ("THLongTensor_newWithSize4d" th-long-tensor-new-with-size-4d)
    th-long-tensor-ptr
  (size0 :long)
  (size1 :long)
  (size2 :long)
  (size3 :long))

;; THTensor *THTensor_(newClone)(THTensor *self)
(cffi:defcfun ("THLongTensor_newClone" th-long-tensor-new-clone) th-long-tensor-ptr
  (tensor th-long-tensor-ptr))
(cffi:defcfun ("THLongTensor_newContiguous" th-long-tensor-new-contiguous) th-long-tensor-ptr
  (tensor th-long-tensor-ptr))
(cffi:defcfun ("THLongTensor_newSelect" th-long-tensor-new-select) th-long-tensor-ptr
  (tensor th-long-tensor-ptr)
  (dimension :int)
  (slice-index :long))
(cffi:defcfun ("THLongTensor_newNarrow" th-long-tensor-new-narrow) th-long-tensor-ptr
  (tensor th-long-tensor-ptr)
  (dimension :int)
  (first-index :long)
  (size :long))

(cffi:defcfun ("THLongTensor_newTranspose" th-long-tensor-new-transpose) th-long-tensor-ptr
  (tensor th-long-tensor-ptr)
  (dimension1 :int)
  (dimension2 :int))
(cffi:defcfun ("THLongTensor_newUnfold" th-long-tensor-new-unfold) th-long-tensor-ptr
  (tensor th-long-tensor-ptr)
  (dimension :int)
  (size :long)
  (step :long))
(cffi:defcfun ("THLongTensor_newView" th-long-tensor-new-view) th-long-tensor-ptr
  (tensor th-long-tensor-ptr)
  (size th-long-storage-ptr))

(cffi:defcfun ("THLongTensor_expand" th-long-tensor-expand) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (size th-long-storage-ptr))

(cffi:defcfun ("THLongTensor_resize" th-long-tensor-resize) :void
  (tensor th-long-tensor-ptr)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
(cffi:defcfun ("THLongTensor_resizeAs" th-long-tensor-resize-as) :void
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr))
(cffi:defcfun ("THLongTensor_resizeNd" th-long-tensor-resize-nd) :void
  (tensor th-long-tensor-ptr)
  (dimension :int)
  (size (:pointer :long))
  (stride (:pointer :long)))
(cffi:defcfun ("THLongTensor_resize1d" th-long-tensor-resize-1d) :void
  (tensor th-long-tensor-ptr)
  (size0 :long))
(cffi:defcfun ("THLongTensor_resize2d" th-long-tensor-resize-2d) :void
  (tensor th-long-tensor-ptr)
  (size0 :long)
  (size1 :long))
(cffi:defcfun ("THLongTensor_resize3d" th-long-tensor-resize-3d) :void
  (tensor th-long-tensor-ptr)
  (size0 :long)
  (size1 :long)
  (size2 :long))
(cffi:defcfun ("THLongTensor_resize4d" th-long-tensor-resize-4d) :void
  (tensor th-long-tensor-ptr)
  (size0 :long)
  (size1 :long)
  (size2 :long)
  (size3 :long))
(cffi:defcfun ("THLongTensor_resize5d" th-long-tensor-resize-5d) :void
  (tensor th-long-tensor-ptr)
  (size0 :long)
  (size1 :long)
  (size2 :long)
  (size3 :long)
  (size4 :long))

(cffi:defcfun ("THLongTensor_set" th-long-tensor-set) :void
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr))
(cffi:defcfun ("THLongTensor_setStorage" th-long-tensor-set-storage) :void
  (tensor th-long-tensor-ptr)
  (storage th-long-storage-ptr)
  (storage-offset :long-long)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
(cffi:defcfun ("THLongTensor_setStorageNd" th-long-tensor-set-storage-nd) :void
  (tensor th-long-tensor-ptr)
  (storage th-long-storage-ptr)
  (storage-offset :long-long)
  (dimension :int)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
(cffi:defcfun ("THLongTensor_setStorage1d" th-long-tensor-set-storage-1d) :void
  (tensor th-long-tensor-ptr)
  (storage th-long-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long))
(cffi:defcfun ("THLongTensor_setStorage2d" th-long-tensor-set-storage-2d) :void
  (tensor th-long-tensor-ptr)
  (storage th-long-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long))
(cffi:defcfun ("THLongTensor_setStorage3d" th-long-tensor-set-storage-3d) :void
  (tensor th-long-tensor-ptr)
  (storage th-long-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long)
  (size2 :long)
  (stride2 :long))
(cffi:defcfun ("THLongTensor_setStorage4d" th-long-tensor-set-storage-4d) :void
  (tensor th-long-tensor-ptr)
  (storage th-long-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long)
  (size2 :long)
  (stride2 :long)
  (size3 :long)
  (stride3 :long))

(cffi:defcfun ("THLongTensor_narrow" th-long-tensor-narrow) :void
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr)
  (dimension :int)
  (first-index :long)
  (size :long))
(cffi:defcfun ("THLongTensor_select" th-long-tensor-select) :void
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr)
  (dimension :int)
  (slice-index :long))
(cffi:defcfun ("THLongTensor_transpose" th-long-tensor-transpose) :void
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr)
  (dimension1 :int)
  (dimension2 :int))
(cffi:defcfun ("THLongTensor_unfold" th-long-tensor-unfold) :void
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr)
  (dimension :int)
  (size :long)
  (step :long))
(cffi:defcfun ("THLongTensor_squeeze" th-long-tensor-squeeze) :void
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr))
(cffi:defcfun ("THLongTensor_squeeze1d" th-long-tensor-squeeze-1d) :void
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr)
  (dimension :int))
(cffi:defcfun ("THLongTensor_unsqueeze1d" th-long-tensor-unsqueeze-1d) :void
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr)
  (dimension :int))

(cffi:defcfun ("THLongTensor_isContiguous" th-long-tensor-is-contiguous) :int
  (tensor th-long-tensor-ptr))
(cffi:defcfun ("THLongTensor_isSameSizeAs" th-long-tensor-is-same-size-as) :int
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr))
(cffi:defcfun ("THLongTensor_isSetTo" th-long-tensor-is-set-to) :int
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr))
(cffi:defcfun ("THLongTensor_isSize" th-long-tensor-is-size) :int
  (tensor th-long-tensor-ptr)
  (dims th-long-storage-ptr))
(cffi:defcfun ("THLongTensor_nElement" th-long-tensor-n-element) :long-long
  (tensor th-long-tensor-ptr))

(cffi:defcfun ("THLongTensor_retain" th-long-tensor-retain) :void
  (tensor th-long-tensor-ptr))
(cffi:defcfun ("THLongTensor_free" th-long-tensor-free) :void
  (tensor th-long-tensor-ptr))
(cffi:defcfun ("THLongTensor_freeCopyTo" th-long-tensor-free-copy-to) :void
  (source th-long-tensor-ptr)
  (target th-long-tensor-ptr))

;; slow access methods [check everything]
;; void THTensor_(set1d)(THTensor *tensor, long x0, real value);
(cffi:defcfun ("THLongTensor_set1d" th-long-tensor-set-1d) :void
  (tensor th-long-tensor-ptr)
  (index0 :long)
  (value :long))
;; void THTensor_(set2d)(THTensor *tensor, long x0, long x1, real value);
(cffi:defcfun ("THLongTensor_set2d" th-long-tensor-set-2d) :void
  (tensor th-long-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (value :long))
;; void THTensor_(set3d)(THTensor *tensor, long x0, long x1, long x2, real value);
(cffi:defcfun ("THLongTensor_set3d" th-long-tensor-set-3d) :void
  (tensor th-long-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long)
  (value :long))
;; void THTensor_(set4d)(THTensor *tensor, long x0, long x1, long x2, long x3, real value);
(cffi:defcfun ("THLongTensor_set4d" th-long-tensor-set-4d) :void
  (tensor th-long-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long)
  (index3 :long)
  (value :long))

(cffi:defcfun ("THLongTensor_get1d" th-long-tensor-get-1d) :long
  (tensor th-long-tensor-ptr)
  (index0 :long))
(cffi:defcfun ("THLongTensor_get2d" th-long-tensor-get-2d) :long
  (tensor th-long-tensor-ptr)
  (index0 :long)
  (index1 :long))
(cffi:defcfun ("THLongTensor_get3d" th-long-tensor-get-3d) :long
  (tensor th-long-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long))
(cffi:defcfun ("THLongTensor_get4d" th-long-tensor-get-4d) :long
  (tensor th-long-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long)
  (index3 :long))

;; support for copy betweeb different tensor types
;; void THTensor_(copy)(THTensor *tensor, THTensor *src);
(cffi:defcfun ("THLongTensor_copy" th-long-tensor-copy) :void
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(copyByte)(THTensor *tensor, struct THByteTensor *src);
(cffi:defcfun ("THLongTensor_copyByte" th-long-tensor-copy-byte) :void
  (tensor th-long-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(copyChar)(THTensor *tensor, struct THCharTensor *src);
(cffi:defcfun ("THLongTensor_copyChar" th-long-tensor-copy-char) :void
  (tensor th-long-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(copyShort)(THTensor *tensor, struct THShortTensor *src);
(cffi:defcfun ("THLongTensor_copyShort" th-long-tensor-copy-short) :void
  (tensor th-long-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(copyInt)(THTensor *tensor, struct THIntTensor *src);
(cffi:defcfun ("THLongTensor_copyInt" th-long-tensor-copy-int) :void
  (tensor th-long-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(copyLong)(THTensor *tensor, struct THLongTensor *src);
(cffi:defcfun ("THLongTensor_copyLong" th-long-tensor-copy-long) :void
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(copyFloat)(THTensor *tensor, struct THFloatTensor *src);
(cffi:defcfun ("THLongTensor_copyFloat" th-long-tensor-copy-float) :void
  (tensor th-long-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(copyDouble)(THTensor *tensor, struct THDoubleTensor *src);
(cffi:defcfun ("THLongTensor_copyDouble" th-long-tensor-copy-double) :void
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr))

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
(cffi:defcfun ("THLongTensor_random" th-long-tensor-random) :void
  (tensor th-long-tensor-ptr)
  (generator th-generator-ptr))
;; void THTensor_(clampedRandom)(THTensor *self, THGenerator *_generator, long min, long max)
(cffi:defcfun ("THLongTensor_clampedRandom" th-long-tensor-clamped-random) :void
  (tensor th-long-tensor-ptr)
  (genrator th-generator-ptr)
  (min :long)
  (max :long))
;; void THTensor_(cappedRandom)(THTensor *self, THGenerator *_generator, long max);
(cffi:defcfun ("THLongTensor_cappedRandom" th-long-tensor-capped-random) :void
  (tensor th-long-tensor-ptr)
  (generator th-generator-ptr)
  (max :long))
;; void THTensor_(geometric)(THTensor *self, THGenerator *_generator, double p);
(cffi:defcfun ("THLongTensor_geometric" th-long-tensor-geometric) :void
  (tensor th-long-tensor-ptr)
  (generator th-generator-ptr)
  (p :double))
;; void THTensor_(bernoulli)(THTensor *self, THGenerator *_generator, double p);
(cffi:defcfun ("THLongTensor_bernoulli" th-long-tensor-bernoulli) :void
  (tensor th-long-tensor-ptr)
  (generator th-generator-ptr)
  (p :double))
;; void THTensor_(bernoulli_FloatTensor)(THTensor *self, THGenerator *_generator, THFloatTensor *p);
(cffi:defcfun ("THLongTensor_bernoulli_FloatTensor" th-long-tensor-bernoulli-float-tensor) :void
  (tensor th-long-tensor-ptr)
  (generator th-generator-ptr)
  (p th-float-tensor-ptr))
;; void THTensor_(bernoulli_DoubleTensor)(THTensor *self, THGenerator *_generator, THDoubleTensor *p);
(cffi:defcfun ("THLongTensor_bernoulli_DoubleTensor" th-long-tensor-bernoulli-double-tensor)
    :void
  (tensor th-long-tensor-ptr)
  (generator th-generator-ptr)
  (p th-double-tensor-ptr))

;; #if defined(TH_REAL_IS_BYTE)
;; void THTensor_(getRNGState)(THGenerator *_generator, THTensor *self);
;; void THTensor_(setRNGState)(THGenerator *_generator, THTensor *self);
;; #endif

;; void THTensor_(fill)(THTensor *r_, real value);
(cffi:defcfun ("THLongTensor_fill" th-long-tensor-fill) :void
  (tensor th-long-tensor-ptr)
  (value :long))
;; void THTensor_(zero)(THTensor *r_);
(cffi:defcfun ("THLongTensor_zero" th-long-tensor-zero) :void
  (tensor th-long-tensor-ptr))

;; void THTensor_(maskedFill)(THTensor *tensor, THByteTensor *mask, real value);
(cffi:defcfun ("THLongTensor_maskedFill" th-long-tensor-masked-fill) :void
  (tensor th-long-tensor-ptr)
  (mask th-byte-tensor-ptr)
  (value :long))
;; void THTensor_(maskedCopy)(THTensor *tensor, THByteTensor *mask, THTensor* src);
(cffi:defcfun ("THLongTensor_maskedCopy" th-long-tensor-masked-copy) :void
  (tensor th-long-tensor-ptr)
  (mask th-byte-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(maskedSelect)(THTensor *tensor, THTensor* src, THByteTensor *mask);
(cffi:defcfun ("THLongTensor_maskedSelect" th-long-tensor-masked-select) :void
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr)
  (mask (th-byte-tensor-ptr)))

;; void THTensor_(nonzero)(THLongTensor *subscript, THTensor *tensor);
(cffi:defcfun ("THLongTensor_nonzero" th-long-tensor-nonzero) :void
  (subscript th-long-tensor-ptr)
  (tensor th-long-tensor-ptr))

;; void THTensor_(indexSelect)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index)
(cffi:defcfun ("THLongTensor_indexSelect" th-long-tensor-index-select) :void
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr))
;; void THTensor_(indexCopy)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THLongTensor_indexCopy" th-long-tensor-index-copy) :void
  (tensor th-long-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(indexAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THLongTensor_indexAdd" th-long-tensor-index-add) :void
  (tensor th-long-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(indexFill)(THTensor *tensor, int dim, THLongTensor *index, real val);
(cffi:defcfun ("THLongTensor_indexFill" th-long-tensor-index-fill) :void
  (tensor th-long-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (value :long))

;; void THTensor_(gather)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index);
(cffi:defcfun ("THLongTensor_gather" th-long-tensor-gather) :void
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr))
;; void THTensor_(scatter)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THLongTensor_scatter" th-long-tensor-scatter) :void
  (tensor th-long-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(scatterAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THLongTensor_scatterAdd" th-long-tensor-scatter-add) :void
  (tensor th-long-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(scatterFill)(THTensor *tensor, int dim, THLongTensor *index, real val);
(cffi:defcfun ("THLongTensor_scatterFill" th-long-tensor-scatter-fill) :void
  (tensor th-long-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (value :long))

;; accreal THTensor_(dot)(THTensor *t, THTensor *src);
(cffi:defcfun ("THLongTensor_dot" th-long-tensor-dot) :long
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr))

;; real THTensor_(minall)(THTensor *t);
(cffi:defcfun ("THLongTensor_minall" th-long-tensor-min-all) :long
  (tensor th-long-tensor-ptr))
;; real THTensor_(maxall)(THTensor *t);
(cffi:defcfun ("THLongTensor_maxall" th-long-tensor-max-all) :long
  (tensor th-long-tensor-ptr))
;; real THTensor_(medianall)(THTensor *t);
(cffi:defcfun ("THLongTensor_medianall" th-long-tensor-median-all) :long
  (tensor th-long-tensor-ptr))
;; accreal THTensor_(sumall)(THTensor *t);
(cffi:defcfun ("THLongTensor_sumall" th-long-tensor-sum-all) :long
  (tensor th-long-tensor-ptr))
;; accreal THTensor_(prodall)(THTensor *t);
(cffi:defcfun ("THLongTensor_prodall" th-long-tensor-prod-all) :long
  (tensor th-long-tensor-ptr))

;; void THTensor_(neg)(THTensor *self, THTensor *src);
(cffi:defcfun ("THLongTensor_neg" th-long-tensor-neg) :void
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr))

;; void THTensor_(add)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THLongTensor_add" th-long-tensor-add) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))
;; void THTensor_(sub)(THTensor *self, THTensor *src, real value);
(cffi:defcfun ("THLongTensor_sub" th-long-tensor-sub) :void
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr)
  (value :long))
;; void THTensor_(mul)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THLongTensor_mul" th-long-tensor-mul) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))
;; void THTensor_(div)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THLongTensor_div" th-long-tensor-div) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))
;; void THTensor_(lshift)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THLongTensor_lshift" th-long-tensor-lshift) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))
;; void THTensor_(rshift)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THLongTensor_rshift" th-long-tensor-rshift) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))
;; void THTensor_(fmod)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THLongTensor_fmod" th-long-tensor-fmod) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))
;; void THTensor_(remainder)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THLongTensor_remainder" th-long-tensor-remainder) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))
;; void THTensor_(clamp)(THTensor *r_, THTensor *t, real min_value, real max_value);
(cffi:defcfun ("THLongTensor_clamp" th-long-tensor-clamp) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (min-value :long)
  (max-value :long))
;; void THTensor_(bitand)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THLongTensor_bitand" th-long-tensor-bitand) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))
;; void THTensor_(bitor)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THLongTensor_bitor" th-long-tensor-bitor) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))
;; void THTensor_(bitxor)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THLongTensor_bitxor" th-long-tensor-bitxor) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))

;; void THTensor_(cadd)(THTensor *r_, THTensor *t, real value, THTensor *src);
(cffi:defcfun ("THLongTensor_cadd" th-long-tensor-cadd) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long)
  (src th-long-tensor-ptr))
;; void THTensor_(csub)(THTensor *self, THTensor *src1, real value, THTensor *src2);
(cffi:defcfun ("THLongTensor_csub" th-long-tensor-csub) :void
  (tensor th-long-tensor-ptr)
  (src1 th-long-tensor-ptr)
  (value :long)
  (src2 th-long-tensor-ptr))
;; void THTensor_(cmul)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THLongTensor_cmul" th-long-tensor-cmul) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(cpow)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THLongTensor_cpow" th-long-tensor-cpow) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(cdiv)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THLongTensor_cdiv" th-long-tensor-cdiv) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(clshift)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THLongTensor_clshift" th-long-tensor-clshift) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(crshift)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THLongTensor_crshift" th-long-tensor-crshift) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(cfmod)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THLongTensor_cfmod" th-long-tensor-cfmod) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(cremainder)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THLongTensor_cremainder" th-long-tensor-cremainder) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(cbitand)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THLongTensor_cbitand" th-long-tensor-cbitand) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(cbitor)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THLongTensor_cbitor" th-long-tensor-cbitor) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(cbitxor)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THLongTensor_cbitxor" th-long-tensor-cbitxor) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr))

;; void THTensor_(addcmul)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2);
(cffi:defcfun ("THLongTensor_addcmul" th-long-tensor-add-cmul) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long)
  (src1 th-long-tensor-ptr)
  (src2 th-long-tensor-ptr))
;; void THTensor_(addcdiv)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2);
(cffi:defcfun ("THLongTensor_addcdiv" th-long-tensor-add-cdiv) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long)
  (src1 th-long-tensor-ptr)
  (src2 th-long-tensor-ptr))
;; void THTensor_(addmv)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat,  THTensor *vec);
(cffi:defcfun ("THLongTensor_addmv" th-long-tensor-add-mv) :void
  (result th-long-tensor-ptr)
  (beta :long)
  (tensor th-long-tensor-ptr)
  (alpha :long)
  (matrix th-long-tensor-ptr)
  (vector th-long-tensor-ptr))
;; void THTensor_(addmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat1, THTensor *mat2);
(cffi:defcfun ("THLongTensor_addmm" th-long-tensor-add-mm) :void
  (result th-long-tensor-ptr)
  (beta :long)
  (tensor th-long-tensor-ptr)
  (alpha :long)
  (matrix1 th-long-tensor-ptr)
  (matrix2 th-long-tensor-ptr))
;; void THTensor_(addr)(THTensor *r_,  real beta, THTensor *t, real alpha, THTensor *vec1, THTensor *vec2);
(cffi:defcfun ("THLongTensor_addr" th-long-tensor-add-r) :void
  (result th-long-tensor-ptr)
  (beta :long)
  (tensor th-long-tensor-ptr)
  (alpha :long)
  (vector1 th-long-tensor-ptr)
  (vector2 th-long-tensor-ptr))
;; void THTensor_(addbmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *batch1, THTensor *batch2);
(cffi:defcfun ("THLongTensor_addbmm" th-long-tensor-add-bmm) :void
  (result th-long-tensor-ptr)
  (beta :long)
  (tensor th-long-tensor-ptr)
  (alpha :long)
  (batch1 th-long-tensor-ptr)
  (batch2 th-long-tensor-ptr))
;; void THTensor_(baddbmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *batch1, THTensor *batch2);
(cffi:defcfun ("THLongTensor_baddbmm" th-long-tensor-badd-bmm) :void
  (result th-long-tensor-ptr)
  (beta :long)
  (tensor th-long-tensor-ptr)
  (alpha :long)
  (batch1 th-long-tensor-ptr)
  (batch2 th-long-tensor-ptr))

;; void THTensor_(match)(THTensor *r_, THTensor *m1, THTensor *m2, real gain);
(cffi:defcfun ("THLongTensor_match" th-long-tensor-match) :void
  (result th-long-tensor-ptr)
  (m1 th-long-tensor-ptr)
  (m2 th-long-tensor-ptr)
  (gain :long))

;; ptrdiff_t THTensor_(numel)(THTensor *t);
(cffi:defcfun ("THLongTensor_numel" th-long-tensor-numel) :long-long
  (tensor th-long-tensor-ptr))
;; void THTensor_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THLongTensor_max" th-long-tensor-max) :void
  (values th-long-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THLongTensor_min" th-long-tensor-min) :void
  (values th-long-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(kthvalue)(THTensor *values_, THLongTensor *indices_, THTensor *t, long k, int dimension, int keepdim);
(cffi:defcfun ("THLongTensor_kthvalue" th-long-tensor-kth-value) :void
  (values th-long-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (k :long)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(mode)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THLongTensor_mode" th-long-tensor-mode) :void
  (values th-long-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(median)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THLongTensor_median" th-long-tensor-median) :void
  (values th-long-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(sum)(THTensor *r_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THLongTensor_sum" th-long-tensor-sum) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(prod)(THTensor *r_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THLongTensor_prod" th-long-tensor-prod) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(cumsum)(THTensor *r_, THTensor *t, int dimension);
(cffi:defcfun ("THLongTensor_cumsum" th-long-tensor-cum-sum) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (dimension :int))
;; void THTensor_(cumprod)(THTensor *r_, THTensor *t, int dimension);
(cffi:defcfun ("THLongTensor_cumprod" th-long-tensor-cum-prod) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (dimension :int))
;; void THTensor_(sign)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THLongTensor_sign" th-long-tensor-sign) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr))
;; accreal THTensor_(trace)(THTensor *t);
(cffi:defcfun ("THLongTensor_trace" th-long-tensor-trace) :long
  (tensor th-long-tensor-ptr))
;; void THTensor_(cross)(THTensor *r_, THTensor *a, THTensor *b, int dimension);
(cffi:defcfun ("THLongTensor_cross" th-long-tensor-cross) :void
  (result th-long-tensor-ptr)
  (a th-long-tensor-ptr)
  (b th-long-tensor-ptr)
  (dimension :int))

;; void THTensor_(cmax)(THTensor *r, THTensor *t, THTensor *src);
(cffi:defcfun ("THLongTensor_cmax" th-long-tensor-cmax) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(cmin)(THTensor *r, THTensor *t, THTensor *src);
(cffi:defcfun ("THLongTensor_cmin" th-long-tensor-cmin) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(cmaxValue)(THTensor *r, THTensor *t, real value);
(cffi:defcfun ("THLongTensor_cmaxValue" th-long-tensor-cmax-value) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))
;; void THTensor_(cminValue)(THTensor *r, THTensor *t, real value);
(cffi:defcfun ("THLongTensor_cminValue" th-long-tensor-cmin-value) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))

;; void THTensor_(zeros)(THTensor *r_, THLongStorage *size);
(cffi:defcfun ("THLongTensor_zeros" th-long-tensor-zeros) :void
  (result th-long-tensor-ptr)
  (size th-long-storage-ptr))
;; void THTensor_(zerosLike)(THTensor *r_, THTensor *input);
(cffi:defcfun ("THLongTensor_zerosLike" th-long-tensor-zero-like) :void
  (result th-long-tensor-ptr)
  (input th-long-tensor-ptr))
;; void THTensor_(ones)(THTensor *r_, THLongStorage *size);
(cffi:defcfun ("THLongTensor_ones" th-long-tensor-ones) :void
  (result th-long-tensor-ptr)
  (size th-long-storage-ptr))
;; void THTensor_(onesLike)(THTensor *r_, THTensor *input);
(cffi:defcfun ("THLongTensor_onesLike" th-long-tensor-one-like) :void
  (result th-long-tensor-ptr)
  (input th-long-tensor-ptr))
;; void THTensor_(diag)(THTensor *r_, THTensor *t, int k);
(cffi:defcfun ("THLongTensor_diag" th-long-tensor-diag) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (k :int))
;; void THTensor_(eye)(THTensor *r_, long n, long m);
(cffi:defcfun ("THLongTensor_eye" th-long-tensor-eye) :void
  (result th-long-tensor-ptr)
  (n :long)
  (m :long))
;; void THTensor_(arange)(THTensor *r_, accreal xmin, accreal xmax, accreal step);
(cffi:defcfun ("THLongTensor_arange" th-long-tensor-arange) :void
  (result th-long-tensor-ptr)
  (xmin :long)
  (xmax :long)
  (step :long))
;; void THTensor_(range)(THTensor *r_, accreal xmin, accreal xmax, accreal step);
(cffi:defcfun ("THLongTensor_range" th-long-tensor-range) :void
  (result th-long-tensor-ptr)
  (xmin :long)
  (xmax :long)
  (step :long))
;; void THTensor_(randperm)(THTensor *r_, THGenerator *_generator, long n);
(cffi:defcfun ("THLongTensor_randperm" th-long-tensor-rand-perm) :void
  (result th-long-tensor-ptr)
  (generator th-generator-ptr)
  (n :long))

;; void THTensor_(reshape)(THTensor *r_, THTensor *t, THLongStorage *size);
(cffi:defcfun ("THLongTensor_reshape" th-long-tensor-reshape) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (size th-long-storage-ptr))
;; void THTensor_(sort)(THTensor *rt_, THLongTensor *ri_, THTensor *t, int dimension, int descendingOrder);
(cffi:defcfun ("THLongTensor_sort" th-long-tensor-sort) :void
  (result-tensor th-long-tensor-ptr)
  (result-indices th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (dimension :int)
  (discending-order :int))
;; void THTensor_(topk)(THTensor *rt_, THLongTensor *ri_, THTensor *t, long k, int dim, int dir, int sorted);
(cffi:defcfun ("THLongTensor_topk" th-long-tensor-topk) :void
  (result-tensor th-long-tensor-ptr)
  (result-indices th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (k :long)
  (dim :int)
  (dir :int)
  (sorted :int))
;; void THTensor_(tril)(THTensor *r_, THTensor *t, long k);
(cffi:defcfun ("THLongTensor_tril" th-long-tensor-tril) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (k :long))
;; void THTensor_(triu)(THTensor *r_, THTensor *t, long k);
(cffi:defcfun ("THLongTensor_triu" th-long-tensor-triu) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (k :long))
;; void THTensor_(cat)(THTensor *r_, THTensor *ta, THTensor *tb, int dimension);
(cffi:defcfun ("THLongTensor_cat" th-long-tensor-cat) :void
  (result th-long-tensor-ptr)
  (tensora th-long-tensor-ptr)
  (tensorb th-long-tensor-ptr)
  (dimension :int))
;; void THTensor_(catArray)(THTensor *result, THTensor **inputs, int numInputs, int dimension);
(cffi:defcfun ("THLongTensor_catArray" th-long-tensor-cat-array) :void
  (result th-long-tensor-ptr)
  (inputs (:pointer th-long-tensor-ptr))
  (num-inputs :int)
  (dimension :int))

;; int THTensor_(equal)(THTensor *ta, THTensor *tb);
(cffi:defcfun ("THLongTensor_equal" th-long-tensor-equal) :int
  (tensora th-long-tensor-ptr)
  (tensorb th-long-tensor-ptr))

;; void THTensor_(ltValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THLongTensor_ltValue" th-long-tensor-lt-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))
;; void THTensor_(leValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THLongTensor_leValue" th-long-tensor-le-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))
;; void THTensor_(gtValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THLongTensor_gtValue" th-long-tensor-gt-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))
;; void THTensor_(geValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THLongTensor_geValue" th-long-tensor-ge-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))
;; void THTensor_(neValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THLongTensor_neValue" th-long-tensor-ne-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))
;; void THTensor_(eqValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THLongTensor_eqValue" th-long-tensor-eq-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))

;; void THTensor_(ltValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THLongTensor_ltValueT" th-long-tensor-lt-value-t) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))
;; void THTensor_(leValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THLongTensor_leValueT" th-long-tensor-le-value-t) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))
;; void THTensor_(gtValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THLongTensor_gtValueT" th-long-tensor-gt-value-t) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))
;; void THTensor_(geValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THLongTensor_geValueT" th-long-tensor-ge-value-t) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))
;; void THTensor_(neValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THLongTensor_neValueT" th-long-tensor-ne-value-t) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))
;; void THTensor_(eqValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THLongTensor_eqValueT" th-long-tensor-eq-value-t) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr)
  (value :long))

;; void THTensor_(ltTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THLongTensor_ltTensor" th-long-tensor-lt-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-long-tensor-ptr)
  (tensorb th-long-tensor-ptr))
;; void THTensor_(leTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THLongTensor_leTensor" th-long-tensor-le-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-long-tensor-ptr)
  (tensorb th-long-tensor-ptr))
;; void THTensor_(gtTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THLongTensor_gtTensor" th-long-tensor-gt-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-long-tensor-ptr)
  (tensorb th-long-tensor-ptr))
;; void THTensor_(geTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THLongTensor_geTensor" th-long-tensor-ge-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-long-tensor-ptr)
  (tensorb th-long-tensor-ptr))
;; void THTensor_(neTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THLongTensor_neTensor" th-long-tensor-ne-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-long-tensor-ptr)
  (tensorb th-long-tensor-ptr))
;; void THTensor_(eqTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THLongTensor_eqTensor" th-long-tensor-eq-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-long-tensor-ptr)
  (tensorb th-long-tensor-ptr))

;; void THTensor_(ltTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THLongTensor_ltTensorT" th-long-tensor-lt-tensor-t) :void
  (result th-long-tensor-ptr)
  (tensora th-long-tensor-ptr)
  (tensorb th-long-tensor-ptr))
;; void THTensor_(leTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THLongTensor_leTensorT" th-long-tensor-le-tensor-t) :void
  (result th-long-tensor-ptr)
  (tensora th-long-tensor-ptr)
  (tensorb th-long-tensor-ptr))
;; void THTensor_(gtTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THLongTensor_gtTensorT" th-long-tensor-gt-tensor-t) :void
  (result th-long-tensor-ptr)
  (tensora th-long-tensor-ptr)
  (tensorb th-long-tensor-ptr))
;; void THTensor_(geTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THLongTensor_geTensorT" th-long-tensor-ge-tensor-t) :void
  (result th-long-tensor-ptr)
  (tensora th-long-tensor-ptr)
  (tensorb th-long-tensor-ptr))
;; void THTensor_(neTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THLongTensor_neTensorT" th-long-tensor-ne-tensor-t) :void
  (result th-long-tensor-ptr)
  (tensora th-long-tensor-ptr)
  (tensorb th-long-tensor-ptr))
;; void THTensor_(eqTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THLongTensor_eqTensorT" th-long-tensor-eq-tensor-t) :void
  (result th-long-tensor-ptr)
  (tensora th-long-tensor-ptr)
  (tensorb th-long-tensor-ptr))

;; void THTensor_(abs)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THLongTensor_abs" th-long-tensor-abs) :void
  (result th-long-tensor-ptr)
  (tensor th-long-tensor-ptr))

;; #if defined(TH_REAL_IS_BYTE)

;; int THTensor_(logicalall)(THTensor *self);
;; (cffi:defcfun ("THLongTensor_logicalall" th-long-tensor-logical-all) :int
;;   (tensor (:pointer :void)))
;; int THTensor_(logicalany)(THTensor *self);
;; (cffi:defcfun ("THLongTensor_logicalany" th-long-tensor-logical-any) :int
;;   (tensor (:pointer :void)))

;; #endif /* TH_REAL_IS_BYTE */


;; void THTensor_(validXCorr2Dptr)(real *r_,
;;                                 real alpha,
;;                                 real *t_, long ir, long ic,
;;                                 real *k_, long kr, long kc,
;;                                 long sr, long sc);
(cffi:defcfun ("THLongTensor_validXCorr2Dptr" th-long-tensor-valid-x-corr-2d-ptr) :void
  (res (:pointer :long))
  (alpha :long)
  (ten (:pointer :long))
  (ir :long)
  (ic :long)
  (k (:pointer :long))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(validConv2Dptr)(real *r_,
;;                                real alpha,
;;                                real *t_, long ir, long ic,
;;                                real *k_, long kr, long kc,
;;                                long sr, long sc);
(cffi:defcfun ("THLongTensor_validConv2Dptr" th-long-tensor-valid-conv-2d-ptr) :void
  (res (:pointer :long))
  (alpha :long)
  (ten (:pointer :long))
  (ir :long)
  (ic :long)
  (k (:pointer :long))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(fullXCorr2Dptr)(real *r_,
;;                                real alpha,
;;                                real *t_, long ir, long ic,
;;                                real *k_, long kr, long kc,
;;                                long sr, long sc);
(cffi:defcfun ("THLongTensor_fullXCorr2Dptr" th-long-tensor-full-x-corr-2d-ptr) :void
  (res (:pointer :long))
  (alpha :long)
  (ten (:pointer :long))
  (ir :long)
  (ic :long)
  (k (:pointer :long))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(fullConv2Dptr)(real *r_,
;;                               real alpha,
;;                               real *t_, long ir, long ic,
;;                               real *k_, long kr, long kc,
;;                               long sr, long sc);
(cffi:defcfun ("THLongTensor_fullConv2Dptr" th-long-tensor-full-conv-2d-ptr) :void
  (res (:pointer :long))
  (alpha :long)
  (ten (:pointer :long))
  (ir :long)
  (ic :long)
  (k (:pointer :long))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(validXCorr2DRevptr)(real *r_,
;;                                    real alpha,
;;                                    real *t_, long ir, long ic,
;;                                    real *k_, long kr, long kc,
;;                                    long sr, long sc);
(cffi:defcfun ("THLongTensor_validXCorr2DRevptr" th-long-tensor-valid-x-corr-2d-rev-ptr) :void
  (res (:pointer :long))
  (alpha :long)
  (ten (:pointer :long))
  (ir :long)
  (ic :long)
  (k (:pointer :long))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(conv2DRevger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol);
(cffi:defcfun ("THLongTensor_conv2DRevger" th-long-tensor-conv-2d-rev-ger) :void
  (result th-long-tensor-ptr)
  (beta :long)
  (alpha :long)
  (tensor th-long-tensor-ptr)
  (k th-long-tensor-ptr)
  (srow :long)
  (scol :long))
;; void THTensor_(conv2DRevgerm)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol);
(cffi:defcfun ("THLongTensor_conv2DRevgerm" th-long-tensor-conv-2d-rev-germ) :void
  (result th-long-tensor-ptr)
  (beta :long)
  (alpha :long)
  (tensor th-long-tensor-ptr)
  (k th-long-tensor-ptr)
  (srow :long)
  (scol :long))
;; void THTensor_(conv2Dger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THLongTensor_conv2Dger" th-long-tensor-conv-2d-ger) :void
  (result th-long-tensor-ptr)
  (beta :long)
  (alpha :long)
  (tensor th-long-tensor-ptr)
  (k th-long-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv2Dmv)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THLongTensor_conv2Dmv" th-long-tensor-conv-2d-mv) :void
  (result th-long-tensor-ptr)
  (beta :long)
  (alpha :long)
  (tensor th-long-tensor-ptr)
  (k th-long-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv2Dmm)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THLongTensor_conv2Dmm" th-long-tensor-conv-2d-mm) :void
  (result th-long-tensor-ptr)
  (beta :long)
  (alpha :long)
  (tensor th-long-tensor-ptr)
  (k th-long-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv2Dmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THLongTensor_conv2Dmul" th-long-tensor-conv-2d-mul) :void
  (result th-long-tensor-ptr)
  (beta :long)
  (alpha :long)
  (tensor th-long-tensor-ptr)
  (k th-long-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv2Dcmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THLongTensor_conv2Dcmul" th-long-tensor-conv-2d-cmul) :void
  (result th-long-tensor-ptr)
  (beta :long)
  (alpha :long)
  (tensor th-long-tensor-ptr)
  (k th-long-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))

;; void THTensor_(validXCorr3Dptr)(real *r_,
;;                                 real alpha,
;;                                 real *t_, long it, long ir, long ic,
;;                                 real *k_, long kt, long kr, long kc,
;;                                 long st, long sr, long sc);
(cffi:defcfun ("THLongTensor_validXCorr3Dptr" th-long-tensor-valid-x-corr-3d-ptr) :void
  (res (:pointer :long))
  (alpha :long)
  (ten (:pointer :long))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :long))
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
(cffi:defcfun ("THLongTensor_validConv3Dptr" th-long-tensor-valid-conv-3d-ptr) :void
  (res (:pointer :long))
  (alpha :long)
  (ten (:pointer :long))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :long))
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
(cffi:defcfun ("THLongTensor_fullXCorr3Dptr" th-long-tensor-full-x-corr-3d-ptr) :void
  (res (:pointer :long))
  (alpha :long)
  (ten (:pointer :long))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :long))
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
(cffi:defcfun ("THLongTensor_fullConv3Dptr" th-long-tensor-full-conv-3d-ptr) :void
  (res (:pointer :long))
  (alpha :long)
  (ten (:pointer :long))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :long))
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
(cffi:defcfun ("THLongTensor_validXCorr3DRevptr" th-long-tensor-valid-x-corr-3d-rev-ptr) :void
  (res (:pointer :long))
  (alpha :long)
  (ten (:pointer :long))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :long))
  (kt :long)
  (kr :long)
  (kc :long)
  (st :long)
  (sr :long)
  (sc :long))

;; void THTensor_(conv3DRevger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol);
(cffi:defcfun ("THLongTensor_conv3DRevger" th-long-tensor-conv-3d-rev-ger) :void
  (result th-long-tensor-ptr)
  (beta :long)
  (alpha :long)
  (tensor th-long-tensor-ptr)
  (k th-long-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long))
;; void THTensor_(conv3Dger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THLongTensor_conv3Dger" th-long-tensor-conv-3d-ger) :void
  (result th-long-tensor-ptr)
  (beta :long)
  (alpha :long)
  (tensor th-long-tensor-ptr)
  (k th-long-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv3Dmv)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THLongTensor_conv3Dmv" th-long-tensor-conv-3d-mv) :void
  (result th-long-tensor-ptr)
  (beta :long)
  (alpha :long)
  (tensor th-long-tensor-ptr)
  (k th-long-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv3Dmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THLongTensor_conv3Dmul" th-long-tensor-conv-3d-mul) :void
  (result th-long-tensor-ptr)
  (beta :long)
  (alpha :long)
  (tensor th-long-tensor-ptr)
  (k th-long-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv3Dcmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THLongTensor_conv3Dcmul" th-long-tensor-conv-3d-cmul) :void
  (result th-long-tensor-ptr)
  (beta :long)
  (alpha :long)
  (tensor th-long-tensor-ptr)
  (k th-long-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
