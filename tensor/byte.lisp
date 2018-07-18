(declaim (optimize (speed 3) (debug 0) (safety 0)))

(in-package :th)

;; ACCESS METHODS
;; THStorage* THTensor_(storage)(const THTensor *self)
(cffi:defcfun ("THByteTensor_storage" th-byte-tensor-storage) th-byte-storage-ptr
  (tensor th-byte-tensor-ptr))
;; ptrdiff_t THTensor_(storageOffset)(const THTensor *self)
(cffi:defcfun ("THByteTensor_storageOffset" th-byte-tensor-storage-offset) :long-long
  (tensor th-byte-tensor-ptr))
;; int THTensor_(nDimension)(const THTensor *self)
(cffi:defcfun ("THByteTensor_nDimension" th-byte-tensor-n-dimension) :int
  (tensor th-byte-tensor-ptr))
;; long THTensor_(size)(const THTensor *self, int dim)
(cffi:defcfun ("THByteTensor_size" th-byte-tensor-size) :long
  (tensor th-byte-tensor-ptr)
  (dim :int))
;; long THTensor_(stride)(const THTensor *self, int dim)
(cffi:defcfun ("THByteTensor_stride" th-byte-tensor-stride) :long
  (tensor th-byte-tensor-ptr)
  (dim :int))
;; THLongStorage *THTensor_(newSizeOf)(THTensor *self)
(cffi:defcfun ("THByteTensor_newSizeOf" th-byte-tensor-new-size-of) th-long-storage-ptr
  (tensor th-byte-tensor-ptr))
;; THLongStorage *THTensor_(newStrideOf)(THTensor *self)
(cffi:defcfun ("THByteTensor_newStrideOf" th-byte-tensor-new-stride-of) th-long-storage-ptr
  (tensor th-byte-tensor-ptr))
;; real *THTensor_(data)(const THTensor *self)
(cffi:defcfun ("THByteTensor_data" th-byte-tensor-data) (:pointer :unsigned-char)
  (tensor th-byte-tensor-ptr))

;; void THTensor_(setFlag)(THTensor *self, const char flag)
(cffi:defcfun ("THByteTensor_setFlag" th-byte-tensor-set-flag) :void
  (tensor th-byte-tensor-ptr)
  (flag :char))
;; void THTensor_(clearFlag)(THTensor *self, const char flag)
(cffi:defcfun ("THByteTensor_clearFlag" th-byte-tensor-clear-flag) :void
  (tensor th-byte-tensor-ptr)
  (flag :char))

;; CREATION METHODS
;; THTensor *THTensor_(new)(void)
(cffi:defcfun ("THByteTensor_new" th-byte-tensor-new) th-byte-tensor-ptr)
;; THTensor *THTensor_(newWithTensor)(THTensor *tensor)
(cffi:defcfun ("THByteTensor_newWithTensor" th-byte-tensor-new-with-tensor) th-byte-tensor-ptr
  (tensor th-byte-tensor-ptr))
;; stride might be NULL
;; THTensor *THTensor_(newWithStorage)(THStorage *storage_, ptrdiff_t storageOffset_,
;;                                            THLongStorage *size_, THLongStorage *stride_)
(cffi:defcfun ("THByteTensor_newWithStorage" th-byte-tensor-new-with-storage)
    th-byte-tensor-ptr
  (storage th-byte-storage-ptr)
  (storage-offset :long-long)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
;; THTensor *THTensor_(newWithStorage1d)(THStorage *storage_, ptrdiff_t storageOffset_,
;;                                              long size0_, long stride0_);
(cffi:defcfun ("THByteTensor_newWithStorage1d" th-byte-tensor-new-with-storage-1d)
    th-byte-tensor-ptr
  (storage th-byte-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long))
(cffi:defcfun ("THByteTensor_newWithStorage2d" th-byte-tensor-new-with-storage-2d)
    th-byte-tensor-ptr
  (storage th-byte-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long))
(cffi:defcfun ("THByteTensor_newWithStorage3d" th-byte-tensor-new-with-storage-3d)
    th-byte-tensor-ptr
  (storage th-byte-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long)
  (size2 :long)
  (stride2 :long))
(cffi:defcfun ("THByteTensor_newWithStorage4d" th-byte-tensor-new-with-storage-4d)
    th-byte-tensor-ptr
  (storage th-byte-storage-ptr)
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
(cffi:defcfun ("THByteTensor_newWithSize" th-byte-tensor-new-with-size) th-byte-tensor-ptr
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
;; THTensor *THTensor_(newWithSize1d)(long size0_);
(cffi:defcfun ("THByteTensor_newWithSize1d" th-byte-tensor-new-with-size-1d)
    th-byte-tensor-ptr
  (size0 :long))
(cffi:defcfun ("THByteTensor_newWithSize2d" th-byte-tensor-new-with-size-2d)
    th-byte-tensor-ptr
  (size0 :long)
  (size1 :long))
(cffi:defcfun ("THByteTensor_newWithSize3d" th-byte-tensor-new-with-size-3d)
    th-byte-tensor-ptr
  (size0 :long)
  (size1 :long)
  (size2 :long))
(cffi:defcfun ("THByteTensor_newWithSize4d" th-byte-tensor-new-with-size-4d)
    th-byte-tensor-ptr
  (size0 :long)
  (size1 :long)
  (size2 :long)
  (size3 :long))

;; THTensor *THTensor_(newClone)(THTensor *self)
(cffi:defcfun ("THByteTensor_newClone" th-byte-tensor-new-clone) th-byte-tensor-ptr
  (tensor th-byte-tensor-ptr))
(cffi:defcfun ("THByteTensor_newContiguous" th-byte-tensor-new-contiguous) th-byte-tensor-ptr
  (tensor th-byte-tensor-ptr))
(cffi:defcfun ("THByteTensor_newSelect" th-byte-tensor-new-select) th-byte-tensor-ptr
  (tensor th-byte-tensor-ptr)
  (dimension :int)
  (slice-index :long))
(cffi:defcfun ("THByteTensor_newNarrow" th-byte-tensor-new-narrow) th-byte-tensor-ptr
  (tensor th-byte-tensor-ptr)
  (dimension :int)
  (first-index :long)
  (size :long))

(cffi:defcfun ("THByteTensor_newTranspose" th-byte-tensor-new-transpose) th-byte-tensor-ptr
  (tensor th-byte-tensor-ptr)
  (dimension1 :int)
  (dimension2 :int))
(cffi:defcfun ("THByteTensor_newUnfold" th-byte-tensor-new-unfold) th-byte-tensor-ptr
  (tensor th-byte-tensor-ptr)
  (dimension :int)
  (size :long)
  (step :long))
(cffi:defcfun ("THByteTensor_newView" th-byte-tensor-new-view) th-byte-tensor-ptr
  (tensor th-byte-tensor-ptr)
  (size th-long-storage-ptr))

(cffi:defcfun ("THByteTensor_expand" th-byte-tensor-expand) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (size th-long-storage-ptr))

(cffi:defcfun ("THByteTensor_resize" th-byte-tensor-resize) :void
  (tensor th-byte-tensor-ptr)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
(cffi:defcfun ("THByteTensor_resizeAs" th-byte-tensor-resize-as) :void
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr))
(cffi:defcfun ("THByteTensor_resizeNd" th-byte-tensor-resize-nd) :void
  (tensor th-byte-tensor-ptr)
  (dimension :int)
  (size (:pointer :long))
  (stride (:pointer :long)))
(cffi:defcfun ("THByteTensor_resize1d" th-byte-tensor-resize-1d) :void
  (tensor th-byte-tensor-ptr)
  (size0 :long))
(cffi:defcfun ("THByteTensor_resize2d" th-byte-tensor-resize-2d) :void
  (tensor th-byte-tensor-ptr)
  (size0 :long)
  (size1 :long))
(cffi:defcfun ("THByteTensor_resize3d" th-byte-tensor-resize-3d) :void
  (tensor th-byte-tensor-ptr)
  (size0 :long)
  (size1 :long)
  (size2 :long))
(cffi:defcfun ("THByteTensor_resize4d" th-byte-tensor-resize-4d) :void
  (tensor th-byte-tensor-ptr)
  (size0 :long)
  (size1 :long)
  (size2 :long)
  (size3 :long))
(cffi:defcfun ("THByteTensor_resize5d" th-byte-tensor-resize-5d) :void
  (tensor th-byte-tensor-ptr)
  (size0 :long)
  (size1 :long)
  (size2 :long)
  (size3 :long)
  (size4 :long))

(cffi:defcfun ("THByteTensor_set" th-byte-tensor-set) :void
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr))
(cffi:defcfun ("THByteTensor_setStorage" th-byte-tensor-set-storage) :void
  (tensor th-byte-tensor-ptr)
  (storage th-byte-storage-ptr)
  (storage-offset :long-long)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
(cffi:defcfun ("THByteTensor_setStorageNd" th-byte-tensor-set-storage-nd) :void
  (tensor th-byte-tensor-ptr)
  (storage th-byte-storage-ptr)
  (storage-offset :long-long)
  (dimension :int)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
(cffi:defcfun ("THByteTensor_setStorage1d" th-byte-tensor-set-storage-1d) :void
  (tensor th-byte-tensor-ptr)
  (storage th-byte-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long))
(cffi:defcfun ("THByteTensor_setStorage2d" th-byte-tensor-set-storage-2d) :void
  (tensor th-byte-tensor-ptr)
  (storage th-byte-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long))
(cffi:defcfun ("THByteTensor_setStorage3d" th-byte-tensor-set-storage-3d) :void
  (tensor th-byte-tensor-ptr)
  (storage th-byte-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long)
  (size2 :long)
  (stride2 :long))
(cffi:defcfun ("THByteTensor_setStorage4d" th-byte-tensor-set-storage-4d) :void
  (tensor th-byte-tensor-ptr)
  (storage th-byte-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long)
  (size2 :long)
  (stride2 :long)
  (size3 :long)
  (stride3 :long))

(cffi:defcfun ("THByteTensor_narrow" th-byte-tensor-narrow) :void
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr)
  (dimension :int)
  (first-index :long)
  (size :long))
(cffi:defcfun ("THByteTensor_select" th-byte-tensor-select) :void
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr)
  (dimension :int)
  (slice-index :long))
(cffi:defcfun ("THByteTensor_transpose" th-byte-tensor-transpose) :void
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr)
  (dimension1 :int)
  (dimension2 :int))
(cffi:defcfun ("THByteTensor_unfold" th-byte-tensor-unfold) :void
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr)
  (dimension :int)
  (size :long)
  (step :long))
(cffi:defcfun ("THByteTensor_squeeze" th-byte-tensor-squeeze) :void
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr))
(cffi:defcfun ("THByteTensor_squeeze1d" th-byte-tensor-squeeze-1d) :void
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr)
  (dimension :int))
(cffi:defcfun ("THByteTensor_unsqueeze1d" th-byte-tensor-unsqueeze-1d) :void
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr)
  (dimension :int))

(cffi:defcfun ("THByteTensor_isContiguous" th-byte-tensor-is-contiguous) :int
  (tensor th-byte-tensor-ptr))
(cffi:defcfun ("THByteTensor_isSameSizeAs" th-byte-tensor-is-same-size-as) :int
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr))
(cffi:defcfun ("THByteTensor_isSetTo" th-byte-tensor-is-set-to) :int
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr))
(cffi:defcfun ("THByteTensor_isSize" th-byte-tensor-is-size) :int
  (tensor th-byte-tensor-ptr)
  (dims th-long-storage-ptr))
(cffi:defcfun ("THByteTensor_nElement" th-byte-tensor-n-element) :long-long
  (tensor th-byte-tensor-ptr))

(cffi:defcfun ("THByteTensor_retain" th-byte-tensor-retain) :void
  (tensor th-byte-tensor-ptr))
(cffi:defcfun ("THByteTensor_free" th-byte-tensor-free) :void
  (tensor th-byte-tensor-ptr))
(cffi:defcfun ("THByteTensor_freeCopyTo" th-byte-tensor-free-copy-to) :void
  (source th-byte-tensor-ptr)
  (target th-byte-tensor-ptr))

;; slow access methods [check everything]
;; void THTensor_(set1d)(THTensor *tensor, long x0, real value);
(cffi:defcfun ("THByteTensor_set1d" th-byte-tensor-set-1d) :void
  (tensor th-byte-tensor-ptr)
  (index0 :long)
  (value :unsigned-char))
;; void THTensor_(set2d)(THTensor *tensor, long x0, long x1, real value);
(cffi:defcfun ("THByteTensor_set2d" th-byte-tensor-set-2d) :void
  (tensor th-byte-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (value :unsigned-char))
;; void THTensor_(set3d)(THTensor *tensor, long x0, long x1, long x2, real value);
(cffi:defcfun ("THByteTensor_set3d" th-byte-tensor-set-3d) :void
  (tensor th-byte-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long)
  (value :unsigned-char))
;; void THTensor_(set4d)(THTensor *tensor, long x0, long x1, long x2, long x3, real value);
(cffi:defcfun ("THByteTensor_set4d" th-byte-tensor-set-4d) :void
  (tensor th-byte-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long)
  (index3 :long)
  (value :unsigned-char))

(cffi:defcfun ("THByteTensor_get1d" th-byte-tensor-get-1d) :unsigned-char
  (tensor th-byte-tensor-ptr)
  (index0 :long))
(cffi:defcfun ("THByteTensor_get2d" th-byte-tensor-get-2d) :unsigned-char
  (tensor th-byte-tensor-ptr)
  (index0 :long)
  (index1 :long))
(cffi:defcfun ("THByteTensor_get3d" th-byte-tensor-get-3d) :unsigned-char
  (tensor th-byte-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long))
(cffi:defcfun ("THByteTensor_get4d" th-byte-tensor-get-4d) :unsigned-char
  (tensor th-byte-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long)
  (index3 :long))

;; support for copy betweeb different tensor types
;; void THTensor_(copy)(THTensor *tensor, THTensor *src);
(cffi:defcfun ("THByteTensor_copy" th-byte-tensor-copy) :void
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(copyByte)(THTensor *tensor, struct THByteTensor *src);
(cffi:defcfun ("THByteTensor_copyByte" th-byte-tensor-copy-byte) :void
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(copyChar)(THTensor *tensor, struct THCharTensor *src);
(cffi:defcfun ("THByteTensor_copyChar" th-byte-tensor-copy-char) :void
  (tensor th-byte-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(copyShort)(THTensor *tensor, struct THShortTensor *src);
(cffi:defcfun ("THByteTensor_copyShort" th-byte-tensor-copy-short) :void
  (tensor th-byte-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(copyInt)(THTensor *tensor, struct THIntTensor *src);
(cffi:defcfun ("THByteTensor_copyInt" th-byte-tensor-copy-int) :void
  (tensor th-byte-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(copyLong)(THTensor *tensor, struct THLongTensor *src);
(cffi:defcfun ("THByteTensor_copyLong" th-byte-tensor-copy-long) :void
  (tensor th-byte-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(copyFloat)(THTensor *tensor, struct THFloatTensor *src);
(cffi:defcfun ("THByteTensor_copyFloat" th-byte-tensor-copy-float) :void
  (tensor th-byte-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(copyDouble)(THTensor *tensor, struct THDoubleTensor *src);
(cffi:defcfun ("THByteTensor_copyDouble" th-byte-tensor-copy-double) :void
  (tensor th-byte-tensor-ptr)
  (src th-double-tensor-ptr))

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
(cffi:defcfun ("THByteTensor_random" th-byte-tensor-random) :void
  (tensor th-byte-tensor-ptr)
  (generator th-generator-ptr))
;; void THTensor_(clampedRandom)(THTensor *self, THGenerator *_generator, long min, long max)
(cffi:defcfun ("THByteTensor_clampedRandom" th-byte-tensor-clamped-random) :void
  (tensor th-byte-tensor-ptr)
  (genrator th-generator-ptr)
  (min :long)
  (max :long))
;; void THTensor_(cappedRandom)(THTensor *self, THGenerator *_generator, long max);
(cffi:defcfun ("THByteTensor_cappedRandom" th-byte-tensor-capped-random) :void
  (tensor th-byte-tensor-ptr)
  (generator th-generator-ptr)
  (max :long))
;; void THTensor_(geometric)(THTensor *self, THGenerator *_generator, double p);
(cffi:defcfun ("THByteTensor_geometric" th-byte-tensor-geometric) :void
  (tensor th-byte-tensor-ptr)
  (generator th-generator-ptr)
  (p :double))
;; void THTensor_(bernoulli)(THTensor *self, THGenerator *_generator, double p);
(cffi:defcfun ("THByteTensor_bernoulli" th-byte-tensor-bernoulli) :void
  (tensor th-byte-tensor-ptr)
  (generator th-generator-ptr)
  (p :double))
;; void THTensor_(bernoulli_FloatTensor)(THTensor *self, THGenerator *_generator, THFloatTensor *p);
(cffi:defcfun ("THByteTensor_bernoulli_FloatTensor" th-byte-tensor-bernoulli-float-tensor) :void
  (tensor th-byte-tensor-ptr)
  (generator th-generator-ptr)
  (p th-float-tensor-ptr))
;; void THTensor_(bernoulli_DoubleTensor)(THTensor *self, THGenerator *_generator, THDoubleTensor *p);
(cffi:defcfun ("THByteTensor_bernoulli_DoubleTensor" th-byte-tensor-bernoulli-double-tensor)
    :void
  (tensor th-byte-tensor-ptr)
  (generator th-generator-ptr)
  (p th-double-tensor-ptr))

;; #if defined(TH_REAL_IS_BYTE)
;; void THTensor_(getRNGState)(THGenerator *_generator, THTensor *self);
(cffi:defcfun ("THByteTensor_getRNGState" th-byte-tensor-get-rng-state) :void
  (generator th-generator-ptr)
  (tensor th-byte-tensor-ptr))
;; void THTensor_(setRNGState)(THGenerator *_generator, THTensor *self);
(cffi:defcfun ("THByteTensor_setRNGState" th-byte-tensor-set-rng-state) :void
  (generator th-generator-ptr)
  (tensor th-byte-tensor-ptr))
;; #endif

;; void THTensor_(fill)(THTensor *r_, real value);
(cffi:defcfun ("THByteTensor_fill" th-byte-tensor-fill) :void
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))
;; void THTensor_(zero)(THTensor *r_);
(cffi:defcfun ("THByteTensor_zero" th-byte-tensor-zero) :void
  (tensor th-byte-tensor-ptr))

;; void THTensor_(maskedFill)(THTensor *tensor, THByteTensor *mask, real value);
(cffi:defcfun ("THByteTensor_maskedFill" th-byte-tensor-masked-fill) :void
  (tensor th-byte-tensor-ptr)
  (mask th-byte-tensor-ptr)
  (value :unsigned-char))
;; void THTensor_(maskedCopy)(THTensor *tensor, THByteTensor *mask, THTensor* src);
(cffi:defcfun ("THByteTensor_maskedCopy" th-byte-tensor-masked-copy) :void
  (tensor th-byte-tensor-ptr)
  (mask th-byte-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(maskedSelect)(THTensor *tensor, THTensor* src, THByteTensor *mask);
(cffi:defcfun ("THByteTensor_maskedSelect" th-byte-tensor-masked-select) :void
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr)
  (mask (th-byte-tensor-ptr)))

;; void THTensor_(nonzero)(THLongTensor *subscript, THTensor *tensor);
(cffi:defcfun ("THByteTensor_nonzero" th-byte-tensor-nonzero) :void
  (subscript th-long-tensor-ptr)
  (tensor th-byte-tensor-ptr))

;; void THTensor_(indexSelect)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index)
(cffi:defcfun ("THByteTensor_indexSelect" th-byte-tensor-index-select) :void
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr))
;; void THTensor_(indexCopy)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THByteTensor_indexCopy" th-byte-tensor-index-copy) :void
  (tensor th-byte-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(indexAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THByteTensor_indexAdd" th-byte-tensor-index-add) :void
  (tensor th-byte-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(indexFill)(THTensor *tensor, int dim, THLongTensor *index, real val);
(cffi:defcfun ("THByteTensor_indexFill" th-byte-tensor-index-fill) :void
  (tensor th-byte-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (value :unsigned-char))

;; void THTensor_(gather)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index);
(cffi:defcfun ("THByteTensor_gather" th-byte-tensor-gather) :void
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr))
;; void THTensor_(scatter)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THByteTensor_scatter" th-byte-tensor-scatter) :void
  (tensor th-byte-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(scatterAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THByteTensor_scatterAdd" th-byte-tensor-scatter-add) :void
  (tensor th-byte-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(scatterFill)(THTensor *tensor, int dim, THLongTensor *index, real val);
(cffi:defcfun ("THByteTensor_scatterFill" th-byte-tensor-scatter-fill) :void
  (tensor th-byte-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (value :unsigned-char))

;; accreal THTensor_(dot)(THTensor *t, THTensor *src);
(cffi:defcfun ("THByteTensor_dot" th-byte-tensor-dot) :long
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr))

;; real THTensor_(minall)(THTensor *t);
(cffi:defcfun ("THByteTensor_minall" th-byte-tensor-min-all) :unsigned-char
  (tensor th-byte-tensor-ptr))
;; real THTensor_(maxall)(THTensor *t);
(cffi:defcfun ("THByteTensor_maxall" th-byte-tensor-max-all) :unsigned-char
  (tensor th-byte-tensor-ptr))
;; real THTensor_(medianall)(THTensor *t);
(cffi:defcfun ("THByteTensor_medianall" th-byte-tensor-median-all) :unsigned-char
  (tensor th-byte-tensor-ptr))
;; accreal THTensor_(sumall)(THTensor *t);
(cffi:defcfun ("THByteTensor_sumall" th-byte-tensor-sum-all) :long
  (tensor th-byte-tensor-ptr))
;; accreal THTensor_(prodall)(THTensor *t);
(cffi:defcfun ("THByteTensor_prodall" th-byte-tensor-prod-all) :long
  (tensor th-byte-tensor-ptr))

;; void THTensor_(add)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THByteTensor_add" th-byte-tensor-add) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))
;; void THTensor_(sub)(THTensor *self, THTensor *src, real value);
(cffi:defcfun ("THByteTensor_sub" th-byte-tensor-sub) :void
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr)
  (value :unsigned-char))
;; void THTensor_(mul)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THByteTensor_mul" th-byte-tensor-mul) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))
;; void THTensor_(div)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THByteTensor_div" th-byte-tensor-div) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))
;; void THTensor_(lshift)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THByteTensor_lshift" th-byte-tensor-lshift) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))
;; void THTensor_(rshift)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THByteTensor_rshift" th-byte-tensor-rshift) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))
;; void THTensor_(fmod)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THByteTensor_fmod" th-byte-tensor-fmod) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))
;; void THTensor_(remainder)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THByteTensor_remainder" th-byte-tensor-remainder) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))
;; void THTensor_(clamp)(THTensor *r_, THTensor *t, real min_value, real max_value);
(cffi:defcfun ("THByteTensor_clamp" th-byte-tensor-clamp) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (min-value :unsigned-char)
  (max-value :unsigned-char))
;; void THTensor_(bitand)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THByteTensor_bitand" th-byte-tensor-bitand) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))
;; void THTensor_(bitor)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THByteTensor_bitor" th-byte-tensor-bitor) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))
;; void THTensor_(bitxor)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THByteTensor_bitxor" th-byte-tensor-bitxor) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))

;; void THTensor_(cadd)(THTensor *r_, THTensor *t, real value, THTensor *src);
(cffi:defcfun ("THByteTensor_cadd" th-byte-tensor-cadd) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char)
  (src th-byte-tensor-ptr))
;; void THTensor_(csub)(THTensor *self, THTensor *src1, real value, THTensor *src2);
(cffi:defcfun ("THByteTensor_csub" th-byte-tensor-csub) :void
  (tensor th-byte-tensor-ptr)
  (src1 th-byte-tensor-ptr)
  (value :unsigned-char)
  (src2 th-byte-tensor-ptr))
;; void THTensor_(cmul)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THByteTensor_cmul" th-byte-tensor-cmul) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(cpow)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THByteTensor_cpow" th-byte-tensor-cpow) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(cdiv)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THByteTensor_cdiv" th-byte-tensor-cdiv) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(clshift)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THByteTensor_clshift" th-byte-tensor-clshift) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(crshift)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THByteTensor_crshift" th-byte-tensor-crshift) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(cfmod)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THByteTensor_cfmod" th-byte-tensor-cfmod) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(cremainder)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THByteTensor_cremainder" th-byte-tensor-cremainder) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(cbitand)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THByteTensor_cbitand" th-byte-tensor-cbitand) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(cbitor)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THByteTensor_cbitor" th-byte-tensor-cbitor) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(cbitxor)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THByteTensor_cbitxor" th-byte-tensor-cbitxor) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr))

;; void THTensor_(addcmul)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2);
(cffi:defcfun ("THByteTensor_addcmul" th-byte-tensor-add-cmul) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char)
  (src1 th-byte-tensor-ptr)
  (src2 th-byte-tensor-ptr))
;; void THTensor_(addcdiv)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2);
(cffi:defcfun ("THByteTensor_addcdiv" th-byte-tensor-add-cdiv) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char)
  (src1 th-byte-tensor-ptr)
  (src2 th-byte-tensor-ptr))
;; void THTensor_(addmv)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat,  THTensor *vec);
(cffi:defcfun ("THByteTensor_addmv" th-byte-tensor-add-mv) :void
  (result th-byte-tensor-ptr)
  (beta :unsigned-char)
  (tensor th-byte-tensor-ptr)
  (alpha :unsigned-char)
  (matrix th-byte-tensor-ptr)
  (vector th-byte-tensor-ptr))
;; void THTensor_(addmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat1, THTensor *mat2);
(cffi:defcfun ("THByteTensor_addmm" th-byte-tensor-add-mm) :void
  (result th-byte-tensor-ptr)
  (beta :unsigned-char)
  (tensor th-byte-tensor-ptr)
  (alpha :unsigned-char)
  (matrix1 th-byte-tensor-ptr)
  (matrix2 th-byte-tensor-ptr))
;; void THTensor_(addr)(THTensor *r_,  real beta, THTensor *t, real alpha, THTensor *vec1, THTensor *vec2);
(cffi:defcfun ("THByteTensor_addr" th-byte-tensor-add-r) :void
  (result th-byte-tensor-ptr)
  (beta :unsigned-char)
  (tensor th-byte-tensor-ptr)
  (alpha :unsigned-char)
  (vector1 th-byte-tensor-ptr)
  (vector2 th-byte-tensor-ptr))
;; void THTensor_(addbmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *batch1, THTensor *batch2);
(cffi:defcfun ("THByteTensor_addbmm" th-byte-tensor-add-bmm) :void
  (result th-byte-tensor-ptr)
  (beta :unsigned-char)
  (tensor th-byte-tensor-ptr)
  (alpha :unsigned-char)
  (batch1 th-byte-tensor-ptr)
  (batch2 th-byte-tensor-ptr))
;; void THTensor_(baddbmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *batch1, THTensor *batch2);
(cffi:defcfun ("THByteTensor_baddbmm" th-byte-tensor-badd-bmm) :void
  (result th-byte-tensor-ptr)
  (beta :unsigned-char)
  (tensor th-byte-tensor-ptr)
  (alpha :unsigned-char)
  (batch1 th-byte-tensor-ptr)
  (batch2 th-byte-tensor-ptr))

;; void THTensor_(match)(THTensor *r_, THTensor *m1, THTensor *m2, real gain);
(cffi:defcfun ("THByteTensor_match" th-byte-tensor-match) :void
  (result th-byte-tensor-ptr)
  (m1 th-byte-tensor-ptr)
  (m2 th-byte-tensor-ptr)
  (gain :unsigned-char))

;; ptrdiff_t THTensor_(numel)(THTensor *t);
(cffi:defcfun ("THByteTensor_numel" th-byte-tensor-numel) :long-long
  (tensor th-byte-tensor-ptr))
;; void THTensor_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THByteTensor_max" th-byte-tensor-max) :void
  (values th-byte-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THByteTensor_min" th-byte-tensor-min) :void
  (values th-byte-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(kthvalue)(THTensor *values_, THLongTensor *indices_, THTensor *t, long k, int dimension, int keepdim);
(cffi:defcfun ("THByteTensor_kthvalue" th-byte-tensor-kth-value) :void
  (values th-byte-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (k :long)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(mode)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THByteTensor_mode" th-byte-tensor-mode) :void
  (values th-byte-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(median)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THByteTensor_median" th-byte-tensor-median) :void
  (values th-byte-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(sum)(THTensor *r_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THByteTensor_sum" th-byte-tensor-sum) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(prod)(THTensor *r_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THByteTensor_prod" th-byte-tensor-prod) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(cumsum)(THTensor *r_, THTensor *t, int dimension);
(cffi:defcfun ("THByteTensor_cumsum" th-byte-tensor-cum-sum) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (dimension :int))
;; void THTensor_(cumprod)(THTensor *r_, THTensor *t, int dimension);
(cffi:defcfun ("THByteTensor_cumprod" th-byte-tensor-cum-prod) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (dimension :int))
;; void THTensor_(sign)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THByteTensor_sign" th-byte-tensor-sign) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr))
;; accreal THTensor_(trace)(THTensor *t);
(cffi:defcfun ("THByteTensor_trace" th-byte-tensor-trace) :long
  (tensor th-byte-tensor-ptr))
;; void THTensor_(cross)(THTensor *r_, THTensor *a, THTensor *b, int dimension);
(cffi:defcfun ("THByteTensor_cross" th-byte-tensor-cross) :void
  (result th-byte-tensor-ptr)
  (a th-byte-tensor-ptr)
  (b th-byte-tensor-ptr)
  (dimension :int))

;; void THTensor_(cmax)(THTensor *r, THTensor *t, THTensor *src);
(cffi:defcfun ("THByteTensor_cmax" th-byte-tensor-cmax) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(cmin)(THTensor *r, THTensor *t, THTensor *src);
(cffi:defcfun ("THByteTensor_cmin" th-byte-tensor-cmin) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(cmaxValue)(THTensor *r, THTensor *t, real value);
(cffi:defcfun ("THByteTensor_cmaxValue" th-byte-tensor-cmax-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))
;; void THTensor_(cminValue)(THTensor *r, THTensor *t, real value);
(cffi:defcfun ("THByteTensor_cminValue" th-byte-tensor-cmin-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))

;; void THTensor_(zeros)(THTensor *r_, THLongStorage *size);
(cffi:defcfun ("THByteTensor_zeros" th-byte-tensor-zeros) :void
  (result th-byte-tensor-ptr)
  (size th-long-storage-ptr))
;; void THTensor_(zerosLike)(THTensor *r_, THTensor *input);
(cffi:defcfun ("THByteTensor_zerosLike" th-byte-tensor-zero-like) :void
  (result th-byte-tensor-ptr)
  (input th-byte-tensor-ptr))
;; void THTensor_(ones)(THTensor *r_, THLongStorage *size);
(cffi:defcfun ("THByteTensor_ones" th-byte-tensor-ones) :void
  (result th-byte-tensor-ptr)
  (size th-long-storage-ptr))
;; void THTensor_(onesLike)(THTensor *r_, THTensor *input);
(cffi:defcfun ("THByteTensor_onesLike" th-byte-tensor-one-like) :void
  (result th-byte-tensor-ptr)
  (input th-byte-tensor-ptr))
;; void THTensor_(diag)(THTensor *r_, THTensor *t, int k);
(cffi:defcfun ("THByteTensor_diag" th-byte-tensor-diag) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (k :int))
;; void THTensor_(eye)(THTensor *r_, long n, long m);
(cffi:defcfun ("THByteTensor_eye" th-byte-tensor-eye) :void
  (result th-byte-tensor-ptr)
  (n :long)
  (m :long))
;; void THTensor_(arange)(THTensor *r_, accreal xmin, accreal xmax, accreal step);
(cffi:defcfun ("THByteTensor_arange" th-byte-tensor-arange) :void
  (result th-byte-tensor-ptr)
  (xmin :long)
  (xmax :long)
  (step :long))
;; void THTensor_(range)(THTensor *r_, accreal xmin, accreal xmax, accreal step);
(cffi:defcfun ("THByteTensor_range" th-byte-tensor-range) :void
  (result th-byte-tensor-ptr)
  (xmin :long)
  (xmax :long)
  (step :long))
;; void THTensor_(randperm)(THTensor *r_, THGenerator *_generator, long n);
(cffi:defcfun ("THByteTensor_randperm" th-byte-tensor-rand-perm) :void
  (result th-byte-tensor-ptr)
  (generator th-generator-ptr)
  (n :long))

;; void THTensor_(reshape)(THTensor *r_, THTensor *t, THLongStorage *size);
(cffi:defcfun ("THByteTensor_reshape" th-byte-tensor-reshape) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (size th-long-storage-ptr))
;; void THTensor_(sort)(THTensor *rt_, THLongTensor *ri_, THTensor *t, int dimension, int descendingOrder);
(cffi:defcfun ("THByteTensor_sort" th-byte-tensor-sort) :void
  (result-tensor th-byte-tensor-ptr)
  (result-indices th-long-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (dimension :int)
  (discending-order :int))
;; void THTensor_(topk)(THTensor *rt_, THLongTensor *ri_, THTensor *t, long k, int dim, int dir, int sorted);
(cffi:defcfun ("THByteTensor_topk" th-byte-tensor-topk) :void
  (result-tensor th-byte-tensor-ptr)
  (result-indices th-long-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (k :long)
  (dim :int)
  (dir :int)
  (sorted :int))
;; void THTensor_(tril)(THTensor *r_, THTensor *t, long k);
(cffi:defcfun ("THByteTensor_tril" th-byte-tensor-tril) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (k :long))
;; void THTensor_(triu)(THTensor *r_, THTensor *t, long k);
(cffi:defcfun ("THByteTensor_triu" th-byte-tensor-triu) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (k :long))
;; void THTensor_(cat)(THTensor *r_, THTensor *ta, THTensor *tb, int dimension);
(cffi:defcfun ("THByteTensor_cat" th-byte-tensor-cat) :void
  (result th-byte-tensor-ptr)
  (tensora th-byte-tensor-ptr)
  (tensorb th-byte-tensor-ptr)
  (dimension :int))
;; void THTensor_(catArray)(THTensor *result, THTensor **inputs, int numInputs, int dimension);
(cffi:defcfun ("THByteTensor_catArray" th-byte-tensor-cat-array) :void
  (result th-byte-tensor-ptr)
  (inputs (:pointer th-byte-tensor-ptr))
  (num-inputs :int)
  (dimension :int))

;; int THTensor_(equal)(THTensor *ta, THTensor *tb);
(cffi:defcfun ("THByteTensor_equal" th-byte-tensor-equal) :int
  (tensora th-byte-tensor-ptr)
  (tensorb th-byte-tensor-ptr))

;; void THTensor_(ltValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THByteTensor_ltValue" th-byte-tensor-lt-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))
;; void THTensor_(leValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THByteTensor_leValue" th-byte-tensor-le-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))
;; void THTensor_(gtValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THByteTensor_gtValue" th-byte-tensor-gt-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))
;; void THTensor_(geValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THByteTensor_geValue" th-byte-tensor-ge-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))
;; void THTensor_(neValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THByteTensor_neValue" th-byte-tensor-ne-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))
;; void THTensor_(eqValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THByteTensor_eqValue" th-byte-tensor-eq-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))

;; void THTensor_(ltValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THByteTensor_ltValueT" th-byte-tensor-lt-value-t) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))
;; void THTensor_(leValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THByteTensor_leValueT" th-byte-tensor-le-value-t) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))
;; void THTensor_(gtValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THByteTensor_gtValueT" th-byte-tensor-gt-value-t) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))
;; void THTensor_(geValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THByteTensor_geValueT" th-byte-tensor-ge-value-t) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))
;; void THTensor_(neValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THByteTensor_neValueT" th-byte-tensor-ne-value-t) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))
;; void THTensor_(eqValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THByteTensor_eqValueT" th-byte-tensor-eq-value-t) :void
  (result th-byte-tensor-ptr)
  (tensor th-byte-tensor-ptr)
  (value :unsigned-char))

;; void THTensor_(ltTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THByteTensor_ltTensor" th-byte-tensor-lt-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-byte-tensor-ptr)
  (tensorb th-byte-tensor-ptr))
;; void THTensor_(leTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THByteTensor_leTensor" th-byte-tensor-le-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-byte-tensor-ptr)
  (tensorb th-byte-tensor-ptr))
;; void THTensor_(gtTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THByteTensor_gtTensor" th-byte-tensor-gt-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-byte-tensor-ptr)
  (tensorb th-byte-tensor-ptr))
;; void THTensor_(geTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THByteTensor_geTensor" th-byte-tensor-ge-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-byte-tensor-ptr)
  (tensorb th-byte-tensor-ptr))
;; void THTensor_(neTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THByteTensor_neTensor" th-byte-tensor-ne-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-byte-tensor-ptr)
  (tensorb th-byte-tensor-ptr))
;; void THTensor_(eqTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THByteTensor_eqTensor" th-byte-tensor-eq-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-byte-tensor-ptr)
  (tensorb th-byte-tensor-ptr))

;; void THTensor_(ltTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THByteTensor_ltTensorT" th-byte-tensor-lt-tensor-t) :void
  (result th-byte-tensor-ptr)
  (tensora th-byte-tensor-ptr)
  (tensorb th-byte-tensor-ptr))
;; void THTensor_(leTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THByteTensor_leTensorT" th-byte-tensor-le-tensor-t) :void
  (result th-byte-tensor-ptr)
  (tensora th-byte-tensor-ptr)
  (tensorb th-byte-tensor-ptr))
;; void THTensor_(gtTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THByteTensor_gtTensorT" th-byte-tensor-gt-tensor-t) :void
  (result th-byte-tensor-ptr)
  (tensora th-byte-tensor-ptr)
  (tensorb th-byte-tensor-ptr))
;; void THTensor_(geTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THByteTensor_geTensorT" th-byte-tensor-ge-tensor-t) :void
  (result th-byte-tensor-ptr)
  (tensora th-byte-tensor-ptr)
  (tensorb th-byte-tensor-ptr))
;; void THTensor_(neTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THByteTensor_neTensorT" th-byte-tensor-ne-tensor-t) :void
  (result th-byte-tensor-ptr)
  (tensora th-byte-tensor-ptr)
  (tensorb th-byte-tensor-ptr))
;; void THTensor_(eqTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THByteTensor_eqTensorT" th-byte-tensor-eq-tensor-t) :void
  (result th-byte-tensor-ptr)
  (tensora th-byte-tensor-ptr)
  (tensorb th-byte-tensor-ptr))

;; void THTensor_(validXCorr2Dptr)(real *r_,
;;                                 real alpha,
;;                                 real *t_, long ir, long ic,
;;                                 real *k_, long kr, long kc,
;;                                 long sr, long sc);
(cffi:defcfun ("THByteTensor_validXCorr2Dptr" th-byte-tensor-valid-x-corr-2d-ptr) :void
  (res (:pointer :unsigned-char))
  (alpha :unsigned-char)
  (ten (:pointer :unsigned-char))
  (ir :long)
  (ic :long)
  (k (:pointer :unsigned-char))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(validConv2Dptr)(real *r_,
;;                                real alpha,
;;                                real *t_, long ir, long ic,
;;                                real *k_, long kr, long kc,
;;                                long sr, long sc);
(cffi:defcfun ("THByteTensor_validConv2Dptr" th-byte-tensor-valid-conv-2d-ptr) :void
  (res (:pointer :unsigned-char))
  (alpha :unsigned-char)
  (ten (:pointer :unsigned-char))
  (ir :long)
  (ic :long)
  (k (:pointer :unsigned-char))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(fullXCorr2Dptr)(real *r_,
;;                                real alpha,
;;                                real *t_, long ir, long ic,
;;                                real *k_, long kr, long kc,
;;                                long sr, long sc);
(cffi:defcfun ("THByteTensor_fullXCorr2Dptr" th-byte-tensor-full-x-corr-2d-ptr) :void
  (res (:pointer :unsigned-char))
  (alpha :unsigned-char)
  (ten (:pointer :unsigned-char))
  (ir :long)
  (ic :long)
  (k (:pointer :unsigned-char))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(fullConv2Dptr)(real *r_,
;;                               real alpha,
;;                               real *t_, long ir, long ic,
;;                               real *k_, long kr, long kc,
;;                               long sr, long sc);
(cffi:defcfun ("THByteTensor_fullConv2Dptr" th-byte-tensor-full-conv-2d-ptr) :void
  (res (:pointer :unsigned-char))
  (alpha :unsigned-char)
  (ten (:pointer :unsigned-char))
  (ir :long)
  (ic :long)
  (k (:pointer :unsigned-char))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(validXCorr2DRevptr)(real *r_,
;;                                    real alpha,
;;                                    real *t_, long ir, long ic,
;;                                    real *k_, long kr, long kc,
;;                                    long sr, long sc);
(cffi:defcfun ("THByteTensor_validXCorr2DRevptr" th-byte-tensor-valid-x-corr-2d-rev-ptr) :void
  (res (:pointer :unsigned-char))
  (alpha :unsigned-char)
  (ten (:pointer :unsigned-char))
  (ir :long)
  (ic :long)
  (k (:pointer :unsigned-char))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(conv2DRevger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol);
(cffi:defcfun ("THByteTensor_conv2DRevger" th-byte-tensor-conv-2d-rev-ger) :void
  (result th-byte-tensor-ptr)
  (beta :unsigned-char)
  (alpha :unsigned-char)
  (tensor th-byte-tensor-ptr)
  (k th-byte-tensor-ptr)
  (srow :long)
  (scol :long))
;; void THTensor_(conv2DRevgerm)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol);
(cffi:defcfun ("THByteTensor_conv2DRevgerm" th-byte-tensor-conv-2d-rev-germ) :void
  (result th-byte-tensor-ptr)
  (beta :unsigned-char)
  (alpha :unsigned-char)
  (tensor th-byte-tensor-ptr)
  (k th-byte-tensor-ptr)
  (srow :long)
  (scol :long))
;; void THTensor_(conv2Dger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THByteTensor_conv2Dger" th-byte-tensor-conv-2d-ger) :void
  (result th-byte-tensor-ptr)
  (beta :unsigned-char)
  (alpha :unsigned-char)
  (tensor th-byte-tensor-ptr)
  (k th-byte-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv2Dmv)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THByteTensor_conv2Dmv" th-byte-tensor-conv-2d-mv) :void
  (result th-byte-tensor-ptr)
  (beta :unsigned-char)
  (alpha :unsigned-char)
  (tensor th-byte-tensor-ptr)
  (k th-byte-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv2Dmm)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THByteTensor_conv2Dmm" th-byte-tensor-conv-2d-mm) :void
  (result th-byte-tensor-ptr)
  (beta :unsigned-char)
  (alpha :unsigned-char)
  (tensor th-byte-tensor-ptr)
  (k th-byte-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv2Dmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THByteTensor_conv2Dmul" th-byte-tensor-conv-2d-mul) :void
  (result th-byte-tensor-ptr)
  (beta :unsigned-char)
  (alpha :unsigned-char)
  (tensor th-byte-tensor-ptr)
  (k th-byte-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv2Dcmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THByteTensor_conv2Dcmul" th-byte-tensor-conv-2d-cmul) :void
  (result th-byte-tensor-ptr)
  (beta :unsigned-char)
  (alpha :unsigned-char)
  (tensor th-byte-tensor-ptr)
  (k th-byte-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))

;; void THTensor_(validXCorr3Dptr)(real *r_,
;;                                 real alpha,
;;                                 real *t_, long it, long ir, long ic,
;;                                 real *k_, long kt, long kr, long kc,
;;                                 long st, long sr, long sc);
(cffi:defcfun ("THByteTensor_validXCorr3Dptr" th-byte-tensor-valid-x-corr-3d-ptr) :void
  (res (:pointer :unsigned-char))
  (alpha :unsigned-char)
  (ten (:pointer :unsigned-char))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :unsigned-char))
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
(cffi:defcfun ("THByteTensor_validConv3Dptr" th-byte-tensor-valid-conv-3d-ptr) :void
  (res (:pointer :unsigned-char))
  (alpha :unsigned-char)
  (ten (:pointer :unsigned-char))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :unsigned-char))
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
(cffi:defcfun ("THByteTensor_fullXCorr3Dptr" th-byte-tensor-full-x-corr-3d-ptr) :void
  (res (:pointer :unsigned-char))
  (alpha :unsigned-char)
  (ten (:pointer :unsigned-char))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :unsigned-char))
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
(cffi:defcfun ("THByteTensor_fullConv3Dptr" th-byte-tensor-full-conv-3d-ptr) :void
  (res (:pointer :unsigned-char))
  (alpha :unsigned-char)
  (ten (:pointer :unsigned-char))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :unsigned-char))
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
(cffi:defcfun ("THByteTensor_validXCorr3DRevptr" th-byte-tensor-valid-x-corr-3d-rev-ptr) :void
  (res (:pointer :unsigned-char))
  (alpha :unsigned-char)
  (ten (:pointer :unsigned-char))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :unsigned-char))
  (kt :long)
  (kr :long)
  (kc :long)
  (st :long)
  (sr :long)
  (sc :long))

;; void THTensor_(conv3DRevger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol);
(cffi:defcfun ("THByteTensor_conv3DRevger" th-byte-tensor-conv-3d-rev-ger) :void
  (result th-byte-tensor-ptr)
  (beta :unsigned-char)
  (alpha :unsigned-char)
  (tensor th-byte-tensor-ptr)
  (k th-byte-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long))
;; void THTensor_(conv3Dger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THByteTensor_conv3Dger" th-byte-tensor-conv-3d-ger) :void
  (result th-byte-tensor-ptr)
  (beta :unsigned-char)
  (alpha :unsigned-char)
  (tensor th-byte-tensor-ptr)
  (k th-byte-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv3Dmv)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THByteTensor_conv3Dmv" th-byte-tensor-conv-3d-mv) :void
  (result th-byte-tensor-ptr)
  (beta :unsigned-char)
  (alpha :unsigned-char)
  (tensor th-byte-tensor-ptr)
  (k th-byte-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv3Dmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THByteTensor_conv3Dmul" th-byte-tensor-conv-3d-mul) :void
  (result th-byte-tensor-ptr)
  (beta :unsigned-char)
  (alpha :unsigned-char)
  (tensor th-byte-tensor-ptr)
  (k th-byte-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv3Dcmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THByteTensor_conv3Dcmul" th-byte-tensor-conv-3d-cmul) :void
  (result th-byte-tensor-ptr)
  (beta :unsigned-char)
  (alpha :unsigned-char)
  (tensor th-byte-tensor-ptr)
  (k th-byte-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
