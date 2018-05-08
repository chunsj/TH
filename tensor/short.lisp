(in-package :th)

;; ACCESS METHODS
;; THStorage* THTensor_(storage)(const THTensor *self)
(cffi:defcfun ("THShortTensor_storage" th-short-tensor-storage) th-short-storage-ptr
  (tensor th-short-tensor-ptr))
;; ptrdiff_t THTensor_(storageOffset)(const THTensor *self)
(cffi:defcfun ("THShortTensor_storageOffset" th-short-tensor-storage-offset) :long-long
  (tensor th-short-tensor-ptr))
;; int THTensor_(nDimension)(const THTensor *self)
(cffi:defcfun ("THShortTensor_nDimension" th-short-tensor-n-dimension) :int
  (tensor th-short-tensor-ptr))
;; long THTensor_(size)(const THTensor *self, int dim)
(cffi:defcfun ("THShortTensor_size" th-short-tensor-size) :long
  (tensor th-short-tensor-ptr)
  (dim :int))
;; long THTensor_(stride)(const THTensor *self, int dim)
(cffi:defcfun ("THShortTensor_stride" th-short-tensor-stride) :long
  (tensor th-short-tensor-ptr)
  (dim :int))
;; THLongStorage *THTensor_(newSizeOf)(THTensor *self)
(cffi:defcfun ("THShortTensor_newSizeOf" th-short-tensor-new-size-of) th-long-storage-ptr
  (tensor th-short-tensor-ptr))
;; THLongStorage *THTensor_(newStrideOf)(THTensor *self)
(cffi:defcfun ("THShortTensor_newStrideOf" th-short-tensor-new-stride-of) th-long-storage-ptr
  (tensor th-short-tensor-ptr))
;; real *THTensor_(data)(const THTensor *self)
(cffi:defcfun ("THShortTensor_data" th-short-tensor-data) (:pointer :double)
  (tensor th-short-tensor-ptr))

;; void THTensor_(setFlag)(THTensor *self, const char flag)
(cffi:defcfun ("THShortTensor_setFlag" th-short-tensor-set-flag) :void
  (tensor th-short-tensor-ptr)
  (flag :char))
;; void THTensor_(clearFlag)(THTensor *self, const char flag)
(cffi:defcfun ("THShortTensor_clearFlag" th-short-tensor-clear-flag) :void
  (tensor th-short-tensor-ptr)
  (flag :char))

;; CREATION METHODS
;; THTensor *THTensor_(new)(void)
(cffi:defcfun ("THShortTensor_new" th-short-tensor-new) th-short-tensor-ptr)
;; THTensor *THTensor_(newWithTensor)(THTensor *tensor)
(cffi:defcfun ("THShortTensor_newWithTensor" th-short-tensor-new-with-tensor) th-short-tensor-ptr
  (tensor th-short-tensor-ptr))
;; stride might be NULL
;; THTensor *THTensor_(newWithStorage)(THStorage *storage_, ptrdiff_t storageOffset_,
;;                                            THLongStorage *size_, THLongStorage *stride_)
(cffi:defcfun ("THShortTensor_newWithStorage" th-short-tensor-new-with-storage)
    th-short-tensor-ptr
  (storage th-short-storage-ptr)
  (storage-offset :long-long)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
;; THTensor *THTensor_(newWithStorage1d)(THStorage *storage_, ptrdiff_t storageOffset_,
;;                                              long size0_, long stride0_);
(cffi:defcfun ("THShortTensor_newWithStorage1d" th-short-tensor-new-with-storage-1d)
    th-short-tensor-ptr
  (storage th-short-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long))
(cffi:defcfun ("THShortTensor_newWithStorage2d" th-short-tensor-new-with-storage-2d)
    th-short-tensor-ptr
  (storage th-short-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long))
(cffi:defcfun ("THShortTensor_newWithStorage3d" th-short-tensor-new-with-storage-3d)
    th-short-tensor-ptr
  (storage th-short-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long)
  (size2 :long)
  (stride2 :long))
(cffi:defcfun ("THShortTensor_newWithStorage4d" th-short-tensor-new-with-storage-4d)
    th-short-tensor-ptr
  (storage th-short-storage-ptr)
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
(cffi:defcfun ("THShortTensor_newWithSize" th-short-tensor-new-with-size) th-short-tensor-ptr
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
;; THTensor *THTensor_(newWithSize1d)(long size0_);
(cffi:defcfun ("THShortTensor_newWithSize1d" th-short-tensor-new-with-size-1d)
    th-short-tensor-ptr
  (size0 :long))
(cffi:defcfun ("THShortTensor_newWithSize2d" th-short-tensor-new-with-size-2d)
    th-short-tensor-ptr
  (size0 :long)
  (size1 :long))
(cffi:defcfun ("THShortTensor_newWithSize3d" th-short-tensor-new-with-size-3d)
    th-short-tensor-ptr
  (size0 :long)
  (size1 :long)
  (size2 :long))
(cffi:defcfun ("THShortTensor_newWithSize4d" th-short-tensor-new-with-size-4d)
    th-short-tensor-ptr
  (size0 :long)
  (size1 :long)
  (size2 :long)
  (size3 :long))

;; THTensor *THTensor_(newClone)(THTensor *self)
(cffi:defcfun ("THShortTensor_newClone" th-short-tensor-new-clone) th-short-tensor-ptr
  (tensor th-short-tensor-ptr))
(cffi:defcfun ("THShortTensor_newContiguous" th-short-tensor-new-contiguous) th-short-tensor-ptr
  (tensor th-short-tensor-ptr))
(cffi:defcfun ("THShortTensor_newSelect" th-short-tensor-new-select) th-short-tensor-ptr
  (tensor th-short-tensor-ptr)
  (dimension :int)
  (slice-index :long))
(cffi:defcfun ("THShortTensor_newNarrow" th-short-tensor-new-narrow) th-short-tensor-ptr
  (tensor th-short-tensor-ptr)
  (dimension :int)
  (first-index :long)
  (size :long))

(cffi:defcfun ("THShortTensor_newTranspose" th-short-tensor-new-transpose) th-short-tensor-ptr
  (tensor th-short-tensor-ptr)
  (dimension1 :int)
  (dimension2 :int))
(cffi:defcfun ("THShortTensor_newUnfold" th-short-tensor-new-unfold) th-short-tensor-ptr
  (tensor th-short-tensor-ptr)
  (dimension :int)
  (size :long)
  (step :long))
(cffi:defcfun ("THShortTensor_newView" th-short-tensor-new-view) th-short-tensor-ptr
  (tensor th-short-tensor-ptr)
  (size th-long-storage-ptr))

(cffi:defcfun ("THShortTensor_resize" th-short-tensor-resize) :void
  (tensor th-short-tensor-ptr)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
(cffi:defcfun ("THShortTensor_resizeAs" th-short-tensor-resize-as) :void
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr))
(cffi:defcfun ("THShortTensor_resizeNd" th-short-tensor-resize-nd) :void
  (tensor th-short-tensor-ptr)
  (dimension :int)
  (size (:pointer :long))
  (stride (:pointer :long)))
(cffi:defcfun ("THShortTensor_resize1d" th-short-tensor-resize-1d) :void
  (tensor th-short-tensor-ptr)
  (size0 :long))
(cffi:defcfun ("THShortTensor_resize2d" th-short-tensor-resize-2d) :void
  (tensor th-short-tensor-ptr)
  (size0 :long)
  (size1 :long))
(cffi:defcfun ("THShortTensor_resize3d" th-short-tensor-resize-3d) :void
  (tensor th-short-tensor-ptr)
  (size0 :long)
  (size1 :long)
  (size2 :long))
(cffi:defcfun ("THShortTensor_resize4d" th-short-tensor-resize-4d) :void
  (tensor th-short-tensor-ptr)
  (size0 :long)
  (size1 :long)
  (size2 :long)
  (size3 :long))
(cffi:defcfun ("THShortTensor_resize5d" th-short-tensor-resize-5d) :void
  (tensor th-short-tensor-ptr)
  (size0 :long)
  (size1 :long)
  (size2 :long)
  (size3 :long)
  (size4 :long))

(cffi:defcfun ("THShortTensor_set" th-short-tensor-set) :void
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr))
(cffi:defcfun ("THShortTensor_setStorage" th-short-tensor-set-storage) :void
  (tensor th-short-tensor-ptr)
  (storage th-short-storage-ptr)
  (storage-offset :long-long)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
(cffi:defcfun ("THShortTensor_setStorageNd" th-short-tensor-set-storage-nd) :void
  (tensor th-short-tensor-ptr)
  (storage th-short-storage-ptr)
  (storage-offset :long-long)
  (dimension :int)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
(cffi:defcfun ("THShortTensor_setStorage1d" th-short-tensor-set-storage-1d) :void
  (tensor th-short-tensor-ptr)
  (storage th-short-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long))
(cffi:defcfun ("THShortTensor_setStorage2d" th-short-tensor-set-storage-2d) :void
  (tensor th-short-tensor-ptr)
  (storage th-short-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long))
(cffi:defcfun ("THShortTensor_setStorage3d" th-short-tensor-set-storage-3d) :void
  (tensor th-short-tensor-ptr)
  (storage th-short-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long)
  (size2 :long)
  (stride2 :long))
(cffi:defcfun ("THShortTensor_setStorage4d" th-short-tensor-set-storage-4d) :void
  (tensor th-short-tensor-ptr)
  (storage th-short-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long)
  (size2 :long)
  (stride2 :long)
  (size3 :long)
  (stride3 :long))

(cffi:defcfun ("THShortTensor_narrow" th-short-tensor-narrow) :void
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr)
  (dimension :int)
  (first-index :long)
  (size :long))
(cffi:defcfun ("THShortTensor_select" th-short-tensor-select) :void
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr)
  (dimension :int)
  (slice-index :long))
(cffi:defcfun ("THShortTensor_transpose" th-short-tensor-transpose) :void
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr)
  (dimension1 :int)
  (dimension2 :int))
(cffi:defcfun ("THShortTensor_unfold" th-short-tensor-unfold) :void
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr)
  (dimension :int)
  (size :long)
  (step :long))
(cffi:defcfun ("THShortTensor_squeeze" th-short-tensor-squeeze) :void
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr))
(cffi:defcfun ("THShortTensor_squeeze1d" th-short-tensor-squeeze-1d) :void
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr)
  (dimension :int))
(cffi:defcfun ("THShortTensor_unsqueeze1d" th-short-tensor-unsqueeze-1d) :void
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr)
  (dimension :int))

(cffi:defcfun ("THShortTensor_isContiguous" th-short-tensor-is-contiguous) :int
  (tensor th-short-tensor-ptr))
(cffi:defcfun ("THShortTensor_isSameSizeAs" th-short-tensor-is-same-size-as) :int
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr))
(cffi:defcfun ("THShortTensor_isSetTo" th-short-tensor-is-set-to) :int
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr))
(cffi:defcfun ("THShortTensor_isSize" th-short-tensor-is-size) :int
  (tensor th-short-tensor-ptr)
  (dims th-long-storage-ptr))
(cffi:defcfun ("THShortTensor_nElement" th-short-tensor-n-element) :long-long
  (tensor th-short-tensor-ptr))

(cffi:defcfun ("THShortTensor_retain" th-short-tensor-retain) :void
  (tensor th-short-tensor-ptr))
(cffi:defcfun ("THShortTensor_free" th-short-tensor-free) :void
  (tensor th-short-tensor-ptr))
(cffi:defcfun ("THShortTensor_freeCopyTo" th-short-tensor-free-copy-to) :void
  (source th-short-tensor-ptr)
  (target th-short-tensor-ptr))

;; slow access methods [check everything]
;; void THTensor_(set1d)(THTensor *tensor, long x0, real value);
(cffi:defcfun ("THShortTensor_set1d" th-short-tensor-set-1d) :void
  (tensor th-short-tensor-ptr)
  (index0 :long)
  (value :short))
;; void THTensor_(set2d)(THTensor *tensor, long x0, long x1, real value);
(cffi:defcfun ("THShortTensor_set2d" th-short-tensor-set-2d) :void
  (tensor th-short-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (value :short))
;; void THTensor_(set3d)(THTensor *tensor, long x0, long x1, long x2, real value);
(cffi:defcfun ("THShortTensor_set3d" th-short-tensor-set-3d) :void
  (tensor th-short-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long)
  (value :short))
;; void THTensor_(set4d)(THTensor *tensor, long x0, long x1, long x2, long x3, real value);
(cffi:defcfun ("THShortTensor_set4d" th-short-tensor-set-4d) :void
  (tensor th-short-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long)
  (index3 :long)
  (value :short))

(cffi:defcfun ("THShortTensor_get1d" th-short-tensor-get-1d) :short
  (tensor th-short-tensor-ptr)
  (index0 :long))
(cffi:defcfun ("THShortTensor_get2d" th-short-tensor-get-2d) :short
  (tensor th-short-tensor-ptr)
  (index0 :long)
  (index1 :long))
(cffi:defcfun ("THShortTensor_get3d" th-short-tensor-get-3d) :short
  (tensor th-short-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long))
(cffi:defcfun ("THShortTensor_get4d" th-short-tensor-get-4d) :short
  (tensor th-short-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long)
  (index3 :long))

;; support for copy betweeb different tensor types
;; void THTensor_(copy)(THTensor *tensor, THTensor *src);
(cffi:defcfun ("THShortTensor_copy" th-short-tensor-copy) :void
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(copyByte)(THTensor *tensor, struct THByteTensor *src);
(cffi:defcfun ("THShortTensor_copyByte" th-short-tensor-copy-byte) :void
  (tensor th-short-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(copyChar)(THTensor *tensor, struct THCharTensor *src);
(cffi:defcfun ("THShortTensor_copyChar" th-short-tensor-copy-char) :void
  (tensor th-short-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(copyShort)(THTensor *tensor, struct THShortTensor *src);
(cffi:defcfun ("THShortTensor_copyShort" th-short-tensor-copy-short) :void
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(copyInt)(THTensor *tensor, struct THIntTensor *src);
(cffi:defcfun ("THShortTensor_copyInt" th-short-tensor-copy-int) :void
  (tensor th-short-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(copyLong)(THTensor *tensor, struct THLongTensor *src);
(cffi:defcfun ("THShortTensor_copyLong" th-short-tensor-copy-long) :void
  (tensor th-short-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(copyFloat)(THTensor *tensor, struct THFloatTensor *src);
(cffi:defcfun ("THShortTensor_copyFloat" th-short-tensor-copy-float) :void
  (tensor th-short-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(copyDouble)(THTensor *tensor, struct THDoubleTensor *src);
(cffi:defcfun ("THShortTensor_copyDouble" th-short-tensor-copy-double) :void
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr))

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
(cffi:defcfun ("THShortTensor_random" th-short-tensor-random) :void
  (tensor th-short-tensor-ptr)
  (generator th-generator-ptr))
;; void THTensor_(clampedRandom)(THTensor *self, THGenerator *_generator, long min, long max)
(cffi:defcfun ("THShortTensor_clampedRandom" th-short-tensor-clamped-random) :void
  (tensor th-short-tensor-ptr)
  (genrator th-generator-ptr)
  (min :long)
  (max :long))
;; void THTensor_(cappedRandom)(THTensor *self, THGenerator *_generator, long max);
(cffi:defcfun ("THShortTensor_cappedRandom" th-short-tensor-capped-random) :void
  (tensor th-short-tensor-ptr)
  (generator th-generator-ptr)
  (max :long))
;; void THTensor_(geometric)(THTensor *self, THGenerator *_generator, double p);
(cffi:defcfun ("THShortTensor_geometric" th-short-tensor-geometric) :void
  (tensor th-short-tensor-ptr)
  (generator th-generator-ptr)
  (p :double))
;; void THTensor_(bernoulli)(THTensor *self, THGenerator *_generator, double p);
(cffi:defcfun ("THShortTensor_bernoulli" th-short-tensor-bernoulli) :void
  (tensor th-short-tensor-ptr)
  (generator th-generator-ptr)
  (p :double))
;; void THTensor_(bernoulli_FloatTensor)(THTensor *self, THGenerator *_generator, THFloatTensor *p);
(cffi:defcfun ("THShortTensor_bernoulli_FloatTensor" th-short-tensor-bernoulli-float-tensor) :void
  (tensor th-short-tensor-ptr)
  (generator th-generator-ptr)
  (p th-float-tensor-ptr))
;; void THTensor_(bernoulli_DoubleTensor)(THTensor *self, THGenerator *_generator, THDoubleTensor *p);
(cffi:defcfun ("THShortTensor_bernoulli_DoubleTensor" th-short-tensor-bernoulli-double-tensor)
    :void
  (tensor th-short-tensor-ptr)
  (generator th-generator-ptr)
  (p th-short-tensor-ptr))

;; void THTensor_(fill)(THTensor *r_, real value);
(cffi:defcfun ("THShortTensor_fill" th-short-tensor-fill) :void
  (tensor th-short-tensor-ptr)
  (value :short))
;; void THTensor_(zero)(THTensor *r_);
(cffi:defcfun ("THShortTensor_zero" th-short-tensor-zero) :void
  (tensor th-short-tensor-ptr))

;; void THTensor_(maskedFill)(THTensor *tensor, THByteTensor *mask, real value);
(cffi:defcfun ("THShortTensor_maskedFill" th-short-tensor-masked-fill) :void
  (tensor th-short-tensor-ptr)
  (mask th-byte-tensor-ptr)
  (value :short))
;; void THTensor_(maskedCopy)(THTensor *tensor, THByteTensor *mask, THTensor* src);
(cffi:defcfun ("THShortTensor_maskedCopy" th-short-tensor-masked-copy) :void
  (tensor th-short-tensor-ptr)
  (mask th-byte-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(maskedSelect)(THTensor *tensor, THTensor* src, THByteTensor *mask);
(cffi:defcfun ("THShortTensor_maskedSelect" th-short-tensor-masked-select) :void
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr)
  (mask (th-byte-tensor-ptr)))

;; void THTensor_(nonzero)(THLongTensor *subscript, THTensor *tensor);
(cffi:defcfun ("THShortTensor_nonzero" th-short-tensor-nonzero) :void
  (subscript th-long-tensor-ptr)
  (tensor th-short-tensor-ptr))

;; void THTensor_(indexSelect)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index)
(cffi:defcfun ("THShortTensor_indexSelect" th-short-tensor-index-select) :void
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr))
;; void THTensor_(indexCopy)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THShortTensor_indexCopy" th-short-tensor-index-copy) :void
  (tensor th-short-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(indexAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THShortTensor_indexAdd" th-short-tensor-index-add) :void
  (tensor th-short-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(indexFill)(THTensor *tensor, int dim, THLongTensor *index, real val);
(cffi:defcfun ("THShortTensor_indexFill" th-short-tensor-index-fill) :void
  (tensor th-short-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (value :short))

;; void THTensor_(gather)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index);
(cffi:defcfun ("THShortTensor_gather" th-short-tensor-gather) :void
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr))
;; void THTensor_(scatter)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THShortTensor_scatter" th-short-tensor-scatter) :void
  (tensor th-short-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(scatterAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THShortTensor_scatterAdd" th-short-tensor-scatter-add) :void
  (tensor th-short-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(scatterFill)(THTensor *tensor, int dim, THLongTensor *index, real val);
(cffi:defcfun ("THShortTensor_scatterFill" th-short-tensor-scatter-fill) :void
  (tensor th-short-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (value :short))

;; accreal THTensor_(dot)(THTensor *t, THTensor *src);
(cffi:defcfun ("THShortTensor_dot" th-short-tensor-dot) :long
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr))

;; real THTensor_(minall)(THTensor *t);
(cffi:defcfun ("THShortTensor_minall" th-short-tensor-min-all) :short
  (tensor th-short-tensor-ptr))
;; real THTensor_(maxall)(THTensor *t);
(cffi:defcfun ("THShortTensor_maxall" th-short-tensor-max-all) :short
  (tensor th-short-tensor-ptr))
;; real THTensor_(medianall)(THTensor *t);
(cffi:defcfun ("THShortTensor_medianall" th-short-tensor-median-all) :short
  (tensor th-short-tensor-ptr))
;; accreal THTensor_(sumall)(THTensor *t);
(cffi:defcfun ("THShortTensor_sumall" th-short-tensor-sum-all) :long
  (tensor th-short-tensor-ptr))
;; accreal THTensor_(prodall)(THTensor *t);
(cffi:defcfun ("THShortTensor_prodall" th-short-tensor-prod-all) :long
  (tensor th-short-tensor-ptr))

;; void THTensor_(neg)(THTensor *self, THTensor *src);
(cffi:defcfun ("THShortTensor_neg" th-short-tensor-neg) :void
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr))

;; void THTensor_(add)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THShortTensor_add" th-short-tensor-add) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))
;; void THTensor_(sub)(THTensor *self, THTensor *src, real value);
(cffi:defcfun ("THShortTensor_sub" th-short-tensor-sub) :void
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr)
  (value :short))
;; void THTensor_(mul)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THShortTensor_mul" th-short-tensor-mul) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))
;; void THTensor_(div)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THShortTensor_div" th-short-tensor-div) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))
;; void THTensor_(lshift)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THShortTensor_lshift" th-short-tensor-lshift) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))
;; void THTensor_(rshift)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THShortTensor_rshift" th-short-tensor-rshift) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))
;; void THTensor_(fmod)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THShortTensor_fmod" th-short-tensor-fmod) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))
;; void THTensor_(remainder)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THShortTensor_remainder" th-short-tensor-remainder) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))
;; void THTensor_(clamp)(THTensor *r_, THTensor *t, real min_value, real max_value);
(cffi:defcfun ("THShortTensor_clamp" th-short-tensor-clamp) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (min-value :short)
  (max-value :short))
;; void THTensor_(bitand)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THShortTensor_bitand" th-short-tensor-bitand) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))
;; void THTensor_(bitor)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THShortTensor_bitor" th-short-tensor-bitor) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))
;; void THTensor_(bitxor)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THShortTensor_bitxor" th-short-tensor-bitxor) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))

;; void THTensor_(cadd)(THTensor *r_, THTensor *t, real value, THTensor *src);
(cffi:defcfun ("THShortTensor_cadd" th-short-tensor-cadd) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short)
  (src th-short-tensor-ptr))
;; void THTensor_(csub)(THTensor *self, THTensor *src1, real value, THTensor *src2);
(cffi:defcfun ("THShortTensor_csub" th-short-tensor-csub) :void
  (tensor th-short-tensor-ptr)
  (src1 th-short-tensor-ptr)
  (value :short)
  (src2 th-short-tensor-ptr))
;; void THTensor_(cmul)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THShortTensor_cmul" th-short-tensor-cmul) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(cpow)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THShortTensor_cpow" th-short-tensor-cpow) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(cdiv)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THShortTensor_cdiv" th-short-tensor-cdiv) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(clshift)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THShortTensor_clshift" th-short-tensor-clshift) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(crshift)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THShortTensor_crshift" th-short-tensor-crshift) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(cfmod)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THShortTensor_cfmod" th-short-tensor-cfmod) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(cremainder)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THShortTensor_cremainder" th-short-tensor-cremainder) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(cbitand)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THShortTensor_cbitand" th-short-tensor-cbitand) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(cbitor)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THShortTensor_cbitor" th-short-tensor-cbitor) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(cbitxor)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THShortTensor_cbitxor" th-short-tensor-cbitxor) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr))

;; void THTensor_(addcmul)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2);
(cffi:defcfun ("THShortTensor_addcmul" th-short-tensor-add-cmul) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short)
  (src1 th-short-tensor-ptr)
  (src2 th-short-tensor-ptr))
;; void THTensor_(addcdiv)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2);
(cffi:defcfun ("THShortTensor_addcdiv" th-short-tensor-add-cdiv) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short)
  (src1 th-short-tensor-ptr)
  (src2 th-short-tensor-ptr))
;; void THTensor_(addmv)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat,  THTensor *vec);
(cffi:defcfun ("THShortTensor_addmv" th-short-tensor-add-mv) :void
  (result th-short-tensor-ptr)
  (beta :short)
  (tensor th-short-tensor-ptr)
  (alpha :short)
  (matrix th-short-tensor-ptr)
  (vector th-short-tensor-ptr))
;; void THTensor_(addmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat1, THTensor *mat2);
(cffi:defcfun ("THShortTensor_addmm" th-short-tensor-add-mm) :void
  (result th-short-tensor-ptr)
  (beta :short)
  (tensor th-short-tensor-ptr)
  (alpha :short)
  (matrix1 th-short-tensor-ptr)
  (matrix2 th-short-tensor-ptr))
;; void THTensor_(addr)(THTensor *r_,  real beta, THTensor *t, real alpha, THTensor *vec1, THTensor *vec2);
(cffi:defcfun ("THShortTensor_addr" th-short-tensor-add-r) :void
  (result th-short-tensor-ptr)
  (beta :short)
  (tensor th-short-tensor-ptr)
  (alpha :short)
  (vector1 th-short-tensor-ptr)
  (vector2 th-short-tensor-ptr))
;; void THTensor_(addbmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *batch1, THTensor *batch2);
(cffi:defcfun ("THShortTensor_addbmm" th-short-tensor-add-bmm) :void
  (result th-short-tensor-ptr)
  (beta :short)
  (tensor th-short-tensor-ptr)
  (alpha :short)
  (batch1 th-short-tensor-ptr)
  (batch2 th-short-tensor-ptr))
;; void THTensor_(baddbmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *batch1, THTensor *batch2);
(cffi:defcfun ("THShortTensor_baddbmm" th-short-tensor-badd-bmm) :void
  (result th-short-tensor-ptr)
  (beta :short)
  (tensor th-short-tensor-ptr)
  (alpha :short)
  (batch1 th-short-tensor-ptr)
  (batch2 th-short-tensor-ptr))

;; void THTensor_(match)(THTensor *r_, THTensor *m1, THTensor *m2, real gain);
(cffi:defcfun ("THShortTensor_match" th-short-tensor-match) :void
  (result th-short-tensor-ptr)
  (m1 th-short-tensor-ptr)
  (m2 th-short-tensor-ptr)
  (gain :short))

;; ptrdiff_t THTensor_(numel)(THTensor *t);
(cffi:defcfun ("THShortTensor_numel" th-short-tensor-numel) :long-long
  (tensor th-short-tensor-ptr))
;; void THTensor_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THShortTensor_max" th-short-tensor-max) :void
  (values th-short-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THShortTensor_min" th-short-tensor-min) :void
  (values th-short-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(kthvalue)(THTensor *values_, THLongTensor *indices_, THTensor *t, long k, int dimension, int keepdim);
(cffi:defcfun ("THShortTensor_kthvalue" th-short-tensor-kth-value) :void
  (values th-short-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (k :long)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(mode)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THShortTensor_mode" th-short-tensor-mode) :void
  (values th-short-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(median)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THShortTensor_median" th-short-tensor-median) :void
  (values th-short-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(sum)(THTensor *r_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THShortTensor_sum" th-short-tensor-sum) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(prod)(THTensor *r_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THShortTensor_prod" th-short-tensor-prod) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(cumsum)(THTensor *r_, THTensor *t, int dimension);
(cffi:defcfun ("THShortTensor_cumsum" th-short-tensor-cum-sum) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (dimension :int))
;; void THTensor_(cumprod)(THTensor *r_, THTensor *t, int dimension);
(cffi:defcfun ("THShortTensor_cumprod" th-short-tensor-cum-prod) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (dimension :int))
;; void THTensor_(sign)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THShortTensor_sign" th-short-tensor-sign) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr))
;; accreal THTensor_(trace)(THTensor *t);
(cffi:defcfun ("THShortTensor_trace" th-short-tensor-trace) :long
  (tensor th-short-tensor-ptr))
;; void THTensor_(cross)(THTensor *r_, THTensor *a, THTensor *b, int dimension);
(cffi:defcfun ("THShortTensor_cross" th-short-tensor-cross) :void
  (result th-short-tensor-ptr)
  (a th-short-tensor-ptr)
  (b th-short-tensor-ptr)
  (dimension :int))

;; void THTensor_(cmax)(THTensor *r, THTensor *t, THTensor *src);
(cffi:defcfun ("THShortTensor_cmax" th-short-tensor-cmax) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(cmin)(THTensor *r, THTensor *t, THTensor *src);
(cffi:defcfun ("THShortTensor_cmin" th-short-tensor-cmin) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(cmaxValue)(THTensor *r, THTensor *t, real value);
(cffi:defcfun ("THShortTensor_cmaxValue" th-short-tensor-cmax-value) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))
;; void THTensor_(cminValue)(THTensor *r, THTensor *t, real value);
(cffi:defcfun ("THShortTensor_cminValue" th-short-tensor-cmin-value) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))

;; void THTensor_(zeros)(THTensor *r_, THLongStorage *size);
(cffi:defcfun ("THShortTensor_zeros" th-short-tensor-zeros) :void
  (result th-short-tensor-ptr)
  (size th-long-storage-ptr))
;; void THTensor_(zerosLike)(THTensor *r_, THTensor *input);
(cffi:defcfun ("THShortTensor_zerosLike" th-short-tensor-zero-like) :void
  (result th-short-tensor-ptr)
  (input th-short-tensor-ptr))
;; void THTensor_(ones)(THTensor *r_, THLongStorage *size);
(cffi:defcfun ("THShortTensor_ones" th-short-tensor-ones) :void
  (result th-short-tensor-ptr)
  (size th-long-storage-ptr))
;; void THTensor_(onesLike)(THTensor *r_, THTensor *input);
(cffi:defcfun ("THShortTensor_onesLike" th-short-tensor-one-like) :void
  (result th-short-tensor-ptr)
  (input th-short-tensor-ptr))
;; void THTensor_(diag)(THTensor *r_, THTensor *t, int k);
(cffi:defcfun ("THShortTensor_diag" th-short-tensor-diag) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (k :int))
;; void THTensor_(eye)(THTensor *r_, long n, long m);
(cffi:defcfun ("THShortTensor_eye" th-short-tensor-eye) :void
  (result th-short-tensor-ptr)
  (n :long)
  (m :long))
;; void THTensor_(arange)(THTensor *r_, accreal xmin, accreal xmax, accreal step);
(cffi:defcfun ("THShortTensor_arange" th-short-tensor-arange) :void
  (result th-short-tensor-ptr)
  (xmin :long)
  (xmax :long)
  (step :long))
;; void THTensor_(range)(THTensor *r_, accreal xmin, accreal xmax, accreal step);
(cffi:defcfun ("THShortTensor_range" th-short-tensor-range) :void
  (result th-short-tensor-ptr)
  (xmin :long)
  (xmax :long)
  (step :long))
;; void THTensor_(randperm)(THTensor *r_, THGenerator *_generator, long n);
(cffi:defcfun ("THShortTensor_randperm" th-short-tensor-rand-perm) :void
  (result th-short-tensor-ptr)
  (generator th-generator-ptr)
  (n :long))

;; void THTensor_(reshape)(THTensor *r_, THTensor *t, THLongStorage *size);
(cffi:defcfun ("THShortTensor_reshape" th-short-tensor-reshape) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (size th-long-storage-ptr))
;; void THTensor_(sort)(THTensor *rt_, THLongTensor *ri_, THTensor *t, int dimension, int descendingOrder);
(cffi:defcfun ("THShortTensor_sort" th-short-tensor-sort) :void
  (result-tensor th-short-tensor-ptr)
  (result-indices th-long-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (dimension :int)
  (discending-order :int))
;; void THTensor_(topk)(THTensor *rt_, THLongTensor *ri_, THTensor *t, long k, int dim, int dir, int sorted);
(cffi:defcfun ("THShortTensor_topk" th-short-tensor-topk) :void
  (result-tensor th-short-tensor-ptr)
  (result-indices th-long-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (k :long)
  (dim :int)
  (dir :int)
  (sorted :int))
;; void THTensor_(tril)(THTensor *r_, THTensor *t, long k);
(cffi:defcfun ("THShortTensor_tril" th-short-tensor-tril) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (k :long))
;; void THTensor_(triu)(THTensor *r_, THTensor *t, long k);
(cffi:defcfun ("THShortTensor_triu" th-short-tensor-triu) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (k :long))
;; void THTensor_(cat)(THTensor *r_, THTensor *ta, THTensor *tb, int dimension);
(cffi:defcfun ("THShortTensor_cat" th-short-tensor-cat) :void
  (result th-short-tensor-ptr)
  (tensora th-short-tensor-ptr)
  (tensorb th-short-tensor-ptr)
  (dimension :int))
;; void THTensor_(catArray)(THTensor *result, THTensor **inputs, int numInputs, int dimension);
(cffi:defcfun ("THShortTensor_catArray" th-short-tensor-cat-array) :void
  (result th-short-tensor-ptr)
  (inputs (:pointer th-short-tensor-ptr))
  (num-inputs :int)
  (dimension :int))

;; int THTensor_(equal)(THTensor *ta, THTensor *tb);
(cffi:defcfun ("THShortTensor_equal" th-short-tensor-equal) :int
  (tensora th-short-tensor-ptr)
  (tensorb th-short-tensor-ptr))

;; void THTensor_(ltValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THShortTensor_ltValue" th-short-tensor-lt-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))
;; void THTensor_(leValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THShortTensor_leValue" th-short-tensor-le-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))
;; void THTensor_(gtValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THShortTensor_gtValue" th-short-tensor-gt-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))
;; void THTensor_(geValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THShortTensor_geValue" th-short-tensor-ge-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))
;; void THTensor_(neValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THShortTensor_neValue" th-short-tensor-ne-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))
;; void THTensor_(eqValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THShortTensor_eqValue" th-short-tensor-eq-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))

;; void THTensor_(ltValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THShortTensor_ltValueT" th-short-tensor-lt-value-t) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))
;; void THTensor_(leValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THShortTensor_leValueT" th-short-tensor-le-value-t) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))
;; void THTensor_(gtValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THShortTensor_gtValueT" th-short-tensor-gt-value-t) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))
;; void THTensor_(geValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THShortTensor_geValueT" th-short-tensor-ge-value-t) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))
;; void THTensor_(neValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THShortTensor_neValueT" th-short-tensor-ne-value-t) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))
;; void THTensor_(eqValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THShortTensor_eqValueT" th-short-tensor-eq-value-t) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr)
  (value :short))

;; void THTensor_(ltTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THShortTensor_ltTensor" th-short-tensor-lt-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-short-tensor-ptr)
  (tensorb th-short-tensor-ptr))
;; void THTensor_(leTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THShortTensor_leTensor" th-short-tensor-le-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-short-tensor-ptr)
  (tensorb th-short-tensor-ptr))
;; void THTensor_(gtTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THShortTensor_gtTensor" th-short-tensor-gt-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-short-tensor-ptr)
  (tensorb th-short-tensor-ptr))
;; void THTensor_(geTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THShortTensor_geTensor" th-short-tensor-ge-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-short-tensor-ptr)
  (tensorb th-short-tensor-ptr))
;; void THTensor_(neTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THShortTensor_neTensor" th-short-tensor-ne-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-short-tensor-ptr)
  (tensorb th-short-tensor-ptr))
;; void THTensor_(eqTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THShortTensor_eqTensor" th-short-tensor-eq-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-short-tensor-ptr)
  (tensorb th-short-tensor-ptr))

;; void THTensor_(ltTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THShortTensor_ltTensorT" th-short-tensor-lt-tensor-t) :void
  (result th-short-tensor-ptr)
  (tensora th-short-tensor-ptr)
  (tensorb th-short-tensor-ptr))
;; void THTensor_(leTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THShortTensor_leTensorT" th-short-tensor-le-tensor-t) :void
  (result th-short-tensor-ptr)
  (tensora th-short-tensor-ptr)
  (tensorb th-short-tensor-ptr))
;; void THTensor_(gtTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THShortTensor_gtTensorT" th-short-tensor-gt-tensor-t) :void
  (result th-short-tensor-ptr)
  (tensora th-short-tensor-ptr)
  (tensorb th-short-tensor-ptr))
;; void THTensor_(geTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THShortTensor_geTensorT" th-short-tensor-ge-tensor-t) :void
  (result th-short-tensor-ptr)
  (tensora th-short-tensor-ptr)
  (tensorb th-short-tensor-ptr))
;; void THTensor_(neTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THShortTensor_neTensorT" th-short-tensor-ne-tensor-t) :void
  (result th-short-tensor-ptr)
  (tensora th-short-tensor-ptr)
  (tensorb th-short-tensor-ptr))
;; void THTensor_(eqTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THShortTensor_eqTensorT" th-short-tensor-eq-tensor-t) :void
  (result th-short-tensor-ptr)
  (tensora th-short-tensor-ptr)
  (tensorb th-short-tensor-ptr))

;; void THTensor_(abs)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THShortTensor_abs" th-short-tensor-abs) :void
  (result th-short-tensor-ptr)
  (tensor th-short-tensor-ptr))

;; void THTensor_(validXCorr2Dptr)(real *r_,
;;                                 real alpha,
;;                                 real *t_, long ir, long ic,
;;                                 real *k_, long kr, long kc,
;;                                 long sr, long sc);
(cffi:defcfun ("THShortTensor_validXCorr2Dptr" th-short-tensor-valid-x-corr-2d-ptr) :void
  (res (:pointer :short))
  (alpha :short)
  (ten (:pointer :short))
  (ir :long)
  (ic :long)
  (k (:pointer :short))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(validConv2Dptr)(real *r_,
;;                                real alpha,
;;                                real *t_, long ir, long ic,
;;                                real *k_, long kr, long kc,
;;                                long sr, long sc);
(cffi:defcfun ("THShortTensor_validConv2Dptr" th-short-tensor-valid-conv-2d-ptr) :void
  (res (:pointer :short))
  (alpha :short)
  (ten (:pointer :short))
  (ir :long)
  (ic :long)
  (k (:pointer :short))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(fullXCorr2Dptr)(real *r_,
;;                                real alpha,
;;                                real *t_, long ir, long ic,
;;                                real *k_, long kr, long kc,
;;                                long sr, long sc);
(cffi:defcfun ("THShortTensor_fullXCorr2Dptr" th-short-tensor-full-x-corr-2d-ptr) :void
  (res (:pointer :short))
  (alpha :short)
  (ten (:pointer :short))
  (ir :long)
  (ic :long)
  (k (:pointer :short))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(fullConv2Dptr)(real *r_,
;;                               real alpha,
;;                               real *t_, long ir, long ic,
;;                               real *k_, long kr, long kc,
;;                               long sr, long sc);
(cffi:defcfun ("THShortTensor_fullConv2Dptr" th-short-tensor-full-conv-2d-ptr) :void
  (res (:pointer :short))
  (alpha :short)
  (ten (:pointer :short))
  (ir :long)
  (ic :long)
  (k (:pointer :short))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(validXCorr2DRevptr)(real *r_,
;;                                    real alpha,
;;                                    real *t_, long ir, long ic,
;;                                    real *k_, long kr, long kc,
;;                                    long sr, long sc);
(cffi:defcfun ("THShortTensor_validXCorr2DRevptr" th-short-tensor-valid-x-corr-2d-rev-ptr) :void
  (res (:pointer :short))
  (alpha :short)
  (ten (:pointer :short))
  (ir :long)
  (ic :long)
  (k (:pointer :short))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(conv2DRevger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol);
(cffi:defcfun ("THShortTensor_conv2DRevger" th-short-tensor-conv-2d-rev-ger) :void
  (result th-short-tensor-ptr)
  (beta :short)
  (alpha :short)
  (tensor th-short-tensor-ptr)
  (k th-short-tensor-ptr)
  (srow :long)
  (scol :long))
;; void THTensor_(conv2DRevgerm)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol);
(cffi:defcfun ("THShortTensor_conv2DRevgerm" th-short-tensor-conv-2d-rev-germ) :void
  (result th-short-tensor-ptr)
  (beta :short)
  (alpha :short)
  (tensor th-short-tensor-ptr)
  (k th-short-tensor-ptr)
  (srow :long)
  (scol :long))
;; void THTensor_(conv2Dger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THShortTensor_conv2Dger" th-short-tensor-conv-2d-ger) :void
  (result th-short-tensor-ptr)
  (beta :short)
  (alpha :short)
  (tensor th-short-tensor-ptr)
  (k th-short-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv2Dmv)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THShortTensor_conv2Dmv" th-short-tensor-conv-2d-mv) :void
  (result th-short-tensor-ptr)
  (beta :short)
  (alpha :short)
  (tensor th-short-tensor-ptr)
  (k th-short-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv2Dmm)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THShortTensor_conv2Dmm" th-short-tensor-conv-2d-mm) :void
  (result th-short-tensor-ptr)
  (beta :short)
  (alpha :short)
  (tensor th-short-tensor-ptr)
  (k th-short-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv2Dmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THShortTensor_conv2Dmul" th-short-tensor-conv-2d-mul) :void
  (result th-short-tensor-ptr)
  (beta :short)
  (alpha :short)
  (tensor th-short-tensor-ptr)
  (k th-short-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv2Dcmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THShortTensor_conv2Dcmul" th-short-tensor-conv-2d-cmul) :void
  (result th-short-tensor-ptr)
  (beta :short)
  (alpha :short)
  (tensor th-short-tensor-ptr)
  (k th-short-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))

;; void THTensor_(validXCorr3Dptr)(real *r_,
;;                                 real alpha,
;;                                 real *t_, long it, long ir, long ic,
;;                                 real *k_, long kt, long kr, long kc,
;;                                 long st, long sr, long sc);
(cffi:defcfun ("THShortTensor_validXCorr3Dptr" th-short-tensor-valid-x-corr-3d-ptr) :void
  (res (:pointer :short))
  (alpha :short)
  (ten (:pointer :short))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :short))
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
(cffi:defcfun ("THShortTensor_validConv3Dptr" th-short-tensor-valid-conv-3d-ptr) :void
  (res (:pointer :short))
  (alpha :short)
  (ten (:pointer :short))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :short))
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
(cffi:defcfun ("THShortTensor_fullXCorr3Dptr" th-short-tensor-full-x-corr-3d-ptr) :void
  (res (:pointer :short))
  (alpha :short)
  (ten (:pointer :short))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :short))
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
(cffi:defcfun ("THShortTensor_fullConv3Dptr" th-short-tensor-full-conv-3d-ptr) :void
  (res (:pointer :short))
  (alpha :short)
  (ten (:pointer :short))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :short))
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
(cffi:defcfun ("THShortTensor_validXCorr3DRevptr" th-short-tensor-valid-x-corr-3d-rev-ptr) :void
  (res (:pointer :short))
  (alpha :short)
  (ten (:pointer :short))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :short))
  (kt :long)
  (kr :long)
  (kc :long)
  (st :long)
  (sr :long)
  (sc :long))

;; void THTensor_(conv3DRevger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol);
(cffi:defcfun ("THShortTensor_conv3DRevger" th-short-tensor-conv-3d-rev-ger) :void
  (result th-short-tensor-ptr)
  (beta :short)
  (alpha :short)
  (tensor th-short-tensor-ptr)
  (k th-short-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long))
;; void THTensor_(conv3Dger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THShortTensor_conv3Dger" th-short-tensor-conv-3d-ger) :void
  (result th-short-tensor-ptr)
  (beta :short)
  (alpha :short)
  (tensor th-short-tensor-ptr)
  (k th-short-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv3Dmv)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THShortTensor_conv3Dmv" th-short-tensor-conv-3d-mv) :void
  (result th-short-tensor-ptr)
  (beta :short)
  (alpha :short)
  (tensor th-short-tensor-ptr)
  (k th-short-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv3Dmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THShortTensor_conv3Dmul" th-short-tensor-conv-3d-mul) :void
  (result th-short-tensor-ptr)
  (beta :short)
  (alpha :short)
  (tensor th-short-tensor-ptr)
  (k th-short-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv3Dcmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THShortTensor_conv3Dcmul" th-short-tensor-conv-3d-cmul) :void
  (result th-short-tensor-ptr)
  (beta :short)
  (alpha :short)
  (tensor th-short-tensor-ptr)
  (k th-short-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
