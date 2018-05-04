(in-package :th)

;; ACCESS METHODS
;; THStorage* THTensor_(storage)(const THTensor *self)
(cffi:defcfun ("THCharTensor_storage" th-char-tensor-storage) th-char-storage-ptr
  (tensor th-char-tensor-ptr))
;; ptrdiff_t THTensor_(storageOffset)(const THTensor *self)
(cffi:defcfun ("THCharTensor_storageOffset" th-char-tensor-storage-offset) :long-long
  (tensor th-char-tensor-ptr))
;; int THTensor_(nDimension)(const THTensor *self)
(cffi:defcfun ("THCharTensor_nDimension" th-char-tensor-n-dimension) :int
  (tensor th-char-tensor-ptr))
;; long THTensor_(size)(const THTensor *self, int dim)
(cffi:defcfun ("THCharTensor_size" th-char-tensor-size) :long
  (tensor th-char-tensor-ptr)
  (dim :int))
;; long THTensor_(stride)(const THTensor *self, int dim)
(cffi:defcfun ("THCharTensor_stride" th-char-tensor-stride) :long
  (tensor th-char-tensor-ptr)
  (dim :int))
;; THLongStorage *THTensor_(newSizeOf)(THTensor *self)
(cffi:defcfun ("THCharTensor_newSizeOf" th-char-tensor-new-size-of) th-long-storage-ptr
  (tensor th-char-tensor-ptr))
;; THLongStorage *THTensor_(newStrideOf)(THTensor *self)
(cffi:defcfun ("THCharTensor_newStrideOf" th-char-tensor-new-stride-of) th-long-storage-ptr
  (tensor th-char-tensor-ptr))
;; real *THTensor_(data)(const THTensor *self)
(cffi:defcfun ("THCharTensor_data" th-char-tensor-data) (:pointer :char)
  (tensor th-char-tensor-ptr))

;; void THTensor_(setFlag)(THTensor *self, const char flag)
(cffi:defcfun ("THCharTensor_setFlag" th-char-tensor-set-flag) :void
  (tensor th-char-tensor-ptr)
  (flag :char))
;; void THTensor_(clearFlag)(THTensor *self, const char flag)
(cffi:defcfun ("THCharTensor_clearFlag" th-char-tensor-clear-flag) :void
  (tensor th-char-tensor-ptr)
  (flag :char))

;; CREATION METHODS
;; THTensor *THTensor_(new)(void)
(cffi:defcfun ("THCharTensor_new" th-char-tensor-new) th-char-tensor-ptr)
;; THTensor *THTensor_(newWithTensor)(THTensor *tensor)
(cffi:defcfun ("THCharTensor_newWithTensor" th-char-tensor-new-with-tensor) th-char-tensor-ptr
  (tensor th-char-tensor-ptr))
;; stride might be NULL
;; THTensor *THTensor_(newWithStorage)(THStorage *storage_, ptrdiff_t storageOffset_,
;;                                            THLongStorage *size_, THLongStorage *stride_)
(cffi:defcfun ("THCharTensor_newWithStorage" th-char-tensor-new-with-storage)
    th-char-tensor-ptr
  (storage th-char-storage-ptr)
  (storage-offset :long-long)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
;; THTensor *THTensor_(newWithStorage1d)(THStorage *storage_, ptrdiff_t storageOffset_,
;;                                              long size0_, long stride0_);
(cffi:defcfun ("THCharTensor_newWithStorage1d" th-char-tensor-new-with-storage-1d)
    th-char-tensor-ptr
  (storage th-char-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long))
(cffi:defcfun ("THCharTensor_newWithStorage2d" th-char-tensor-new-with-storage-2d)
    th-char-tensor-ptr
  (storage th-char-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long))
(cffi:defcfun ("THCharTensor_newWithStorage3d" th-char-tensor-new-with-storage-3d)
    th-char-tensor-ptr
  (storage th-char-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long)
  (size2 :long)
  (stride2 :long))
(cffi:defcfun ("THCharTensor_newWithStorage4d" th-char-tensor-new-with-storage-4d)
    th-char-tensor-ptr
  (storage th-char-storage-ptr)
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
(cffi:defcfun ("THCharTensor_newWithSize" th-char-tensor-new-with-size) th-char-tensor-ptr
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
;; THTensor *THTensor_(newWithSize1d)(long size0_);
(cffi:defcfun ("THCharTensor_newWithSize1d" th-char-tensor-new-with-size-1d)
    th-char-tensor-ptr
  (size0 :long))
(cffi:defcfun ("THCharTensor_newWithSize2d" th-char-tensor-new-with-size-2d)
    th-char-tensor-ptr
  (size0 :long)
  (size1 :long))
(cffi:defcfun ("THCharTensor_newWithSize3d" th-char-tensor-new-with-size-3d)
    th-char-tensor-ptr
  (size0 :long)
  (size1 :long)
  (size2 :long))
(cffi:defcfun ("THCharTensor_newWithSize4d" th-char-tensor-new-with-size-4d)
    th-char-tensor-ptr
  (size0 :long)
  (size1 :long)
  (size2 :long)
  (size3 :long))

;; THTensor *THTensor_(newClone)(THTensor *self)
(cffi:defcfun ("THCharTensor_newClone" th-char-tensor-new-clone) th-char-tensor-ptr
  (tensor th-char-tensor-ptr))
(cffi:defcfun ("THCharTensor_newContiguous" th-char-tensor-new-contiguous) th-char-tensor-ptr
  (tensor th-char-tensor-ptr))
(cffi:defcfun ("THCharTensor_newSelect" th-char-tensor-new-select) th-char-tensor-ptr
  (tensor th-char-tensor-ptr)
  (dimension :int)
  (slice-index :long))
(cffi:defcfun ("THCharTensor_newNarrow" th-char-tensor-new-narrow) th-char-tensor-ptr
  (tensor th-char-tensor-ptr)
  (dimension :int)
  (first-index :long)
  (size :long))

(cffi:defcfun ("THCharTensor_newTranspose" th-char-tensor-new-transpose) th-char-tensor-ptr
  (tensor th-char-tensor-ptr)
  (dimension1 :int)
  (dimension2 :int))
(cffi:defcfun ("THCharTensor_newUnfold" th-char-tensor-new-unfold) th-char-tensor-ptr
  (tensor th-char-tensor-ptr)
  (dimension :int)
  (size :long)
  (step :long))
(cffi:defcfun ("THCharTensor_newView" th-char-tensor-new-view) th-char-tensor-ptr
  (tensor th-char-tensor-ptr)
  (size th-long-storage-ptr))
(cffi:defcfun ("THCharTensor_newExpand" th-char-tensor-new-expand) th-char-tensor-ptr
  (tensor th-char-tensor-ptr)
  (size th-long-storage-ptr))

(cffi:defcfun ("THCharTensor_expand" th-char-tensor-expand) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (size th-long-storage-ptr))
(cffi:defcfun ("THCharTensor_expandNd" th-char-tensor-expand-nd) :void
  (result (:pointer th-char-tensor-ptr))
  (ops (:pointer th-char-tensor-ptr))
  (count :int))

(cffi:defcfun ("THCharTensor_resize" th-char-tensor-resize) :void
  (tensor th-char-tensor-ptr)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
(cffi:defcfun ("THCharTensor_resizeAs" th-char-tensor-resize-as) :void
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr))
(cffi:defcfun ("THCharTensor_resizeNd" th-char-tensor-resize-nd) :void
  (tensor th-char-tensor-ptr)
  (dimension :int)
  (size (:pointer :long))
  (stride (:pointer :long)))
(cffi:defcfun ("THCharTensor_resize1d" th-char-tensor-resize-1d) :void
  (tensor th-char-tensor-ptr)
  (size0 :long))
(cffi:defcfun ("THCharTensor_resize2d" th-char-tensor-resize-2d) :void
  (tensor th-char-tensor-ptr)
  (size0 :long)
  (size1 :long))
(cffi:defcfun ("THCharTensor_resize3d" th-char-tensor-resize-3d) :void
  (tensor th-char-tensor-ptr)
  (size0 :long)
  (size1 :long)
  (size2 :long))
(cffi:defcfun ("THCharTensor_resize4d" th-char-tensor-resize-4d) :void
  (tensor th-char-tensor-ptr)
  (size0 :long)
  (size1 :long)
  (size2 :long)
  (size3 :long))
(cffi:defcfun ("THCharTensor_resize5d" th-char-tensor-resize-5d) :void
  (tensor th-char-tensor-ptr)
  (size0 :long)
  (size1 :long)
  (size2 :long)
  (size3 :long)
  (size4 :long))

(cffi:defcfun ("THCharTensor_set" th-char-tensor-set) :void
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr))
(cffi:defcfun ("THCharTensor_setStorage" th-char-tensor-set-storage) :void
  (tensor th-char-tensor-ptr)
  (storage th-char-storage-ptr)
  (storage-offset :long-long)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
(cffi:defcfun ("THCharTensor_setStorageNd" th-char-tensor-set-storage-nd) :void
  (tensor th-char-tensor-ptr)
  (storage th-char-storage-ptr)
  (storage-offset :long-long)
  (dimension :int)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
(cffi:defcfun ("THCharTensor_setStorage1d" th-char-tensor-set-storage-1d) :void
  (tensor th-char-tensor-ptr)
  (storage th-char-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long))
(cffi:defcfun ("THCharTensor_setStorage2d" th-char-tensor-set-storage-2d) :void
  (tensor th-char-tensor-ptr)
  (storage th-char-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long))
(cffi:defcfun ("THCharTensor_setStorage3d" th-char-tensor-set-storage-3d) :void
  (tensor th-char-tensor-ptr)
  (storage th-char-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long)
  (size2 :long)
  (stride2 :long))
(cffi:defcfun ("THCharTensor_setStorage4d" th-char-tensor-set-storage-4d) :void
  (tensor th-char-tensor-ptr)
  (storage th-char-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long)
  (size2 :long)
  (stride2 :long)
  (size3 :long)
  (stride3 :long))

(cffi:defcfun ("THCharTensor_narrow" th-char-tensor-narrow) :void
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr)
  (dimension :int)
  (first-index :long)
  (size :long))
(cffi:defcfun ("THCharTensor_select" th-char-tensor-select) :void
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr)
  (dimension :int)
  (slice-index :long))
(cffi:defcfun ("THCharTensor_transpose" th-char-tensor-transpose) :void
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr)
  (dimension1 :int)
  (dimension2 :int))
(cffi:defcfun ("THCharTensor_unfold" th-char-tensor-unfold) :void
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr)
  (dimension :int)
  (size :long)
  (step :long))
(cffi:defcfun ("THCharTensor_squeeze" th-char-tensor-squeeze) :void
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr))
(cffi:defcfun ("THCharTensor_squeeze1d" th-char-tensor-squeeze-1d) :void
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr)
  (dimension :int))
(cffi:defcfun ("THCharTensor_unsqueeze1d" th-char-tensor-unsqueeze-1d) :void
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr)
  (dimension :int))

(cffi:defcfun ("THCharTensor_isContiguous" th-char-tensor-is-contiguous) :int
  (tensor th-char-tensor-ptr))
(cffi:defcfun ("THCharTensor_isSameSizeAs" th-char-tensor-is-same-size-as) :int
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr))
(cffi:defcfun ("THCharTensor_isSetTo" th-char-tensor-is-set-to) :int
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr))
(cffi:defcfun ("THCharTensor_isSize" th-char-tensor-is-size) :int
  (tensor th-char-tensor-ptr)
  (dims th-long-storage-ptr))
(cffi:defcfun ("THCharTensor_nElement" th-char-tensor-n-element) :long-long
  (tensor th-char-tensor-ptr))

(cffi:defcfun ("THCharTensor_retain" th-char-tensor-retain) :void
  (tensor th-char-tensor-ptr))
(cffi:defcfun ("THCharTensor_free" th-char-tensor-free) :void
  (tensor th-char-tensor-ptr))
(cffi:defcfun ("THCharTensor_freeCopyTo" th-char-tensor-free-copy-to) :void
  (source th-char-tensor-ptr)
  (target th-char-tensor-ptr))

;; slow access methods [check everything]
;; void THTensor_(set1d)(THTensor *tensor, long x0, real value);
(cffi:defcfun ("THCharTensor_set1d" th-char-tensor-set-1d) :void
  (tensor th-char-tensor-ptr)
  (index0 :long)
  (value :char))
;; void THTensor_(set2d)(THTensor *tensor, long x0, long x1, real value);
(cffi:defcfun ("THCharTensor_set2d" th-char-tensor-set-2d) :void
  (tensor th-char-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (value :char))
;; void THTensor_(set3d)(THTensor *tensor, long x0, long x1, long x2, real value);
(cffi:defcfun ("THCharTensor_set3d" th-char-tensor-set-3d) :void
  (tensor th-char-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long)
  (value :char))
;; void THTensor_(set4d)(THTensor *tensor, long x0, long x1, long x2, long x3, real value);
(cffi:defcfun ("THCharTensor_set4d" th-char-tensor-set-4d) :void
  (tensor th-char-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long)
  (index3 :long)
  (value :char))

(cffi:defcfun ("THCharTensor_get1d" th-char-tensor-get-1d) :char
  (tensor th-char-tensor-ptr)
  (index0 :long))
(cffi:defcfun ("THCharTensor_get2d" th-char-tensor-get-2d) :char
  (tensor th-char-tensor-ptr)
  (index0 :long)
  (index1 :long))
(cffi:defcfun ("THCharTensor_get3d" th-char-tensor-get-3d) :char
  (tensor th-char-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long))
(cffi:defcfun ("THCharTensor_get4d" th-char-tensor-get-4d) :char
  (tensor th-char-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long)
  (index3 :long))

;; support for copy betweeb different tensor types
;; void THTensor_(copy)(THTensor *tensor, THTensor *src);
(cffi:defcfun ("THCharTensor_copy" th-char-tensor-copy) :void
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(copyByte)(THTensor *tensor, struct THByteTensor *src);
(cffi:defcfun ("THCharTensor_copyByte" th-char-tensor-copy-byte) :void
  (tensor th-char-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(copyChar)(THTensor *tensor, struct THCharTensor *src);
(cffi:defcfun ("THCharTensor_copyChar" th-char-tensor-copy-char) :void
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(copyShort)(THTensor *tensor, struct THShortTensor *src);
(cffi:defcfun ("THCharTensor_copyShort" th-char-tensor-copy-short) :void
  (tensor th-char-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(copyInt)(THTensor *tensor, struct THIntTensor *src);
(cffi:defcfun ("THCharTensor_copyInt" th-char-tensor-copy-int) :void
  (tensor th-char-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(copyLong)(THTensor *tensor, struct THLongTensor *src);
(cffi:defcfun ("THCharTensor_copyLong" th-char-tensor-copy-long) :void
  (tensor th-char-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(copyFloat)(THTensor *tensor, struct THFloatTensor *src);
(cffi:defcfun ("THCharTensor_copyFloat" th-char-tensor-copy-float) :void
  (tensor th-char-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(copyDouble)(THTensor *tensor, struct THDoubleTensor *src);
(cffi:defcfun ("THCharTensor_copyDouble" th-char-tensor-copy-double) :void
  (tensor th-char-tensor-ptr)
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
(cffi:defcfun ("THCharTensor_random" th-char-tensor-random) :void
  (tensor th-char-tensor-ptr)
  (generator th-generator-ptr))
;; void THTensor_(clampedRandom)(THTensor *self, THGenerator *_generator, long min, long max)
(cffi:defcfun ("THCharTensor_clampedRandom" th-char-tensor-clamped-random) :void
  (tensor th-char-tensor-ptr)
  (genrator th-generator-ptr)
  (min :long)
  (max :long))
;; void THTensor_(cappedRandom)(THTensor *self, THGenerator *_generator, long max);
(cffi:defcfun ("THCharTensor_cappedRandom" th-char-tensor-capped-random) :void
  (tensor th-char-tensor-ptr)
  (generator th-generator-ptr)
  (max :long))
;; void THTensor_(geometric)(THTensor *self, THGenerator *_generator, double p);
(cffi:defcfun ("THCharTensor_geometric" th-char-tensor-geometric) :void
  (tensor th-char-tensor-ptr)
  (generator th-generator-ptr)
  (p :double))
;; void THTensor_(bernoulli)(THTensor *self, THGenerator *_generator, double p);
(cffi:defcfun ("THCharTensor_bernoulli" th-char-tensor-bernoulli) :void
  (tensor th-char-tensor-ptr)
  (generator th-generator-ptr)
  (p :double))
;; void THTensor_(bernoulli_FloatTensor)(THTensor *self, THGenerator *_generator, THFloatTensor *p);
(cffi:defcfun ("THCharTensor_bernoulli_FloatTensor" th-char-tensor-bernoulli-float-tensor) :void
  (tensor th-char-tensor-ptr)
  (generator th-generator-ptr)
  (p th-float-tensor-ptr))
;; void THTensor_(bernoulli_DoubleTensor)(THTensor *self, THGenerator *_generator, THDoubleTensor *p);
(cffi:defcfun ("THCharTensor_bernoulli_DoubleTensor" th-char-tensor-bernoulli-double-tensor)
    :void
  (tensor th-char-tensor-ptr)
  (generator th-generator-ptr)
  (p th-double-tensor-ptr))

;; void THTensor_(fill)(THTensor *r_, real value);
(cffi:defcfun ("THCharTensor_fill" th-char-tensor-fill) :void
  (tensor th-char-tensor-ptr)
  (value :char))
;; void THTensor_(zero)(THTensor *r_);
(cffi:defcfun ("THCharTensor_zero" th-char-tensor-zero) :void
  (tensor th-char-tensor-ptr))

;; void THTensor_(maskedFill)(THTensor *tensor, THByteTensor *mask, real value);
(cffi:defcfun ("THCharTensor_maskedFill" th-char-tensor-masked-fill) :void
  (tensor th-char-tensor-ptr)
  (mask th-byte-tensor-ptr)
  (value :char))
;; void THTensor_(maskedCopy)(THTensor *tensor, THByteTensor *mask, THTensor* src);
(cffi:defcfun ("THCharTensor_maskedCopy" th-char-tensor-masked-copy) :void
  (tensor th-char-tensor-ptr)
  (mask th-byte-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(maskedSelect)(THTensor *tensor, THTensor* src, THByteTensor *mask);
(cffi:defcfun ("THCharTensor_maskedSelect" th-char-tensor-masked-select) :void
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr)
  (mask (th-byte-tensor-ptr)))

;; void THTensor_(nonzero)(THLongTensor *subscript, THTensor *tensor);
(cffi:defcfun ("THCharTensor_nonzero" th-char-tensor-nonzero) :void
  (subscript th-long-tensor-ptr)
  (tensor th-char-tensor-ptr))

;; void THTensor_(indexSelect)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index)
(cffi:defcfun ("THCharTensor_indexSelect" th-char-tensor-index-select) :void
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr))
;; void THTensor_(indexCopy)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THCharTensor_indexCopy" th-char-tensor-index-copy) :void
  (tensor th-char-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(indexAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THCharTensor_indexAdd" th-char-tensor-index-add) :void
  (tensor th-char-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(indexFill)(THTensor *tensor, int dim, THLongTensor *index, real val);
(cffi:defcfun ("THCharTensor_indexFill" th-char-tensor-index-fill) :void
  (tensor th-char-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (value :char))

;; void THTensor_(gather)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index);
(cffi:defcfun ("THCharTensor_gather" th-char-tensor-gather) :void
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr))
;; void THTensor_(scatter)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THCharTensor_scatter" th-char-tensor-scatter) :void
  (tensor th-char-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(scatterAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THCharTensor_scatterAdd" th-char-tensor-scatter-add) :void
  (tensor th-char-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(scatterFill)(THTensor *tensor, int dim, THLongTensor *index, real val);
(cffi:defcfun ("THCharTensor_scatterFill" th-char-tensor-scatter-fill) :void
  (tensor th-char-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (value :char))

;; accreal THTensor_(dot)(THTensor *t, THTensor *src);
(cffi:defcfun ("THCharTensor_dot" th-char-tensor-dot) :long
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr))

;; real THTensor_(minall)(THTensor *t);
(cffi:defcfun ("THCharTensor_minall" th-char-tensor-min-all) :char
  (tensor th-char-tensor-ptr))
;; real THTensor_(maxall)(THTensor *t);
(cffi:defcfun ("THCharTensor_maxall" th-char-tensor-max-all) :char
  (tensor th-char-tensor-ptr))
;; real THTensor_(medianall)(THTensor *t);
(cffi:defcfun ("THCharTensor_medianall" th-char-tensor-median-all) :char
  (tensor th-char-tensor-ptr))
;; accreal THTensor_(sumall)(THTensor *t);
(cffi:defcfun ("THCharTensor_sumall" th-char-tensor-sum-all) :long
  (tensor th-char-tensor-ptr))
;; accreal THTensor_(prodall)(THTensor *t);
(cffi:defcfun ("THCharTensor_prodall" th-char-tensor-prod-all) :long
  (tensor th-char-tensor-ptr))

;; void THTensor_(add)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THCharTensor_add" th-char-tensor-add) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))
;; void THTensor_(sub)(THTensor *self, THTensor *src, real value);
(cffi:defcfun ("THCharTensor_sub" th-char-tensor-sub) :void
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr)
  (value :char))
;; void THTensor_(mul)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THCharTensor_mul" th-char-tensor-mul) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))
;; void THTensor_(div)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THCharTensor_div" th-char-tensor-div) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))
;; void THTensor_(lshift)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THCharTensor_lshift" th-char-tensor-lshift) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))
;; void THTensor_(rshift)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THCharTensor_rshift" th-char-tensor-rshift) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))
;; void THTensor_(fmod)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THCharTensor_fmod" th-char-tensor-fmod) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))
;; void THTensor_(remainder)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THCharTensor_remainder" th-char-tensor-remainder) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))
;; void THTensor_(clamp)(THTensor *r_, THTensor *t, real min_value, real max_value);
(cffi:defcfun ("THCharTensor_clamp" th-char-tensor-clamp) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (min-value :char)
  (max-value :char))
;; void THTensor_(bitand)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THCharTensor_bitand" th-char-tensor-bitand) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))
;; void THTensor_(bitor)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THCharTensor_bitor" th-char-tensor-bitor) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))
;; void THTensor_(bitxor)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THCharTensor_bitxor" th-char-tensor-bitxor) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))

;; void THTensor_(cadd)(THTensor *r_, THTensor *t, real value, THTensor *src);
(cffi:defcfun ("THCharTensor_cadd" th-char-tensor-cadd) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char)
  (src th-char-tensor-ptr))
;; void THTensor_(csub)(THTensor *self, THTensor *src1, real value, THTensor *src2);
(cffi:defcfun ("THCharTensor_csub" th-char-tensor-csub) :void
  (tensor th-char-tensor-ptr)
  (src1 th-char-tensor-ptr)
  (value :char)
  (src2 th-char-tensor-ptr))
;; void THTensor_(cmul)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THCharTensor_cmul" th-char-tensor-cmul) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(cpow)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THCharTensor_cpow" th-char-tensor-cpow) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(cdiv)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THCharTensor_cdiv" th-char-tensor-cdiv) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(clshift)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THCharTensor_clshift" th-char-tensor-clshift) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(crshift)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THCharTensor_crshift" th-char-tensor-crshift) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(cfmod)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THCharTensor_cfmod" th-char-tensor-cfmod) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(cremainder)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THCharTensor_cremainder" th-char-tensor-cremainder) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(cbitand)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THCharTensor_cbitand" th-char-tensor-cbitand) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(cbitor)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THCharTensor_cbitor" th-char-tensor-cbitor) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(cbitxor)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THCharTensor_cbitxor" th-char-tensor-cbitxor) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr))

;; void THTensor_(addcmul)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2);
(cffi:defcfun ("THCharTensor_addcmul" th-char-tensor-add-cmul) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char)
  (src1 th-char-tensor-ptr)
  (src2 th-char-tensor-ptr))
;; void THTensor_(addcdiv)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2);
(cffi:defcfun ("THCharTensor_addcdiv" th-char-tensor-add-cdiv) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char)
  (src1 th-char-tensor-ptr)
  (src2 th-char-tensor-ptr))
;; void THTensor_(addmv)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat,  THTensor *vec);
(cffi:defcfun ("THCharTensor_addmv" th-char-tensor-add-mv) :void
  (result th-char-tensor-ptr)
  (beta :char)
  (tensor th-char-tensor-ptr)
  (alpha :char)
  (matrix th-char-tensor-ptr)
  (vector th-char-tensor-ptr))
;; void THTensor_(addmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat1, THTensor *mat2);
(cffi:defcfun ("THCharTensor_addmm" th-char-tensor-add-mm) :void
  (result th-char-tensor-ptr)
  (beta :char)
  (tensor th-char-tensor-ptr)
  (alpha :char)
  (matrix1 th-char-tensor-ptr)
  (matrix2 th-char-tensor-ptr))
;; void THTensor_(addr)(THTensor *r_,  real beta, THTensor *t, real alpha, THTensor *vec1, THTensor *vec2);
(cffi:defcfun ("THCharTensor_addr" th-char-tensor-add-r) :void
  (result th-char-tensor-ptr)
  (beta :char)
  (tensor th-char-tensor-ptr)
  (alpha :char)
  (vector1 th-char-tensor-ptr)
  (vector2 th-char-tensor-ptr))
;; void THTensor_(addbmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *batch1, THTensor *batch2);
(cffi:defcfun ("THCharTensor_addbmm" th-char-tensor-add-bmm) :void
  (result th-char-tensor-ptr)
  (beta :char)
  (tensor th-char-tensor-ptr)
  (alpha :char)
  (batch1 th-char-tensor-ptr)
  (batch2 th-char-tensor-ptr))
;; void THTensor_(baddbmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *batch1, THTensor *batch2);
(cffi:defcfun ("THCharTensor_baddbmm" th-char-tensor-badd-bmm) :void
  (result th-char-tensor-ptr)
  (beta :char)
  (tensor th-char-tensor-ptr)
  (alpha :char)
  (batch1 th-char-tensor-ptr)
  (batch2 th-char-tensor-ptr))

;; void THTensor_(match)(THTensor *r_, THTensor *m1, THTensor *m2, real gain);
(cffi:defcfun ("THCharTensor_match" th-char-tensor-match) :void
  (result th-char-tensor-ptr)
  (m1 th-char-tensor-ptr)
  (m2 th-char-tensor-ptr)
  (gain :char))

;; ptrdiff_t THTensor_(numel)(THTensor *t);
(cffi:defcfun ("THCharTensor_numel" th-char-tensor-numel) :long-long
  (tensor th-char-tensor-ptr))
;; void THTensor_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THCharTensor_max" th-char-tensor-max) :void
  (values th-char-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THCharTensor_min" th-char-tensor-min) :void
  (values th-char-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(kthvalue)(THTensor *values_, THLongTensor *indices_, THTensor *t, long k, int dimension, int keepdim);
(cffi:defcfun ("THCharTensor_kthvalue" th-char-tensor-kth-value) :void
  (values th-char-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (k :long)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(mode)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THCharTensor_mode" th-char-tensor-mode) :void
  (values th-char-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(median)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THCharTensor_median" th-char-tensor-median) :void
  (values th-char-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(sum)(THTensor *r_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THCharTensor_sum" th-char-tensor-sum) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(prod)(THTensor *r_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THCharTensor_prod" th-char-tensor-prod) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(cumsum)(THTensor *r_, THTensor *t, int dimension);
(cffi:defcfun ("THCharTensor_cumsum" th-char-tensor-cum-sum) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (dimension :int))
;; void THTensor_(cumprod)(THTensor *r_, THTensor *t, int dimension);
(cffi:defcfun ("THCharTensor_cumprod" th-char-tensor-cum-prod) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (dimension :int))
;; void THTensor_(sign)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THCharTensor_sign" th-char-tensor-sign) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr))
;; accreal THTensor_(trace)(THTensor *t);
(cffi:defcfun ("THCharTensor_trace" th-char-tensor-trace) :long
  (tensor th-char-tensor-ptr))
;; void THTensor_(cross)(THTensor *r_, THTensor *a, THTensor *b, int dimension);
(cffi:defcfun ("THCharTensor_cross" th-char-tensor-cross) :void
  (result th-char-tensor-ptr)
  (a th-char-tensor-ptr)
  (b th-char-tensor-ptr)
  (dimension :int))

;; void THTensor_(cmax)(THTensor *r, THTensor *t, THTensor *src);
(cffi:defcfun ("THCharTensor_cmax" th-char-tensor-cmax) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(cmin)(THTensor *r, THTensor *t, THTensor *src);
(cffi:defcfun ("THCharTensor_cmin" th-char-tensor-cmin) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(cmaxValue)(THTensor *r, THTensor *t, real value);
(cffi:defcfun ("THCharTensor_cmaxValue" th-char-tensor-cmax-value) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))
;; void THTensor_(cminValue)(THTensor *r, THTensor *t, real value);
(cffi:defcfun ("THCharTensor_cminValue" th-char-tensor-cmin-value) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))

;; void THTensor_(zeros)(THTensor *r_, THLongStorage *size);
(cffi:defcfun ("THCharTensor_zeros" th-char-tensor-zeros) :void
  (result th-char-tensor-ptr)
  (size th-long-storage-ptr))
;; void THTensor_(zerosLike)(THTensor *r_, THTensor *input);
(cffi:defcfun ("THCharTensor_zerosLike" th-char-tensor-zero-like) :void
  (result th-char-tensor-ptr)
  (input th-char-tensor-ptr))
;; void THTensor_(ones)(THTensor *r_, THLongStorage *size);
(cffi:defcfun ("THCharTensor_ones" th-char-tensor-ones) :void
  (result th-char-tensor-ptr)
  (size th-long-storage-ptr))
;; void THTensor_(onesLike)(THTensor *r_, THTensor *input);
(cffi:defcfun ("THCharTensor_onesLike" th-char-tensor-one-like) :void
  (result th-char-tensor-ptr)
  (input th-char-tensor-ptr))
;; void THTensor_(diag)(THTensor *r_, THTensor *t, int k);
(cffi:defcfun ("THCharTensor_diag" th-char-tensor-diag) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (k :int))
;; void THTensor_(eye)(THTensor *r_, long n, long m);
(cffi:defcfun ("THCharTensor_eye" th-char-tensor-eye) :void
  (result th-char-tensor-ptr)
  (n :long)
  (m :long))
;; void THTensor_(arange)(THTensor *r_, accreal xmin, accreal xmax, accreal step);
(cffi:defcfun ("THCharTensor_arange" th-char-tensor-arange) :void
  (result th-char-tensor-ptr)
  (xmin :long)
  (xmax :long)
  (step :long))
;; void THTensor_(range)(THTensor *r_, accreal xmin, accreal xmax, accreal step);
(cffi:defcfun ("THCharTensor_range" th-char-tensor-range) :void
  (result th-char-tensor-ptr)
  (xmin :long)
  (xmax :long)
  (step :long))
;; void THTensor_(randperm)(THTensor *r_, THGenerator *_generator, long n);
(cffi:defcfun ("THCharTensor_randperm" th-char-tensor-rand-perm) :void
  (result th-char-tensor-ptr)
  (generator th-generator-ptr)
  (n :long))

;; void THTensor_(reshape)(THTensor *r_, THTensor *t, THLongStorage *size);
(cffi:defcfun ("THCharTensor_reshape" th-char-tensor-reshape) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (size th-long-storage-ptr))
;; void THTensor_(sort)(THTensor *rt_, THLongTensor *ri_, THTensor *t, int dimension, int descendingOrder);
(cffi:defcfun ("THCharTensor_sort" th-char-tensor-sort) :void
  (result-tensor th-char-tensor-ptr)
  (result-indices th-long-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (dimension :int)
  (discending-order :int))
;; void THTensor_(topk)(THTensor *rt_, THLongTensor *ri_, THTensor *t, long k, int dim, int dir, int sorted);
(cffi:defcfun ("THCharTensor_topk" th-char-tensor-topk) :void
  (result-tensor th-char-tensor-ptr)
  (result-indices th-long-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (k :long)
  (dim :int)
  (dir :int)
  (sorted :int))
;; void THTensor_(tril)(THTensor *r_, THTensor *t, long k);
(cffi:defcfun ("THCharTensor_tril" th-char-tensor-tril) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (k :long))
;; void THTensor_(triu)(THTensor *r_, THTensor *t, long k);
(cffi:defcfun ("THCharTensor_triu" th-char-tensor-triu) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (k :long))
;; void THTensor_(cat)(THTensor *r_, THTensor *ta, THTensor *tb, int dimension);
(cffi:defcfun ("THCharTensor_cat" th-char-tensor-cat) :void
  (result th-char-tensor-ptr)
  (tensora th-char-tensor-ptr)
  (tensorb th-char-tensor-ptr)
  (dimension :int))
;; void THTensor_(catArray)(THTensor *result, THTensor **inputs, int numInputs, int dimension);
(cffi:defcfun ("THCharTensor_catArray" th-char-tensor-cat-array) :void
  (result th-char-tensor-ptr)
  (inputs (:pointer th-char-tensor-ptr))
  (num-inputs :int)
  (dimension :int))

;; int THTensor_(equal)(THTensor *ta, THTensor *tb);
(cffi:defcfun ("THCharTensor_equal" th-char-tensor-equal) :int
  (tensora th-char-tensor-ptr)
  (tensorb th-char-tensor-ptr))

;; void THTensor_(ltValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THCharTensor_ltValue" th-char-tensor-lt-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))
;; void THTensor_(leValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THCharTensor_leValue" th-char-tensor-le-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))
;; void THTensor_(gtValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THCharTensor_gtValue" th-char-tensor-gt-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))
;; void THTensor_(geValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THCharTensor_geValue" th-char-tensor-ge-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))
;; void THTensor_(neValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THCharTensor_neValue" th-char-tensor-ne-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))
;; void THTensor_(eqValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THCharTensor_eqValue" th-char-tensor-eq-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))

;; void THTensor_(ltValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THCharTensor_ltValueT" th-char-tensor-lt-value-t) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))
;; void THTensor_(leValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THCharTensor_leValueT" th-char-tensor-le-value-t) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))
;; void THTensor_(gtValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THCharTensor_gtValueT" th-char-tensor-gt-value-t) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))
;; void THTensor_(geValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THCharTensor_geValueT" th-char-tensor-ge-value-t) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))
;; void THTensor_(neValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THCharTensor_neValueT" th-char-tensor-ne-value-t) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))
;; void THTensor_(eqValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THCharTensor_eqValueT" th-char-tensor-eq-value-t) :void
  (result th-char-tensor-ptr)
  (tensor th-char-tensor-ptr)
  (value :char))

;; void THTensor_(ltTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THCharTensor_ltTensor" th-char-tensor-lt-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-char-tensor-ptr)
  (tensorb th-char-tensor-ptr))
;; void THTensor_(leTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THCharTensor_leTensor" th-char-tensor-le-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-char-tensor-ptr)
  (tensorb th-char-tensor-ptr))
;; void THTensor_(gtTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THCharTensor_gtTensor" th-char-tensor-gt-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-char-tensor-ptr)
  (tensorb th-char-tensor-ptr))
;; void THTensor_(geTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THCharTensor_geTensor" th-char-tensor-ge-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-char-tensor-ptr)
  (tensorb th-char-tensor-ptr))
;; void THTensor_(neTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THCharTensor_neTensor" th-char-tensor-ne-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-char-tensor-ptr)
  (tensorb th-char-tensor-ptr))
;; void THTensor_(eqTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THCharTensor_eqTensor" th-char-tensor-eq-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-char-tensor-ptr)
  (tensorb th-char-tensor-ptr))

;; void THTensor_(ltTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THCharTensor_ltTensorT" th-char-tensor-lt-tensor-t) :void
  (result th-char-tensor-ptr)
  (tensora th-char-tensor-ptr)
  (tensorb th-char-tensor-ptr))
;; void THTensor_(leTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THCharTensor_leTensorT" th-char-tensor-le-tensor-t) :void
  (result th-char-tensor-ptr)
  (tensora th-char-tensor-ptr)
  (tensorb th-char-tensor-ptr))
;; void THTensor_(gtTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THCharTensor_gtTensorT" th-char-tensor-gt-tensor-t) :void
  (result th-char-tensor-ptr)
  (tensora th-char-tensor-ptr)
  (tensorb th-char-tensor-ptr))
;; void THTensor_(geTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THCharTensor_geTensorT" th-char-tensor-ge-tensor-t) :void
  (result th-char-tensor-ptr)
  (tensora th-char-tensor-ptr)
  (tensorb th-char-tensor-ptr))
;; void THTensor_(neTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THCharTensor_neTensorT" th-char-tensor-ne-tensor-t) :void
  (result th-char-tensor-ptr)
  (tensora th-char-tensor-ptr)
  (tensorb th-char-tensor-ptr))
;; void THTensor_(eqTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THCharTensor_eqTensorT" th-char-tensor-eq-tensor-t) :void
  (result th-char-tensor-ptr)
  (tensora th-char-tensor-ptr)
  (tensorb th-char-tensor-ptr))

;; void THTensor_(validXCorr2Dptr)(real *r_,
;;                                 real alpha,
;;                                 real *t_, long ir, long ic,
;;                                 real *k_, long kr, long kc,
;;                                 long sr, long sc);
(cffi:defcfun ("THCharTensor_validXCorr2Dptr" th-char-tensor-valid-x-corr-2d-ptr) :void
  (res (:pointer :char))
  (alpha :char)
  (ten (:pointer :char))
  (ir :long)
  (ic :long)
  (k (:pointer :char))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(validConv2Dptr)(real *r_,
;;                                real alpha,
;;                                real *t_, long ir, long ic,
;;                                real *k_, long kr, long kc,
;;                                long sr, long sc);
(cffi:defcfun ("THCharTensor_validConv2Dptr" th-char-tensor-valid-conv-2d-ptr) :void
  (res (:pointer :char))
  (alpha :char)
  (ten (:pointer :char))
  (ir :long)
  (ic :long)
  (k (:pointer :char))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(fullXCorr2Dptr)(real *r_,
;;                                real alpha,
;;                                real *t_, long ir, long ic,
;;                                real *k_, long kr, long kc,
;;                                long sr, long sc);
(cffi:defcfun ("THCharTensor_fullXCorr2Dptr" th-char-tensor-full-x-corr-2d-ptr) :void
  (res (:pointer :char))
  (alpha :char)
  (ten (:pointer :char))
  (ir :long)
  (ic :long)
  (k (:pointer :char))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(fullConv2Dptr)(real *r_,
;;                               real alpha,
;;                               real *t_, long ir, long ic,
;;                               real *k_, long kr, long kc,
;;                               long sr, long sc);
(cffi:defcfun ("THCharTensor_fullConv2Dptr" th-char-tensor-full-conv-2d-ptr) :void
  (res (:pointer :char))
  (alpha :char)
  (ten (:pointer :char))
  (ir :long)
  (ic :long)
  (k (:pointer :char))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(validXCorr2DRevptr)(real *r_,
;;                                    real alpha,
;;                                    real *t_, long ir, long ic,
;;                                    real *k_, long kr, long kc,
;;                                    long sr, long sc);
(cffi:defcfun ("THCharTensor_validXCorr2DRevptr" th-char-tensor-valid-x-corr-2d-rev-ptr) :void
  (res (:pointer :char))
  (alpha :char)
  (ten (:pointer :char))
  (ir :long)
  (ic :long)
  (k (:pointer :char))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(conv2DRevger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol);
(cffi:defcfun ("THCharTensor_conv2DRevger" th-char-tensor-conv-2d-rev-ger) :void
  (result th-char-tensor-ptr)
  (beta :char)
  (alpha :char)
  (tensor th-char-tensor-ptr)
  (k th-char-tensor-ptr)
  (srow :long)
  (scol :long))
;; void THTensor_(conv2DRevgerm)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol);
(cffi:defcfun ("THCharTensor_conv2DRevgerm" th-char-tensor-conv-2d-rev-germ) :void
  (result th-char-tensor-ptr)
  (beta :char)
  (alpha :char)
  (tensor th-char-tensor-ptr)
  (k th-char-tensor-ptr)
  (srow :long)
  (scol :long))
;; void THTensor_(conv2Dger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THCharTensor_conv2Dger" th-char-tensor-conv-2d-ger) :void
  (result th-char-tensor-ptr)
  (beta :char)
  (alpha :char)
  (tensor th-char-tensor-ptr)
  (k th-char-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv2Dmv)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THCharTensor_conv2Dmv" th-char-tensor-conv-2d-mv) :void
  (result th-char-tensor-ptr)
  (beta :char)
  (alpha :char)
  (tensor th-char-tensor-ptr)
  (k th-char-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv2Dmm)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THCharTensor_conv2Dmm" th-char-tensor-conv-2d-mm) :void
  (result th-char-tensor-ptr)
  (beta :char)
  (alpha :char)
  (tensor th-char-tensor-ptr)
  (k th-char-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv2Dmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THCharTensor_conv2Dmul" th-char-tensor-conv-2d-mul) :void
  (result th-char-tensor-ptr)
  (beta :char)
  (alpha :char)
  (tensor th-char-tensor-ptr)
  (k th-char-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv2Dcmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THCharTensor_conv2Dcmul" th-char-tensor-conv-2d-cmul) :void
  (result th-char-tensor-ptr)
  (beta :char)
  (alpha :char)
  (tensor th-char-tensor-ptr)
  (k th-char-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))

;; void THTensor_(validXCorr3Dptr)(real *r_,
;;                                 real alpha,
;;                                 real *t_, long it, long ir, long ic,
;;                                 real *k_, long kt, long kr, long kc,
;;                                 long st, long sr, long sc);
(cffi:defcfun ("THCharTensor_validXCorr3Dptr" th-char-tensor-valid-x-corr-3d-ptr) :void
  (res (:pointer :char))
  (alpha :char)
  (ten (:pointer :char))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :char))
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
(cffi:defcfun ("THCharTensor_validConv3Dptr" th-char-tensor-valid-conv-3d-ptr) :void
  (res (:pointer :char))
  (alpha :char)
  (ten (:pointer :char))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :char))
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
(cffi:defcfun ("THCharTensor_fullXCorr3Dptr" th-char-tensor-full-x-corr-3d-ptr) :void
  (res (:pointer :char))
  (alpha :char)
  (ten (:pointer :char))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :char))
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
(cffi:defcfun ("THCharTensor_fullConv3Dptr" th-char-tensor-full-conv-3d-ptr) :void
  (res (:pointer :char))
  (alpha :char)
  (ten (:pointer :char))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :char))
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
(cffi:defcfun ("THCharTensor_validXCorr3DRevptr" th-char-tensor-valid-x-corr-3d-rev-ptr) :void
  (res (:pointer :char))
  (alpha :char)
  (ten (:pointer :char))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :char))
  (kt :long)
  (kr :long)
  (kc :long)
  (st :long)
  (sr :long)
  (sc :long))

;; void THTensor_(conv3DRevger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol);
(cffi:defcfun ("THCharTensor_conv3DRevger" th-char-tensor-conv-3d-rev-ger) :void
  (result th-char-tensor-ptr)
  (beta :char)
  (alpha :char)
  (tensor th-char-tensor-ptr)
  (k th-char-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long))
;; void THTensor_(conv3Dger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THCharTensor_conv3Dger" th-char-tensor-conv-3d-ger) :void
  (result th-char-tensor-ptr)
  (beta :char)
  (alpha :char)
  (tensor th-char-tensor-ptr)
  (k th-char-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv3Dmv)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THCharTensor_conv3Dmv" th-char-tensor-conv-3d-mv) :void
  (result th-char-tensor-ptr)
  (beta :char)
  (alpha :char)
  (tensor th-char-tensor-ptr)
  (k th-char-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv3Dmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THCharTensor_conv3Dmul" th-char-tensor-conv-3d-mul) :void
  (result th-char-tensor-ptr)
  (beta :char)
  (alpha :char)
  (tensor th-char-tensor-ptr)
  (k th-char-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
;; void THTensor_(conv3Dcmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THCharTensor_conv3Dcmul" th-char-tensor-conv-3d-cmul) :void
  (result th-char-tensor-ptr)
  (beta :char)
  (alpha :char)
  (tensor th-char-tensor-ptr)
  (k th-char-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf (:pointer :char))
  (xc (:pointer :char)))
