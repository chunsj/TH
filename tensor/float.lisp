(in-package :th)

;; ACCESS METHODS
;; THStorage* THTensor_(storage)(const THTensor *self)
(cffi:defcfun ("THFloatTensor_storage" th-float-tensor-storage) th-float-storage-ptr
  (tensor th-float-tensor-ptr))
;; ptrdiff_t THTensor_(storageOffset)(const THTensor *self)
(cffi:defcfun ("THFloatTensor_storageOffset" th-float-tensor-storage-offset) :long-long
  (tensor th-float-tensor-ptr))
;; int THTensor_(nDimension)(const THTensor *self)
(cffi:defcfun ("THFloatTensor_nDimension" th-float-tensor-n-dimension) :int
  (tensor th-float-tensor-ptr))
;; long THTensor_(size)(const THTensor *self, int dim)
(cffi:defcfun ("THFloatTensor_size" th-float-tensor-size) :long
  (tensor th-float-tensor-ptr)
  (dim :int))
;; long THTensor_(stride)(const THTensor *self, int dim)
(cffi:defcfun ("THFloatTensor_stride" th-float-tensor-stride) :long
  (tensor th-float-tensor-ptr)
  (dim :int))
;; THLongStorage *THTensor_(newSizeOf)(THTensor *self)
(cffi:defcfun ("THFloatTensor_newSizeOf" th-float-tensor-new-size-of) th-long-storage-ptr
  (tensor th-float-tensor-ptr))
;; THLongStorage *THTensor_(newStrideOf)(THTensor *self)
(cffi:defcfun ("THFloatTensor_newStrideOf" th-float-tensor-new-stride-of) th-long-storage-ptr
  (tensor th-float-tensor-ptr))
;; real *THTensor_(data)(const THTensor *self)
(cffi:defcfun ("THFloatTensor_data" th-float-tensor-data) (:pointer :float)
  (tensor th-float-tensor-ptr))

;; void THTensor_(setFlag)(THTensor *self, const char flag)
(cffi:defcfun ("THFloatTensor_setFlag" th-float-tensor-set-flag) :void
  (tensor th-float-tensor-ptr)
  (flag :char))
;; void THTensor_(clearFlag)(THTensor *self, const char flag)
(cffi:defcfun ("THFloatTensor_clearFlag" th-float-tensor-clear-flag) :void
  (tensor th-float-tensor-ptr)
  (flag :char))

;; CREATION METHODS
;; THTensor *THTensor_(new)(void)
(cffi:defcfun ("THFloatTensor_new" th-float-tensor-new) th-float-tensor-ptr)
;; THTensor *THTensor_(newWithTensor)(THTensor *tensor)
(cffi:defcfun ("THFloatTensor_newWithTensor" th-float-tensor-new-with-tensor) th-float-tensor-ptr
  (tensor th-float-tensor-ptr))
;; stride might be NULL
;; THTensor *THTensor_(newWithStorage)(THStorage *storage_, ptrdiff_t storageOffset_,
;;                                            THLongStorage *size_, THLongStorage *stride_)
(cffi:defcfun ("THFloatTensor_newWithStorage" th-float-tensor-new-with-storage)
    th-float-tensor-ptr
  (storage th-float-storage-ptr)
  (storage-offset :long-long)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
;; THTensor *THTensor_(newWithStorage1d)(THStorage *storage_, ptrdiff_t storageOffset_,
;;                                              long size0_, long stride0_);
(cffi:defcfun ("THFloatTensor_newWithStorage1d" th-float-tensor-new-with-storage-1d)
    th-float-tensor-ptr
  (storage th-float-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long))
(cffi:defcfun ("THFloatTensor_newWithStorage2d" th-float-tensor-new-with-storage-2d)
    th-float-tensor-ptr
  (storage th-float-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long))
(cffi:defcfun ("THFloatTensor_newWithStorage3d" th-float-tensor-new-with-storage-3d)
    th-float-tensor-ptr
  (storage th-float-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long)
  (size2 :long)
  (stride2 :long))
(cffi:defcfun ("THFloatTensor_newWithStorage4d" th-float-tensor-new-with-storage-4d)
    th-float-tensor-ptr
  (storage th-float-storage-ptr)
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
(cffi:defcfun ("THFloatTensor_newWithSize" th-float-tensor-new-with-size) th-float-tensor-ptr
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
;; THTensor *THTensor_(newWithSize1d)(long size0_);
(cffi:defcfun ("THFloatTensor_newWithSize1d" th-float-tensor-new-with-size-1d)
    th-float-tensor-ptr
  (size0 :long))
(cffi:defcfun ("THFloatTensor_newWithSize2d" th-float-tensor-new-with-size-2d)
    th-float-tensor-ptr
  (size0 :long)
  (size1 :long))
(cffi:defcfun ("THFloatTensor_newWithSize3d" th-float-tensor-new-with-size-3d)
    th-float-tensor-ptr
  (size0 :long)
  (size1 :long)
  (size2 :long))
(cffi:defcfun ("THFloatTensor_newWithSize4d" th-float-tensor-new-with-size-4d)
    th-float-tensor-ptr
  (size0 :long)
  (size1 :long)
  (size2 :long)
  (size3 :long))

;; THTensor *THTensor_(newClone)(THTensor *self)
(cffi:defcfun ("THFloatTensor_newClone" th-float-tensor-new-clone) th-float-tensor-ptr
  (tensor th-float-tensor-ptr))
(cffi:defcfun ("THFloatTensor_newContiguous" th-float-tensor-new-contiguous) th-float-tensor-ptr
  (tensor th-float-tensor-ptr))
(cffi:defcfun ("THFloatTensor_newSelect" th-float-tensor-new-select) th-float-tensor-ptr
  (tensor th-float-tensor-ptr)
  (dimension :int)
  (slice-index :long))
(cffi:defcfun ("THFloatTensor_newNarrow" th-float-tensor-new-narrow) th-float-tensor-ptr
  (tensor th-float-tensor-ptr)
  (dimension :int)
  (first-index :long)
  (size :long))

(cffi:defcfun ("THFloatTensor_newTranspose" th-float-tensor-new-transpose) th-float-tensor-ptr
  (tensor th-float-tensor-ptr)
  (dimension1 :int)
  (dimension2 :int))
(cffi:defcfun ("THFloatTensor_newUnfold" th-float-tensor-new-unfold) th-float-tensor-ptr
  (tensor th-float-tensor-ptr)
  (dimension :int)
  (size :long)
  (step :long))
(cffi:defcfun ("THFloatTensor_newView" th-float-tensor-new-view) th-float-tensor-ptr
  (tensor th-float-tensor-ptr)
  (size th-long-storage-ptr))

(cffi:defcfun ("THFloatTensor_expand" th-float-tensor-expand) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (size th-long-storage-ptr))

(cffi:defcfun ("THFloatTensor_resize" th-float-tensor-resize) :void
  (tensor th-float-tensor-ptr)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
(cffi:defcfun ("THFloatTensor_resizeAs" th-float-tensor-resize-as) :void
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr))
(cffi:defcfun ("THFloatTensor_resizeNd" th-float-tensor-resize-nd) :void
  (tensor th-float-tensor-ptr)
  (dimension :int)
  (size (:pointer :long))
  (stride (:pointer :long)))
(cffi:defcfun ("THFloatTensor_resize1d" th-float-tensor-resize-1d) :void
  (tensor th-float-tensor-ptr)
  (size0 :long))
(cffi:defcfun ("THFloatTensor_resize2d" th-float-tensor-resize-2d) :void
  (tensor th-float-tensor-ptr)
  (size0 :long)
  (size1 :long))
(cffi:defcfun ("THFloatTensor_resize3d" th-float-tensor-resize-3d) :void
  (tensor th-float-tensor-ptr)
  (size0 :long)
  (size1 :long)
  (size2 :long))
(cffi:defcfun ("THFloatTensor_resize4d" th-float-tensor-resize-4d) :void
  (tensor th-float-tensor-ptr)
  (size0 :long)
  (size1 :long)
  (size2 :long)
  (size3 :long))
(cffi:defcfun ("THFloatTensor_resize5d" th-float-tensor-resize-5d) :void
  (tensor th-float-tensor-ptr)
  (size0 :long)
  (size1 :long)
  (size2 :long)
  (size3 :long)
  (size4 :long))

(cffi:defcfun ("THFloatTensor_set" th-float-tensor-set) :void
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr))
(cffi:defcfun ("THFloatTensor_setStorage" th-float-tensor-set-storage) :void
  (tensor th-float-tensor-ptr)
  (storage th-float-storage-ptr)
  (storage-offset :long-long)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
(cffi:defcfun ("THFloatTensor_setStorageNd" th-float-tensor-set-storage-nd) :void
  (tensor th-float-tensor-ptr)
  (storage th-float-storage-ptr)
  (storage-offset :long-long)
  (dimension :int)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
(cffi:defcfun ("THFloatTensor_setStorage1d" th-float-tensor-set-storage-1d) :void
  (tensor th-float-tensor-ptr)
  (storage th-float-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long))
(cffi:defcfun ("THFloatTensor_setStorage2d" th-float-tensor-set-storage-2d) :void
  (tensor th-float-tensor-ptr)
  (storage th-float-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long))
(cffi:defcfun ("THFloatTensor_setStorage3d" th-float-tensor-set-storage-3d) :void
  (tensor th-float-tensor-ptr)
  (storage th-float-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long)
  (size2 :long)
  (stride2 :long))
(cffi:defcfun ("THFloatTensor_setStorage4d" th-float-tensor-set-storage-4d) :void
  (tensor th-float-tensor-ptr)
  (storage th-float-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long)
  (size2 :long)
  (stride2 :long)
  (size3 :long)
  (stride3 :long))

(cffi:defcfun ("THFloatTensor_narrow" th-float-tensor-narrow) :void
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr)
  (dimension :int)
  (first-index :long)
  (size :long))
(cffi:defcfun ("THFloatTensor_select" th-float-tensor-select) :void
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr)
  (dimension :int)
  (slice-index :long))
(cffi:defcfun ("THFloatTensor_transpose" th-float-tensor-transpose) :void
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr)
  (dimension1 :int)
  (dimension2 :int))
(cffi:defcfun ("THFloatTensor_unfold" th-float-tensor-unfold) :void
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr)
  (dimension :int)
  (size :long)
  (step :long))
(cffi:defcfun ("THFloatTensor_squeeze" th-float-tensor-squeeze) :void
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr))
(cffi:defcfun ("THFloatTensor_squeeze1d" th-float-tensor-squeeze-1d) :void
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr)
  (dimension :int))
(cffi:defcfun ("THFloatTensor_unsqueeze1d" th-float-tensor-unsqueeze-1d) :void
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr)
  (dimension :int))

(cffi:defcfun ("THFloatTensor_isContiguous" th-float-tensor-is-contiguous) :int
  (tensor th-float-tensor-ptr))
(cffi:defcfun ("THFloatTensor_isSameSizeAs" th-float-tensor-is-same-size-as) :int
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr))
(cffi:defcfun ("THFloatTensor_isSetTo" th-float-tensor-is-set-to) :int
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr))
(cffi:defcfun ("THFloatTensor_isSize" th-float-tensor-is-size) :int
  (tensor th-float-tensor-ptr)
  (dims th-long-storage-ptr))
(cffi:defcfun ("THFloatTensor_nElement" th-float-tensor-n-element) :long-long
  (tensor th-float-tensor-ptr))

(cffi:defcfun ("THFloatTensor_retain" th-float-tensor-retain) :void
  (tensor th-float-tensor-ptr))
(cffi:defcfun ("THFloatTensor_free" th-float-tensor-free) :void
  (tensor th-float-tensor-ptr))
(cffi:defcfun ("THFloatTensor_freeCopyTo" th-float-tensor-free-copy-to) :void
  (source th-float-tensor-ptr)
  (target th-float-tensor-ptr))

;; slow access methods [check everything]
;; void THTensor_(set1d)(THTensor *tensor, long x0, real value);
(cffi:defcfun ("THFloatTensor_set1d" th-float-tensor-set-1d) :void
  (tensor th-float-tensor-ptr)
  (index0 :long)
  (value :float))
;; void THTensor_(set2d)(THTensor *tensor, long x0, long x1, real value);
(cffi:defcfun ("THFloatTensor_set2d" th-float-tensor-set-2d) :void
  (tensor th-float-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (value :float))
;; void THTensor_(set3d)(THTensor *tensor, long x0, long x1, long x2, real value);
(cffi:defcfun ("THFloatTensor_set3d" th-float-tensor-set-3d) :void
  (tensor th-float-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long)
  (value :float))
;; void THTensor_(set4d)(THTensor *tensor, long x0, long x1, long x2, long x3, real value);
(cffi:defcfun ("THFloatTensor_set4d" th-float-tensor-set-4d) :void
  (tensor th-float-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long)
  (index3 :long)
  (value :float))

(cffi:defcfun ("THFloatTensor_get1d" th-float-tensor-get-1d) :float
  (tensor th-float-tensor-ptr)
  (index0 :long))
(cffi:defcfun ("THFloatTensor_get2d" th-float-tensor-get-2d) :float
  (tensor th-float-tensor-ptr)
  (index0 :long)
  (index1 :long))
(cffi:defcfun ("THFloatTensor_get3d" th-float-tensor-get-3d) :float
  (tensor th-float-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long))
(cffi:defcfun ("THFloatTensor_get4d" th-float-tensor-get-4d) :float
  (tensor th-float-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long)
  (index3 :long))

;; support for copy betweeb different tensor types
;; void THTensor_(copy)(THTensor *tensor, THTensor *src);
(cffi:defcfun ("THFloatTensor_copy" th-float-tensor-copy) :void
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(copyByte)(THTensor *tensor, struct THByteTensor *src);
(cffi:defcfun ("THFloatTensor_copyByte" th-float-tensor-copy-byte) :void
  (tensor th-float-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(copyChar)(THTensor *tensor, struct THCharTensor *src);
(cffi:defcfun ("THFloatTensor_copyChar" th-float-tensor-copy-char) :void
  (tensor th-float-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(copyShort)(THTensor *tensor, struct THShortTensor *src);
(cffi:defcfun ("THFloatTensor_copyShort" th-float-tensor-copy-short) :void
  (tensor th-float-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(copyInt)(THTensor *tensor, struct THIntTensor *src);
(cffi:defcfun ("THFloatTensor_copyInt" th-float-tensor-copy-int) :void
  (tensor th-float-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(copyLong)(THTensor *tensor, struct THLongTensor *src);
(cffi:defcfun ("THFloatTensor_copyLong" th-float-tensor-copy-long) :void
  (tensor th-float-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(copyFloat)(THTensor *tensor, struct THFloatTensor *src);
(cffi:defcfun ("THFloatTensor_copyFloat" th-float-tensor-copy-float) :void
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(copyDouble)(THTensor *tensor, struct THDoubleTensor *src);
(cffi:defcfun ("THFloatTensor_copyDouble" th-float-tensor-copy-double) :void
  (tensor th-float-tensor-ptr)
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
(cffi:defcfun ("THFloatTensor_random" th-float-tensor-random) :void
  (tensor th-float-tensor-ptr)
  (generator th-generator-ptr))
;; void THTensor_(clampedRandom)(THTensor *self, THGenerator *_generator, long min, long max)
(cffi:defcfun ("THFloatTensor_clampedRandom" th-float-tensor-clamped-random) :void
  (tensor th-float-tensor-ptr)
  (genrator th-generator-ptr)
  (min :long)
  (max :long))
;; void THTensor_(cappedRandom)(THTensor *self, THGenerator *_generator, long max);
(cffi:defcfun ("THFloatTensor_cappedRandom" th-float-tensor-capped-random) :void
  (tensor th-float-tensor-ptr)
  (generator th-generator-ptr)
  (max :long))
;; void THTensor_(geometric)(THTensor *self, THGenerator *_generator, double p);
(cffi:defcfun ("THFloatTensor_geometric" th-float-tensor-geometric) :void
  (tensor th-float-tensor-ptr)
  (generator th-generator-ptr)
  (p :double))
;; void THTensor_(bernoulli)(THTensor *self, THGenerator *_generator, double p);
(cffi:defcfun ("THFloatTensor_bernoulli" th-float-tensor-bernoulli) :void
  (tensor th-float-tensor-ptr)
  (generator th-generator-ptr)
  (p :double))
;; void THTensor_(bernoulli_FloatTensor)(THTensor *self, THGenerator *_generator, THFloatTensor *p);
(cffi:defcfun ("THFloatTensor_bernoulli_FloatTensor" th-float-tensor-bernoulli-float-tensor) :void
  (tensor th-float-tensor-ptr)
  (generator th-generator-ptr)
  (p th-float-tensor-ptr))
;; void THTensor_(bernoulli_DoubleTensor)(THTensor *self, THGenerator *_generator, THDoubleTensor *p);
(cffi:defcfun ("THFloatTensor_bernoulli_DoubleTensor" th-float-tensor-bernoulli-double-tensor)
    :void
  (tensor th-float-tensor-ptr)
  (generator th-generator-ptr)
  (p th-double-tensor-ptr))

;; #if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
;; void THTensor_(uniform)(THTensor *self, THGenerator *_generator, double a, double b);
(cffi:defcfun ("THFloatTensor_uniform" th-float-tensor-uniform) :void
  (tensor th-float-tensor-ptr)
  (generator th-generator-ptr)
  (a :double)
  (b :double))
;; void THTensor_(normal)(THTensor *self, THGenerator *_generator, double mean, double stdv);
(cffi:defcfun ("THFloatTensor_normal" th-float-tensor-normal) :void
  (tensor th-float-tensor-ptr)
  (generator th-generator-ptr)
  (mean :double)
  (stdv :double))
;; void THTensor_(normal_means)(THTensor *self, THGenerator *gen, THTensor *means, double stddev);
(cffi:defcfun ("THFloatTensor_normal_means" th-float-tensor-normal-means) :void
  (tensor th-float-tensor-ptr)
  (generator th-generator-ptr)
  (means th-float-tensor-ptr)
  (stdv :double))
;; void THTensor_(normal_stddevs)(THTensor *self, THGenerator *gen, double mean, THTensor *stddevs);
(cffi:defcfun ("THFloatTensor_normal_stddevs" th-float-tensor-normal-stddevs) :void
  (tensor th-float-tensor-ptr)
  (generator th-generator-ptr)
  (mean :double)
  (stddevs th-float-tensor-ptr))
;; void THTensor_(normal_means_stddevs)(THTensor *self, THGenerator *gen, THTensor *means, THTensor *stddevs);
(cffi:defcfun ("THFloatTensor_normal_means_stddevs" th-float-tensor-normal-means-stddevs) :void
  (tensor th-float-tensor-ptr)
  (generator th-generator-ptr)
  (means th-float-tensor-ptr)
  (stddevs th-float-tensor-ptr))
;; void THTensor_(exponential)(THTensor *self, THGenerator *_generator, double lambda);
(cffi:defcfun ("THFloatTensor_exponential" th-float-tensor-exponential) :void
  (tensor th-float-tensor-ptr)
  (generator th-generator-ptr)
  (lam :double))
;; void THTensor_(cauchy)(THTensor *self, THGenerator *_generator, double median, double sigma);
(cffi:defcfun ("THFloatTensor_cauchy" th-float-tensor-cauchy) :void
  (tensor th-float-tensor-ptr)
  (generator th-generator-ptr)
  (median :double)
  (sigma :double))
;; void THTensor_(logNormal)(THTensor *self, THGenerator *_generator, double mean, double stdv);
(cffi:defcfun ("THFloatTensor_logNormal" th-float-tensor-log-normal) :void
  (tensor th-float-tensor-ptr)
  (generator th-generator-ptr)
  (mean :double)
  (stdv :double))
;; void THTensor_(multinomial)(THLongTensor *self, THGenerator *_generator, THTensor *prob_dist, int n_sample, int with_replacement);
(cffi:defcfun ("THFloatTensor_multinomial" th-float-tensor-multinomial) :void
  (tensor th-float-tensor-ptr)
  (generator th-generator-ptr)
  (prob-dist th-float-tensor-ptr)
  (n-sample :int)
  (replacement :int))
;; void THTensor_(multinomialAliasSetup)(THTensor *prob_dist, THLongTensor *J, THTensor *q);
(cffi:defcfun ("THFloatTensor_multinomialAliasSetup" th-float-tensor-multinomial-alias-setup)
    :void
  (prob-dist th-float-tensor-ptr)
  (j th-long-tensor-ptr)
  (q th-float-tensor-ptr))
;; void THTensor_(multinomialAliasDraw)(THLongTensor *self, THGenerator *_generator, THLongTensor *J, THTensor *q);
(cffi:defcfun ("THFloatTensor_multinomialAliasDraw" th-float-tensor-multinomial-alias-draw)
    :void
  (tensor th-float-tensor-ptr)
  (generator th-generator-ptr)
  (j th-long-tensor-ptr)
  (q th-float-tensor-ptr))
;; #endif

;; #if defined(TH_REAL_IS_BYTE)
;; void THTensor_(getRNGState)(THGenerator *_generator, THTensor *self);
;; void THTensor_(setRNGState)(THGenerator *_generator, THTensor *self);
;; #endif

;; void THTensor_(fill)(THTensor *r_, real value);
(cffi:defcfun ("THFloatTensor_fill" th-float-tensor-fill) :void
  (tensor th-float-tensor-ptr)
  (value :float))
;; void THTensor_(zero)(THTensor *r_);
(cffi:defcfun ("THFloatTensor_zero" th-float-tensor-zero) :void
  (tensor th-float-tensor-ptr))

;; void THTensor_(maskedFill)(THTensor *tensor, THByteTensor *mask, real value);
(cffi:defcfun ("THFloatTensor_maskedFill" th-float-tensor-masked-fill) :void
  (tensor th-float-tensor-ptr)
  (mask th-byte-tensor-ptr)
  (value :float))
;; void THTensor_(maskedCopy)(THTensor *tensor, THByteTensor *mask, THTensor* src);
(cffi:defcfun ("THFloatTensor_maskedCopy" th-float-tensor-masked-copy) :void
  (tensor th-float-tensor-ptr)
  (mask th-byte-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(maskedSelect)(THTensor *tensor, THTensor* src, THByteTensor *mask);
(cffi:defcfun ("THFloatTensor_maskedSelect" th-float-tensor-masked-select) :void
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr)
  (mask (th-byte-tensor-ptr)))

;; void THTensor_(nonzero)(THLongTensor *subscript, THTensor *tensor);
(cffi:defcfun ("THFloatTensor_nonzero" th-float-tensor-nonzero) :void
  (subscript th-long-tensor-ptr)
  (tensor th-float-tensor-ptr))

;; void THTensor_(indexSelect)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index)
(cffi:defcfun ("THFloatTensor_indexSelect" th-float-tensor-index-select) :void
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr))
;; void THTensor_(indexCopy)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THFloatTensor_indexCopy" th-float-tensor-index-copy) :void
  (tensor th-float-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(indexAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THFloatTensor_indexAdd" th-float-tensor-index-add) :void
  (tensor th-float-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(indexFill)(THTensor *tensor, int dim, THLongTensor *index, real val);
(cffi:defcfun ("THFloatTensor_indexFill" th-float-tensor-index-fill) :void
  (tensor th-float-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (value :float))

;; void THTensor_(gather)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index);
(cffi:defcfun ("THFloatTensor_gather" th-float-tensor-gather) :void
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr))
;; void THTensor_(scatter)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THFloatTensor_scatter" th-float-tensor-scatter) :void
  (tensor th-float-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(scatterAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THFloatTensor_scatterAdd" th-float-tensor-scatter-add) :void
  (tensor th-float-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(scatterFill)(THTensor *tensor, int dim, THLongTensor *index, real val);
(cffi:defcfun ("THFloatTensor_scatterFill" th-float-tensor-scatter-fill) :void
  (tensor th-float-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (value :float))

;; accreal THTensor_(dot)(THTensor *t, THTensor *src);
(cffi:defcfun ("THFloatTensor_dot" th-float-tensor-dot) :double
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr))

;; real THTensor_(minall)(THTensor *t);
(cffi:defcfun ("THFloatTensor_minall" th-float-tensor-min-all) :float
  (tensor th-float-tensor-ptr))
;; real THTensor_(maxall)(THTensor *t);
(cffi:defcfun ("THFloatTensor_maxall" th-float-tensor-max-all) :float
  (tensor th-float-tensor-ptr))
;; real THTensor_(medianall)(THTensor *t);
(cffi:defcfun ("THFloatTensor_medianall" th-float-tensor-median-all) :float
  (tensor th-float-tensor-ptr))
;; accreal THTensor_(sumall)(THTensor *t);
(cffi:defcfun ("THFloatTensor_sumall" th-float-tensor-sum-all) :double
  (tensor th-float-tensor-ptr))
;; accreal THTensor_(prodall)(THTensor *t);
(cffi:defcfun ("THFloatTensor_prodall" th-float-tensor-prod-all) :double
  (tensor th-float-tensor-ptr))

;; void THTensor_(neg)(THTensor *self, THTensor *src);
(cffi:defcfun ("THFloatTensor_neg" th-float-tensor-neg) :void
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(cinv)(THTensor *self, THTensor *src);
(cffi:defcfun ("THFloatTensor_cinv" th-float-tensor-cinv) :void
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr))

;; void THTensor_(add)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THFloatTensor_add" th-float-tensor-add) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))
;; void THTensor_(sub)(THTensor *self, THTensor *src, real value);
(cffi:defcfun ("THFloatTensor_sub" th-float-tensor-sub) :void
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr)
  (value :float))
;; void THTensor_(mul)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THFloatTensor_mul" th-float-tensor-mul) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))
;; void THTensor_(div)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THFloatTensor_div" th-float-tensor-div) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))
;; void THTensor_(lshift)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THFloatTensor_lshift" th-float-tensor-lshift) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :double))
;; void THTensor_(rshift)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THFloatTensor_rshift" th-float-tensor-rshift) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))
;; void THTensor_(fmod)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THFloatTensor_fmod" th-float-tensor-fmod) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))
;; void THTensor_(remainder)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THFloatTensor_remainder" th-float-tensor-remainder) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))
;; void THTensor_(clamp)(THTensor *r_, THTensor *t, real min_value, real max_value);
(cffi:defcfun ("THFloatTensor_clamp" th-float-tensor-clamp) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (min-value :float)
  (max-value :float))
;; void THTensor_(bitand)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THFloatTensor_bitand" th-float-tensor-bitand) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))
;; void THTensor_(bitor)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THFloatTensor_bitor" th-float-tensor-bitor) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))
;; void THTensor_(bitxor)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THFloatTensor_bitxor" th-float-tensor-bitxor) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))

;; void THTensor_(cadd)(THTensor *r_, THTensor *t, real value, THTensor *src);
(cffi:defcfun ("THFloatTensor_cadd" th-float-tensor-cadd) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float)
  (src th-float-tensor-ptr))
;; void THTensor_(csub)(THTensor *self, THTensor *src1, real value, THTensor *src2);
(cffi:defcfun ("THFloatTensor_csub" th-float-tensor-csub) :void
  (tensor th-float-tensor-ptr)
  (src1 th-float-tensor-ptr)
  (value :float)
  (src2 th-float-tensor-ptr))
;; void THTensor_(cmul)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THFloatTensor_cmul" th-float-tensor-cmul) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(cpow)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THFloatTensor_cpow" th-float-tensor-cpow) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(cdiv)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THFloatTensor_cdiv" th-float-tensor-cdiv) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(clshift)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THFloatTensor_clshift" th-float-tensor-clshift) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(crshift)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THFloatTensor_crshift" th-float-tensor-crshift) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(cfmod)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THFloatTensor_cfmod" th-float-tensor-cfmod) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(cremainder)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THFloatTensor_cremainder" th-float-tensor-cremainder) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(cbitand)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THFloatTensor_cbitand" th-float-tensor-cbitand) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(cbitor)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THFloatTensor_cbitor" th-float-tensor-cbitor) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(cbitxor)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THFloatTensor_cbitxor" th-float-tensor-cbitxor) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr))

;; void THTensor_(addcmul)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2);
(cffi:defcfun ("THFloatTensor_addcmul" th-float-tensor-add-cmul) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float)
  (src1 th-float-tensor-ptr)
  (src2 th-float-tensor-ptr))
;; void THTensor_(addcdiv)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2);
(cffi:defcfun ("THFloatTensor_addcdiv" th-float-tensor-add-cdiv) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float)
  (src1 th-float-tensor-ptr)
  (src2 th-float-tensor-ptr))
;; void THTensor_(addmv)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat,  THTensor *vec);
(cffi:defcfun ("THFloatTensor_addmv" th-float-tensor-add-mv) :void
  (result th-float-tensor-ptr)
  (beta :float)
  (tensor th-float-tensor-ptr)
  (alpha :float)
  (matrix th-float-tensor-ptr)
  (vector th-float-tensor-ptr))
;; void THTensor_(addmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat1, THTensor *mat2);
(cffi:defcfun ("THFloatTensor_addmm" th-float-tensor-add-mm) :void
  (result th-float-tensor-ptr)
  (beta :float)
  (tensor th-float-tensor-ptr)
  (alpha :float)
  (matrix1 th-float-tensor-ptr)
  (matrix2 th-float-tensor-ptr))
;; void THTensor_(addr)(THTensor *r_,  real beta, THTensor *t, real alpha, THTensor *vec1, THTensor *vec2);
(cffi:defcfun ("THFloatTensor_addr" th-float-tensor-add-r) :void
  (result th-float-tensor-ptr)
  (beta :float)
  (tensor th-float-tensor-ptr)
  (alpha :float)
  (vector1 th-float-tensor-ptr)
  (vector2 th-float-tensor-ptr))
;; void THTensor_(addbmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *batch1, THTensor *batch2);
(cffi:defcfun ("THFloatTensor_addbmm" th-float-tensor-add-bmm) :void
  (result th-float-tensor-ptr)
  (beta :float)
  (tensor th-float-tensor-ptr)
  (alpha :float)
  (batch1 th-float-tensor-ptr)
  (batch2 th-float-tensor-ptr))
;; void THTensor_(baddbmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *batch1, THTensor *batch2);
(cffi:defcfun ("THFloatTensor_baddbmm" th-float-tensor-badd-bmm) :void
  (result th-float-tensor-ptr)
  (beta :float)
  (tensor th-float-tensor-ptr)
  (alpha :float)
  (batch1 th-float-tensor-ptr)
  (batch2 th-float-tensor-ptr))

;; void THTensor_(match)(THTensor *r_, THTensor *m1, THTensor *m2, real gain);
(cffi:defcfun ("THFloatTensor_match" th-float-tensor-match) :void
  (result th-float-tensor-ptr)
  (m1 th-float-tensor-ptr)
  (m2 th-float-tensor-ptr)
  (gain :float))

;; ptrdiff_t THTensor_(numel)(THTensor *t);
(cffi:defcfun ("THFloatTensor_numel" th-float-tensor-numel) :long-long
  (tensor th-float-tensor-ptr))
;; void THTensor_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THFloatTensor_max" th-float-tensor-max) :void
  (values th-float-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THFloatTensor_min" th-float-tensor-min) :void
  (values th-float-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(kthvalue)(THTensor *values_, THLongTensor *indices_, THTensor *t, long k, int dimension, int keepdim);
(cffi:defcfun ("THFloatTensor_kthvalue" th-float-tensor-kth-value) :void
  (values th-float-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (k :long)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(mode)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THFloatTensor_mode" th-float-tensor-mode) :void
  (values th-float-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(median)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THFloatTensor_median" th-float-tensor-median) :void
  (values th-float-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(sum)(THTensor *r_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THFloatTensor_sum" th-float-tensor-sum) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(prod)(THTensor *r_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THFloatTensor_prod" th-float-tensor-prod) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(cumsum)(THTensor *r_, THTensor *t, int dimension);
(cffi:defcfun ("THFloatTensor_cumsum" th-float-tensor-cum-sum) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (dimension :int))
;; void THTensor_(cumprod)(THTensor *r_, THTensor *t, int dimension);
(cffi:defcfun ("THFloatTensor_cumprod" th-float-tensor-cum-prod) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (dimension :int))
;; void THTensor_(sign)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THFloatTensor_sign" th-float-tensor-sign) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr))
;; accreal THTensor_(trace)(THTensor *t);
(cffi:defcfun ("THFloatTensor_trace" th-float-tensor-trace) :double
  (tensor th-float-tensor-ptr))
;; void THTensor_(cross)(THTensor *r_, THTensor *a, THTensor *b, int dimension);
(cffi:defcfun ("THFloatTensor_cross" th-float-tensor-cross) :void
  (result th-float-tensor-ptr)
  (a th-float-tensor-ptr)
  (b th-float-tensor-ptr)
  (dimension :int))

;; void THTensor_(cmax)(THTensor *r, THTensor *t, THTensor *src);
(cffi:defcfun ("THFloatTensor_cmax" th-float-tensor-cmax) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(cmin)(THTensor *r, THTensor *t, THTensor *src);
(cffi:defcfun ("THFloatTensor_cmin" th-float-tensor-cmin) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(cmaxValue)(THTensor *r, THTensor *t, real value);
(cffi:defcfun ("THFloatTensor_cmaxValue" th-float-tensor-cmax-value) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))
;; void THTensor_(cminValue)(THTensor *r, THTensor *t, real value);
(cffi:defcfun ("THFloatTensor_cminValue" th-float-tensor-cmin-value) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))

;; void THTensor_(zeros)(THTensor *r_, THLongStorage *size);
(cffi:defcfun ("THFloatTensor_zeros" th-float-tensor-zeros) :void
  (result th-float-tensor-ptr)
  (size th-long-storage-ptr))
;; void THTensor_(zerosLike)(THTensor *r_, THTensor *input);
(cffi:defcfun ("THFloatTensor_zerosLike" th-float-tensor-zero-like) :void
  (result th-float-tensor-ptr)
  (input th-float-tensor-ptr))
;; void THTensor_(ones)(THTensor *r_, THLongStorage *size);
(cffi:defcfun ("THFloatTensor_ones" th-float-tensor-ones) :void
  (result th-float-tensor-ptr)
  (size th-long-storage-ptr))
;; void THTensor_(onesLike)(THTensor *r_, THTensor *input);
(cffi:defcfun ("THFloatTensor_onesLike" th-float-tensor-one-like) :void
  (result th-float-tensor-ptr)
  (input th-float-tensor-ptr))
;; void THTensor_(diag)(THTensor *r_, THTensor *t, int k);
(cffi:defcfun ("THFloatTensor_diag" th-float-tensor-diag) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (k :int))
;; void THTensor_(eye)(THTensor *r_, long n, long m);
(cffi:defcfun ("THFloatTensor_eye" th-float-tensor-eye) :void
  (result th-float-tensor-ptr)
  (n :long)
  (m :long))
;; void THTensor_(arange)(THTensor *r_, accreal xmin, accreal xmax, accreal step);
(cffi:defcfun ("THFloatTensor_arange" th-float-tensor-arange) :void
  (result th-float-tensor-ptr)
  (xmin :double)
  (xmax :double)
  (step :double))
;; void THTensor_(range)(THTensor *r_, accreal xmin, accreal xmax, accreal step);
(cffi:defcfun ("THFloatTensor_range" th-float-tensor-range) :void
  (result th-float-tensor-ptr)
  (xmin :double)
  (xmax :double)
  (step :double))
;; void THTensor_(randperm)(THTensor *r_, THGenerator *_generator, long n);
(cffi:defcfun ("THFloatTensor_randperm" th-float-tensor-rand-perm) :void
  (result th-float-tensor-ptr)
  (generator th-generator-ptr)
  (n :long))

;; void THTensor_(reshape)(THTensor *r_, THTensor *t, THLongStorage *size);
(cffi:defcfun ("THFloatTensor_reshape" th-float-tensor-reshape) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (size th-long-storage-ptr))
;; void THTensor_(sort)(THTensor *rt_, THLongTensor *ri_, THTensor *t, int dimension, int descendingOrder);
(cffi:defcfun ("THFloatTensor_sort" th-float-tensor-sort) :void
  (result-tensor th-float-tensor-ptr)
  (result-indices th-long-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (dimension :int)
  (discending-order :int))
;; void THTensor_(topk)(THTensor *rt_, THLongTensor *ri_, THTensor *t, long k, int dim, int dir, int sorted);
(cffi:defcfun ("THFloatTensor_topk" th-float-tensor-topk) :void
  (result-tensor th-float-tensor-ptr)
  (result-indices th-long-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (k :long)
  (dim :int)
  (dir :int)
  (sorted :int))
;; void THTensor_(tril)(THTensor *r_, THTensor *t, long k);
(cffi:defcfun ("THFloatTensor_tril" th-float-tensor-tril) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (k :long))
;; void THTensor_(triu)(THTensor *r_, THTensor *t, long k);
(cffi:defcfun ("THFloatTensor_triu" th-float-tensor-triu) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (k :long))
;; void THTensor_(cat)(THTensor *r_, THTensor *ta, THTensor *tb, int dimension);
(cffi:defcfun ("THFloatTensor_cat" th-float-tensor-cat) :void
  (result th-float-tensor-ptr)
  (tensora th-float-tensor-ptr)
  (tensorb th-float-tensor-ptr)
  (dimension :int))
;; void THTensor_(catArray)(THTensor *result, THTensor **inputs, int numInputs, int dimension);
(cffi:defcfun ("THFloatTensor_catArray" th-float-tensor-cat-array) :void
  (result th-float-tensor-ptr)
  (inputs (:pointer th-float-tensor-ptr))
  (num-inputs :int)
  (dimension :int))

;; int THTensor_(equal)(THTensor *ta, THTensor *tb);
(cffi:defcfun ("THFloatTensor_equal" th-float-tensor-equal) :int
  (tensora th-float-tensor-ptr)
  (tensorb th-float-tensor-ptr))

;; void THTensor_(ltValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THFloatTensor_ltValue" th-float-tensor-lt-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))
;; void THTensor_(leValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THFloatTensor_leValue" th-float-tensor-le-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))
;; void THTensor_(gtValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THFloatTensor_gtValue" th-float-tensor-gt-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))
;; void THTensor_(geValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THFloatTensor_geValue" th-float-tensor-ge-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))
;; void THTensor_(neValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THFloatTensor_neValue" th-float-tensor-ne-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))
;; void THTensor_(eqValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THFloatTensor_eqValue" th-float-tensor-eq-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))

;; void THTensor_(ltValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THFloatTensor_ltValueT" th-float-tensor-lt-value-t) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))
;; void THTensor_(leValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THFloatTensor_leValueT" th-float-tensor-le-value-t) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))
;; void THTensor_(gtValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THFloatTensor_gtValueT" th-float-tensor-gt-value-t) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))
;; void THTensor_(geValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THFloatTensor_geValueT" th-float-tensor-ge-value-t) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))
;; void THTensor_(neValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THFloatTensor_neValueT" th-float-tensor-ne-value-t) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))
;; void THTensor_(eqValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THFloatTensor_eqValueT" th-float-tensor-eq-value-t) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))

;; void THTensor_(ltTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THFloatTensor_ltTensor" th-float-tensor-lt-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-float-tensor-ptr)
  (tensorb th-float-tensor-ptr))
;; void THTensor_(leTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THFloatTensor_leTensor" th-float-tensor-le-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-float-tensor-ptr)
  (tensorb th-float-tensor-ptr))
;; void THTensor_(gtTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THFloatTensor_gtTensor" th-float-tensor-gt-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-float-tensor-ptr)
  (tensorb th-float-tensor-ptr))
;; void THTensor_(geTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THFloatTensor_geTensor" th-float-tensor-ge-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-float-tensor-ptr)
  (tensorb th-float-tensor-ptr))
;; void THTensor_(neTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THFloatTensor_neTensor" th-float-tensor-ne-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-float-tensor-ptr)
  (tensorb th-float-tensor-ptr))
;; void THTensor_(eqTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THFloatTensor_eqTensor" th-float-tensor-eq-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-float-tensor-ptr)
  (tensorb th-float-tensor-ptr))

;; void THTensor_(ltTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THFloatTensor_ltTensorT" th-float-tensor-lt-tensor-t) :void
  (result th-float-tensor-ptr)
  (tensora th-float-tensor-ptr)
  (tensorb th-float-tensor-ptr))
;; void THTensor_(leTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THFloatTensor_leTensorT" th-float-tensor-le-tensor-t) :void
  (result th-float-tensor-ptr)
  (tensora th-float-tensor-ptr)
  (tensorb th-float-tensor-ptr))
;; void THTensor_(gtTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THFloatTensor_gtTensorT" th-float-tensor-gt-tensor-t) :void
  (result th-float-tensor-ptr)
  (tensora th-float-tensor-ptr)
  (tensorb th-float-tensor-ptr))
;; void THTensor_(geTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THFloatTensor_geTensorT" th-float-tensor-ge-tensor-t) :void
  (result th-float-tensor-ptr)
  (tensora th-float-tensor-ptr)
  (tensorb th-float-tensor-ptr))
;; void THTensor_(neTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THFloatTensor_neTensorT" th-float-tensor-ne-tensor-t) :void
  (result th-float-tensor-ptr)
  (tensora th-float-tensor-ptr)
  (tensorb th-float-tensor-ptr))
;; void THTensor_(eqTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THFloatTensor_eqTensorT" th-float-tensor-eq-tensor-t) :void
  (result th-float-tensor-ptr)
  (tensora th-float-tensor-ptr)
  (tensorb th-float-tensor-ptr))

;; #if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

;; void THTensor_(sigmoid)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THFloatTensor_sigmoid" th-float-tensor-sigmoid) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr))
;; void THTensor_(log)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THFloatTensor_log" th-float-tensor-log) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr))
;; void THTensor_(lgamma)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THFloatTensor_lgamma" th-float-tensor-lgamma) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr))
;; void THTensor_(log1p)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THFloatTensor_log1p" th-float-tensor-log1p) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr))
;; void THTensor_(exp)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THFloatTensor_exp" th-float-tensor-exp) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr))
;; void THTensor_(cos)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THFloatTensor_cos" th-float-tensor-cos) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr))
;; void THTensor_(acos)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THFloatTensor_acos" th-float-tensor-acos) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr))
;; void THTensor_(cosh)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THFloatTensor_cosh" th-float-tensor-cosh) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr))
;; void THTensor_(sin)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THFloatTensor_sin" th-float-tensor-sin) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr))
;; void THTensor_(asin)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THFloatTensor_asin" th-float-tensor-asin) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr))
;; void THTensor_(sinh)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THFloatTensor_sinh" th-float-tensor-sinh) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr))
;; void THTensor_(tan)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THFloatTensor_tan" th-float-tensor-tan) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr))
;; void THTensor_(atan)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THFloatTensor_atan" th-float-tensor-atan) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr))
;; void THTensor_(atan2)(THTensor *r_, THTensor *tx, THTensor *ty);
(cffi:defcfun ("THFloatTensor_atan2" th-float-tensor-atan2) :void
  (result th-float-tensor-ptr)
  (tensorx th-float-tensor-ptr)
  (tensory th-float-tensor-ptr))
;; void THTensor_(tanh)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THFloatTensor_tanh" th-float-tensor-tanh) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr))
;; void THTensor_(pow)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THFloatTensor_pow" th-float-tensor-pow) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float))
;; void THTensor_(tpow)(THTensor *r_, real value, THTensor *t);
(cffi:defcfun ("THFloatTensor_tpow" th-float-tensor-tpow) :void
  (result th-float-tensor-ptr)
  (value :float)
  (tensor th-float-tensor-ptr))
;; void THTensor_(sqrt)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THFloatTensor_sqrt" th-float-tensor-sqrt) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr))
;; void THTensor_(rsqrt)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THFloatTensor_rsqrt" th-float-tensor-rsqrt) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr))
;; void THTensor_(ceil)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THFloatTensor_ceil" th-float-tensor-ceil) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr))
;; void THTensor_(floor)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THFloatTensor_floor" th-float-tensor-floor) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr))
;; void THTensor_(round)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THFloatTensor_round" th-float-tensor-round) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr))
;; void THTensor_(abs)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THFloatTensor_abs" th-float-tensor-abs) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr))
;; void THTensor_(trunc)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THFloatTensor_trunc" th-float-tensor-trunc) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr))
;; void THTensor_(frac)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THFloatTensor_frac" th-float-tensor-frac) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr))
;; void THTensor_(lerp)(THTensor *r_, THTensor *a, THTensor *b, real weight);
(cffi:defcfun ("THFloatTensor_lerp" th-float-tensor-lerp) :void
  (result th-float-tensor-ptr)
  (a th-float-tensor-ptr)
  (b th-float-tensor-ptr)
  (weight :float))

;; void THTensor_(mean)(THTensor *r_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THFloatTensor_mean" th-float-tensor-mean) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(std)(THTensor *r_, THTensor *t, int dimension, int biased, int keepdim);
(cffi:defcfun ("THFloatTensor_std" th-float-tensor-std) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (dimension :int)
  (biased :int)
  (keep-dim :int))
;; void THTensor_(var)(THTensor *r_, THTensor *t, int dimension, int biased, int keepdim);
(cffi:defcfun ("THFloatTensor_var" th-float-tensor-var) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (dimension :int)
  (biased :int)
  (keep-dim :int))
;; void THTensor_(norm)(THTensor *r_, THTensor *t, real value, int dimension, int keepdim);
(cffi:defcfun ("THFloatTensor_norm" th-float-tensor-norm) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(renorm)(THTensor *r_, THTensor *t, real value, int dimension, real maxnorm);
(cffi:defcfun ("THFloatTensor_renorm" th-float-tensor-renorm) :void
  (result th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (value :float)
  (dimension :int)
  (maxnorm :float))
;; accreal THTensor_(dist)(THTensor *a, THTensor *b, real value);
(cffi:defcfun ("THFloatTensor_dist" th-float-tensor-dist) :double
  (a th-float-tensor-ptr)
  (b th-float-tensor-ptr)
  (value :float))
;; void THTensor_(histc)(THTensor *hist, THTensor *tensor, long nbins, real minvalue, real maxvalue)
(cffi:defcfun ("THFloatTensor_histc" th-float-tensor-histc) :void
  (hist th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (nbins :long)
  (min-value :float)
  (max-value :float))
;; void THTensor_(bhistc)(THTensor *hist, THTensor *tensor, long nbins, real minvalue, real maxvalue);
(cffi:defcfun ("THFloatTensor_bhistc" th-float-tensor-bhistc) :void
  (hist th-float-tensor-ptr)
  (tensor th-float-tensor-ptr)
  (nbins :long)
  (min-value :float)
  (max-value :float))

;; accreal THTensor_(meanall)(THTensor *self);
(cffi:defcfun ("THFloatTensor_meanall" th-float-tensor-mean-all) :double
  (tensor th-float-tensor-ptr))
;; accreal THTensor_(varall)(THTensor *self, int biased);
(cffi:defcfun ("THFloatTensor_varall" th-float-tensor-var-all) :double
  (tensor th-float-tensor-ptr)
  (biased :int))
;; accreal THTensor_(stdall)(THTensor *self, int biased);
(cffi:defcfun ("THFloatTensor_stdall" th-float-tensor-std-all) :double
  (tensor th-float-tensor-ptr)
  (biased :int))
;; accreal THTensor_(normall)(THTensor *t, real value);
(cffi:defcfun ("THFloatTensor_normall" th-float-tensor-norm-all) :double
  (tensor th-float-tensor-ptr)
  (value :float))

;; void THTensor_(linspace)(THTensor *r_, real a, real b, long n);
(cffi:defcfun ("THFloatTensor_linspace" th-float-tensor-linspace) :void
  (result th-float-tensor-ptr)
  (a :float)
  (b :float)
  (n :long))
;; void THTensor_(logspace)(THTensor *r_, real a, real b, long n);
(cffi:defcfun ("THFloatTensor_logspace" th-float-tensor-logspace) :void
  (result th-float-tensor-ptr)
  (a :float)
  (b :float)
  (n :long))
;; void THTensor_(rand)(THTensor *r_, THGenerator *_generator, THLongStorage *size);
(cffi:defcfun ("THFloatTensor_rand" th-float-tensor-rand) :void
  (result th-float-tensor-ptr)
  (generator th-generator-ptr)
  (size th-long-storage-ptr))
;; void THTensor_(randn)(THTensor *r_, THGenerator *_generator, THLongStorage *size);
(cffi:defcfun ("THFloatTensor_randn" th-float-tensor-randn) :void
  (result th-float-tensor-ptr)
  (generator th-generator-ptr)
  (size th-long-storage-ptr))

;; #endif /* FLOAT OR DOUBLE */

;; #if defined(TH_REAL_IS_BYTE)

;; int THTensor_(logicalall)(THTensor *self);
;; (cffi:defcfun ("THFloatTensor_logicalall" th-float-tensor-logical-all) :int
;;   (tensor (:pointer :void)))
;; int THTensor_(logicalany)(THTensor *self);
;; (cffi:defcfun ("THFloatTensor_logicalany" th-float-tensor-logical-any) :int
;;   (tensor (:pointer :void)))

;; #endif /* TH_REAL_IS_BYTE */


;; void THTensor_(validXCorr2Dptr)(real *r_,
;;                                 real alpha,
;;                                 real *t_, long ir, long ic,
;;                                 real *k_, long kr, long kc,
;;                                 long sr, long sc);
(cffi:defcfun ("THFloatTensor_validXCorr2Dptr" th-float-tensor-valid-x-corr-2d-ptr) :void
  (res (:pointer :float))
  (alpha :float)
  (ten (:pointer :float))
  (ir :long)
  (ic :long)
  (k (:pointer :float))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(validConv2Dptr)(real *r_,
;;                                real alpha,
;;                                real *t_, long ir, long ic,
;;                                real *k_, long kr, long kc,
;;                                long sr, long sc);
(cffi:defcfun ("THFloatTensor_validConv2Dptr" th-float-tensor-valid-conv-2d-ptr) :void
  (res (:pointer :float))
  (alpha :float)
  (ten (:pointer :float))
  (ir :long)
  (ic :long)
  (k (:pointer :float))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(fullXCorr2Dptr)(real *r_,
;;                                real alpha,
;;                                real *t_, long ir, long ic,
;;                                real *k_, long kr, long kc,
;;                                long sr, long sc);
(cffi:defcfun ("THFloatTensor_fullXCorr2Dptr" th-float-tensor-full-x-corr-2d-ptr) :void
  (res (:pointer :float))
  (alpha :float)
  (ten (:pointer :float))
  (ir :long)
  (ic :long)
  (k (:pointer :float))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(fullConv2Dptr)(real *r_,
;;                               real alpha,
;;                               real *t_, long ir, long ic,
;;                               real *k_, long kr, long kc,
;;                               long sr, long sc);
(cffi:defcfun ("THFloatTensor_fullConv2Dptr" th-float-tensor-full-conv-2d-ptr) :void
  (res (:pointer :float))
  (alpha :float)
  (ten (:pointer :float))
  (ir :long)
  (ic :long)
  (k (:pointer :float))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(validXCorr2DRevptr)(real *r_,
;;                                    real alpha,
;;                                    real *t_, long ir, long ic,
;;                                    real *k_, long kr, long kc,
;;                                    long sr, long sc);
(cffi:defcfun ("THFloatTensor_validXCorr2DRevptr" th-float-tensor-valid-x-corr-2d-rev-ptr) :void
  (res (:pointer :float))
  (alpha :float)
  (ten (:pointer :float))
  (ir :long)
  (ic :long)
  (k (:pointer :float))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(conv2DRevger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol);
(cffi:defcfun ("THFloatTensor_conv2DRevger" th-float-tensor-conv-2d-rev-ger) :void
  (result th-float-tensor-ptr)
  (beta :float)
  (alpha :float)
  (tensor th-float-tensor-ptr)
  (k th-float-tensor-ptr)
  (srow :long)
  (scol :long))
;; void THTensor_(conv2DRevgerm)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol);
(cffi:defcfun ("THFloatTensor_conv2DRevgerm" th-float-tensor-conv-2d-rev-germ) :void
  (result th-float-tensor-ptr)
  (beta :float)
  (alpha :float)
  (tensor th-float-tensor-ptr)
  (k th-float-tensor-ptr)
  (srow :long)
  (scol :long))
;; void THTensor_(conv2Dger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THFloatTensor_conv2Dger" th-float-tensor-conv-2d-ger) :void
  (result th-float-tensor-ptr)
  (beta :float)
  (alpha :float)
  (tensor th-float-tensor-ptr)
  (k th-float-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf :string)
  (xc :string))
;; void THTensor_(conv2Dmv)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THFloatTensor_conv2Dmv" th-float-tensor-conv-2d-mv) :void
  (result th-float-tensor-ptr)
  (beta :float)
  (alpha :float)
  (tensor th-float-tensor-ptr)
  (k th-float-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf :string)
  (xc :string))
;; void THTensor_(conv2Dmm)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THFloatTensor_conv2Dmm" th-float-tensor-conv-2d-mm) :void
  (result th-float-tensor-ptr)
  (beta :float)
  (alpha :float)
  (tensor th-float-tensor-ptr)
  (k th-float-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf :string)
  (xc :string))
;; void THTensor_(conv2Dmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THFloatTensor_conv2Dmul" th-float-tensor-conv-2d-mul) :void
  (result th-float-tensor-ptr)
  (beta :float)
  (alpha :float)
  (tensor th-float-tensor-ptr)
  (k th-float-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf :string)
  (xc :string))
;; void THTensor_(conv2Dcmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THFloatTensor_conv2Dcmul" th-float-tensor-conv-2d-cmul) :void
  (result th-float-tensor-ptr)
  (beta :float)
  (alpha :float)
  (tensor th-float-tensor-ptr)
  (k th-float-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf :string)
  (xc :string))

;; void THTensor_(validXCorr3Dptr)(real *r_,
;;                                 real alpha,
;;                                 real *t_, long it, long ir, long ic,
;;                                 real *k_, long kt, long kr, long kc,
;;                                 long st, long sr, long sc);
(cffi:defcfun ("THFloatTensor_validXCorr3Dptr" th-float-tensor-valid-x-corr-3d-ptr) :void
  (res (:pointer :float))
  (alpha :float)
  (ten (:pointer :float))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :float))
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
(cffi:defcfun ("THFloatTensor_validConv3Dptr" th-float-tensor-valid-conv-3d-ptr) :void
  (res (:pointer :float))
  (alpha :float)
  (ten (:pointer :float))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :float))
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
(cffi:defcfun ("THFloatTensor_fullXCorr3Dptr" th-float-tensor-full-x-corr-3d-ptr) :void
  (res (:pointer :float))
  (alpha :float)
  (ten (:pointer :float))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :float))
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
(cffi:defcfun ("THFloatTensor_fullConv3Dptr" th-float-tensor-full-conv-3d-ptr) :void
  (res (:pointer :float))
  (alpha :float)
  (ten (:pointer :float))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :float))
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
(cffi:defcfun ("THFloatTensor_validXCorr3DRevptr" th-float-tensor-valid-x-corr-3d-rev-ptr) :void
  (res (:pointer :float))
  (alpha :float)
  (ten (:pointer :float))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :float))
  (kt :long)
  (kr :long)
  (kc :long)
  (st :long)
  (sr :long)
  (sc :long))

;; void THTensor_(conv3DRevger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol);
(cffi:defcfun ("THFloatTensor_conv3DRevger" th-float-tensor-conv-3d-rev-ger) :void
  (result th-float-tensor-ptr)
  (beta :float)
  (alpha :float)
  (tensor th-float-tensor-ptr)
  (k th-float-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long))
;; void THTensor_(conv3Dger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THFloatTensor_conv3Dger" th-float-tensor-conv-3d-ger) :void
  (result th-float-tensor-ptr)
  (beta :float)
  (alpha :float)
  (tensor th-float-tensor-ptr)
  (k th-float-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf :string)
  (xc :string))
;; void THTensor_(conv3Dmv)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THFloatTensor_conv3Dmv" th-float-tensor-conv-3d-mv) :void
  (result th-float-tensor-ptr)
  (beta :float)
  (alpha :float)
  (tensor th-float-tensor-ptr)
  (k th-float-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf :string)
  (xc :string))
;; void THTensor_(conv3Dmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THFloatTensor_conv3Dmul" th-float-tensor-conv-3d-mul) :void
  (result th-float-tensor-ptr)
  (beta :float)
  (alpha :float)
  (tensor th-float-tensor-ptr)
  (k th-float-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf :string)
  (xc :string))
;; void THTensor_(conv3Dcmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THFloatTensor_conv3Dcmul" th-float-tensor-conv-3d-cmul) :void
  (result th-float-tensor-ptr)
  (beta :float)
  (alpha :float)
  (tensor th-float-tensor-ptr)
  (k th-float-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf :string)
  (xc :string))

;; /* Level 1 */
;; void THBlas_(swap)(long n, real *x, long incx, real *y, long incy);
(cffi:defcfun ("THDoubleBlas_swap" th-double-blas-swap) :void
  (n :long)
  (x (:pointer :float))
  (incx :long)
  (y (:pointer :float))
  (incy :long))
;; void THBlas_(scal)(long n, real a, real *x, long incx);
(cffi:defcfun ("THDoubleBlas_scal" th-double-blas-scal) :void
  (n :long)
  (a :float)
  (x (:pointer :float))
  (incx :long))
;; void THBlas_(copy)(long n, real *x, long incx, real *y, long incy);
(cffi:defcfun ("THDoubleBlas_copy" th-double-blas-copy) :void
  (n :long)
  (x (:pointer :float))
  (incx :long)
  (y (:pointer :float))
  (incy :long))
;; void THBlas_(axpy)(long n, real a, real *x, long incx, real *y, long incy);
(cffi:defcfun ("THDoubleBlas_axpy" th-double-blas-axpy) :void
  (n :long)
  (a :float)
  (x (:pointer :float))
  (incx :long)
  (y (:pointer :float))
  (incy :long))
;; real THBlas_(dot)(long n, real *x, long incx, real *y, long incy);
(cffi:defcfun ("THDoubleBlas_dot" th-double-blas-dot) :void
  (n :long)
  (x (:pointer :float))
  (incx :long)
  (y (:pointer :float))
  (incy :long))

;; /* Level 2 */
;; void THBlas_(gemv)(char trans, long m, long n, real alpha, real *a, long lda, real *x, long incx, real beta, real *y, long incy);
(cffi:defcfun ("THDoubleBlas_gemv" th-double-blas-gemv) :void
  (trans :char)
  (m :long)
  (n :long)
  (alpha :float)
  (a (:pointer :float))
  (lda :long)
  (x (:pointer :float))
  (incx :long)
  (beta :float)
  (y (:pointer :float))
  (incy :long))
;; void THBlas_(ger)(long m, long n, real alpha, real *x, long incx, real *y, long incy, real *a, long lda);
(cffi:defcfun ("THDoubleBlas_ger" th-double-blas-ger) :void
  (m :long)
  (n :long)
  (alpha :float)
  (x (:pointer :float))
  (incx :long)
  (y (:pointer :float))
  (incy :long)
  (a (:pointer :float))
  (lda :long))

;; /* Level 3 */
;; void THBlas_(gemm)(char transa, char transb, long m, long n, long k, real alpha, real *a, long lda, real *b, long ldb, real beta, real *c, long ldc);
(cffi:defcfun ("THDoubleBlas_gemm" th-double-blas-gemm) :void
  (transa :char)
  (transb :char)
  (m :long)
  (n :long)
  (k :long)
  (alpha :float))

;; void THTensor_(gesv)(THTensor *rb_, THTensor *ra_, THTensor *b_, THTensor *a_);
(cffi:defcfun ("THFloatTensor_gesv" th-float-tensor-gesv) :void
  (resultb th-float-tensor-ptr)
  (resulta th-float-tensor-ptr)
  (b th-float-tensor-ptr)
  (a th-float-tensor-ptr))
;; void THTensor_(trtrs)(THTensor *rb_, THTensor *ra_, THTensor *b_, THTensor *a_, const char *uplo, const char *trans, const char *diag);
(cffi:defcfun ("THFloatTensor_trtrs" th-float-tensor-trtrs) :void
  (resultb th-float-tensor-ptr)
  (resulta th-float-tensor-ptr)
  (b th-float-tensor-ptr)
  (a th-float-tensor-ptr)
  (uplo :string)
  (trans :string)
  (diag :string))
;; void THTensor_(gels)(THTensor *rb_, THTensor *ra_, THTensor *b_, THTensor *a_);
(cffi:defcfun ("THFloatTensor_gels" th-float-tensor-gels) :void
  (resultb th-float-tensor-ptr)
  (resulta th-float-tensor-ptr)
  (b th-float-tensor-ptr)
  (a th-float-tensor-ptr))
;; void THTensor_(syev)(THTensor *re_, THTensor *rv_, THTensor *a_, const char *jobz, const char *uplo);
(cffi:defcfun ("THFloatTensor_syev" th-float-tensor-syev) :void
  (resulte th-float-tensor-ptr)
  (resultv th-float-tensor-ptr)
  (a th-float-tensor-ptr)
  (jobz :string)
  (uplo :string))
;; void THTensor_(geev)(THTensor *re_, THTensor *rv_, THTensor *a_, const char *jobvr);
(cffi:defcfun ("THFloatTensor_geev" th-float-tensor-geev) :void
  (resulte th-float-tensor-ptr)
  (resultv th-float-tensor-ptr)
  (a th-float-tensor-ptr)
  (jobvr :string))
;; void THTensor_(gesvd)(THTensor *ru_, THTensor *rs_, THTensor *rv_, THTensor *a, const char *jobu)
(cffi:defcfun ("THFloatTensor_gesvd" th-float-tensor-gesvd) :void
  (resultu th-float-tensor-ptr)
  (results th-float-tensor-ptr)
  (resultv th-float-tensor-ptr)
  (a th-float-tensor-ptr)
  (jobu :string))
;; void THTensor_(gesvd2)(THTensor *ru_, THTensor *rs_, THTensor *rv_, THTensor *ra_, THTensor *a, const char *jobu);
(cffi:defcfun ("THFloatTensor_gesvd2" th-float-tensor-gesvd2) :void
  (resultu th-float-tensor-ptr)
  (results th-float-tensor-ptr)
  (resultv th-float-tensor-ptr)
  (resulta th-float-tensor-ptr)
  (a th-float-tensor-ptr)
  (jobu :string))
;; void THTensor_(getri)(THTensor *ra_, THTensor *a);
(cffi:defcfun ("THFloatTensor_getri" th-float-tensor-getri) :void
  (resulta th-float-tensor-ptr)
  (a th-float-tensor-ptr))
;; void THTensor_(potrf)(THTensor *ra_, THTensor *a, const char *uplo);
(cffi:defcfun ("THFloatTensor_potrf" th-float-tensor-potrf) :void
  (resulta th-float-tensor-ptr)
  (a th-float-tensor-ptr)
  (uplo :string))
;; void THTensor_(potrs)(THTensor *rb_, THTensor *b_, THTensor *a_,  const char *uplo);
(cffi:defcfun ("THFloatTensor_potrs" th-float-tensor-potrs) :void
  (resultb th-float-tensor-ptr)
  (b th-float-tensor-ptr)
  (a th-float-tensor-ptr)
  (uplo :string))
;; void THTensor_(potri)(THTensor *ra_, THTensor *a, const char *uplo);
(cffi:defcfun ("THFloatTensor_potri" th-float-tensor-potri) :void
  (resulta th-float-tensor-ptr)
  (a th-float-tensor-ptr)
  (uplo :string))
;; void THTensor_(qr)(THTensor *rq_, THTensor *rr_, THTensor *a);
(cffi:defcfun ("THFloatTensor_qr" th-float-tensor-qr) :void
  (resultq th-float-tensor-ptr)
  (resultr th-float-tensor-ptr)
  (a th-float-tensor-ptr))
;; void THTensor_(geqrf)(THTensor *ra_, THTensor *rtau_, THTensor *a);
(cffi:defcfun ("THFloatTensor_geqrf" th-float-tensor-geqrf) :void
  (resulta th-float-tensor-ptr)
  (resulttau th-float-tensor-ptr)
  (a th-float-tensor-ptr))
;; void THTensor_(orgqr)(THTensor *ra_, THTensor *a, THTensor *tau);
(cffi:defcfun ("THFloatTensor_orgqr" th-float-tensor-orgqr) :void
  (resulta th-float-tensor-ptr)
  (a th-float-tensor-ptr)
  (tau th-float-tensor-ptr))
;; void THTensor_(ormqr)(THTensor *ra_, THTensor *a, THTensor *tau, THTensor *c, const char *side, const char *trans);
(cffi:defcfun ("THFloatTensor_ormqr" th-float-tensor-ormqr) :void
  (resulta th-float-tensor-ptr)
  (a th-float-tensor-ptr)
  (tau th-float-tensor-ptr)
  (c th-float-tensor-ptr)
  (side :string)
  (trans :string))
;; void THTensor_(pstrf)(THTensor *ra_, THIntTensor *rpiv_, THTensor*a, const char* uplo, real tol);
(cffi:defcfun ("THFloatTensor_pstrf" th-float-tensor-pstrf) :void
  (resulta th-float-tensor-ptr)
  (resultpiv th-int-tensor-ptr)
  (a th-float-tensor-ptr)
  (uplo :string)
  (tol :float))

;; void THTensor_(btrifact)(THTensor *ra_, THIntTensor *rpivots_, THIntTensor *rinfo_, int pivot, THTensor *a);
(cffi:defcfun ("THFloatTensor_btrifact" th-float-tensor-btrifact) :void
  (resulta th-float-tensor-ptr)
  (resultpivots th-int-tensor-ptr)
  (resultinfo th-int-tensor-ptr)
  (pivot :int)
  (a th-float-tensor-ptr))
;; void THTensor_(btrisolve)(THTensor *rb_, THTensor *b, THTensor *atf, THIntTensor *pivots);
(cffi:defcfun ("THFloatTensor_btrisolve" th-float-tensor-btrisolve) :void
  (resultb th-float-tensor-ptr)
  (b th-float-tensor-ptr)
  (atf th-float-tensor-ptr)
  (pivots th-int-tensor-ptr))
