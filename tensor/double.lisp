(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

;; ACCESS METHODS
;; THStorage* THTensor_(storage)(const THTensor *self)
(cffi:defcfun ("THDoubleTensor_storage" th-double-tensor-storage) th-double-storage-ptr
  (tensor th-double-tensor-ptr))
;; ptrdiff_t THTensor_(storageOffset)(const THTensor *self)
(cffi:defcfun ("THDoubleTensor_storageOffset" th-double-tensor-storage-offset) :long-long
  (tensor th-double-tensor-ptr))
;; int THTensor_(nDimension)(const THTensor *self)
(cffi:defcfun ("THDoubleTensor_nDimension" th-double-tensor-n-dimension) :int
  (tensor th-double-tensor-ptr))
;; long THTensor_(size)(const THTensor *self, int dim)
(cffi:defcfun ("THDoubleTensor_size" th-double-tensor-size) :long
  (tensor th-double-tensor-ptr)
  (dim :int))
;; long THTensor_(stride)(const THTensor *self, int dim)
(cffi:defcfun ("THDoubleTensor_stride" th-double-tensor-stride) :long
  (tensor th-double-tensor-ptr)
  (dim :int))
;; THLongStorage *THTensor_(newSizeOf)(THTensor *self)
(cffi:defcfun ("THDoubleTensor_newSizeOf" th-double-tensor-new-size-of) th-long-storage-ptr
  (tensor th-double-tensor-ptr))
;; THLongStorage *THTensor_(newStrideOf)(THTensor *self)
(cffi:defcfun ("THDoubleTensor_newStrideOf" th-double-tensor-new-stride-of) th-long-storage-ptr
  (tensor th-double-tensor-ptr))
;; real *THTensor_(data)(const THTensor *self)
(cffi:defcfun ("THDoubleTensor_data" th-double-tensor-data) (:pointer :double)
  (tensor th-double-tensor-ptr))

;; void THTensor_(setFlag)(THTensor *self, const char flag)
(cffi:defcfun ("THDoubleTensor_setFlag" th-double-tensor-set-flag) :void
  (tensor th-double-tensor-ptr)
  (flag :char))
;; void THTensor_(clearFlag)(THTensor *self, const char flag)
(cffi:defcfun ("THDoubleTensor_clearFlag" th-double-tensor-clear-flag) :void
  (tensor th-double-tensor-ptr)
  (flag :char))

;; CREATION METHODS
;; THTensor *THTensor_(new)(void)
(cffi:defcfun ("THDoubleTensor_new" th-double-tensor-new) th-double-tensor-ptr)
;; THTensor *THTensor_(newWithTensor)(THTensor *tensor)
(cffi:defcfun ("THDoubleTensor_newWithTensor" th-double-tensor-new-with-tensor) th-double-tensor-ptr
  (tensor th-double-tensor-ptr))
;; stride might be NULL
;; THTensor *THTensor_(newWithStorage)(THStorage *storage_, ptrdiff_t storageOffset_,
;;                                            THLongStorage *size_, THLongStorage *stride_)
(cffi:defcfun ("THDoubleTensor_newWithStorage" th-double-tensor-new-with-storage)
    th-double-tensor-ptr
  (storage th-double-storage-ptr)
  (storage-offset :long-long)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
;; THTensor *THTensor_(newWithStorage1d)(THStorage *storage_, ptrdiff_t storageOffset_,
;;                                              long size0_, long stride0_);
(cffi:defcfun ("THDoubleTensor_newWithStorage1d" th-double-tensor-new-with-storage-1d)
    th-double-tensor-ptr
  (storage th-double-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long))
(cffi:defcfun ("THDoubleTensor_newWithStorage2d" th-double-tensor-new-with-storage-2d)
    th-double-tensor-ptr
  (storage th-double-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long))
(cffi:defcfun ("THDoubleTensor_newWithStorage3d" th-double-tensor-new-with-storage-3d)
    th-double-tensor-ptr
  (storage th-double-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long)
  (size2 :long)
  (stride2 :long))
(cffi:defcfun ("THDoubleTensor_newWithStorage4d" th-double-tensor-new-with-storage-4d)
    th-double-tensor-ptr
  (storage th-double-storage-ptr)
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
(cffi:defcfun ("THDoubleTensor_newWithSize" th-double-tensor-new-with-size) th-double-tensor-ptr
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
;; THTensor *THTensor_(newWithSize1d)(long size0_);
(cffi:defcfun ("THDoubleTensor_newWithSize1d" th-double-tensor-new-with-size-1d)
    th-double-tensor-ptr
  (size0 :long))
(cffi:defcfun ("THDoubleTensor_newWithSize2d" th-double-tensor-new-with-size-2d)
    th-double-tensor-ptr
  (size0 :long)
  (size1 :long))
(cffi:defcfun ("THDoubleTensor_newWithSize3d" th-double-tensor-new-with-size-3d)
    th-double-tensor-ptr
  (size0 :long)
  (size1 :long)
  (size2 :long))
(cffi:defcfun ("THDoubleTensor_newWithSize4d" th-double-tensor-new-with-size-4d)
    th-double-tensor-ptr
  (size0 :long)
  (size1 :long)
  (size2 :long)
  (size3 :long))

;; THTensor *THTensor_(newClone)(THTensor *self)
(cffi:defcfun ("THDoubleTensor_newClone" th-double-tensor-new-clone) th-double-tensor-ptr
  (tensor th-double-tensor-ptr))
(cffi:defcfun ("THDoubleTensor_newContiguous" th-double-tensor-new-contiguous) th-double-tensor-ptr
  (tensor th-double-tensor-ptr))
(cffi:defcfun ("THDoubleTensor_newSelect" th-double-tensor-new-select) th-double-tensor-ptr
  (tensor th-double-tensor-ptr)
  (dimension :int)
  (slice-index :long))
(cffi:defcfun ("THDoubleTensor_newNarrow" th-double-tensor-new-narrow) th-double-tensor-ptr
  (tensor th-double-tensor-ptr)
  (dimension :int)
  (first-index :long)
  (size :long))

(cffi:defcfun ("THDoubleTensor_newTranspose" th-double-tensor-new-transpose) th-double-tensor-ptr
  (tensor th-double-tensor-ptr)
  (dimension1 :int)
  (dimension2 :int))
(cffi:defcfun ("THDoubleTensor_newUnfold" th-double-tensor-new-unfold) th-double-tensor-ptr
  (tensor th-double-tensor-ptr)
  (dimension :int)
  (size :long)
  (step :long))
(cffi:defcfun ("THDoubleTensor_newView" th-double-tensor-new-view) th-double-tensor-ptr
  (tensor th-double-tensor-ptr)
  (size th-long-storage-ptr))

(cffi:defcfun ("THDoubleTensor_expand" th-double-tensor-expand) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (size th-long-storage-ptr))

(cffi:defcfun ("THDoubleTensor_resize" th-double-tensor-resize) :void
  (tensor th-double-tensor-ptr)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
(cffi:defcfun ("THDoubleTensor_resizeAs" th-double-tensor-resize-as) :void
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr))
(cffi:defcfun ("THDoubleTensor_resizeNd" th-double-tensor-resize-nd) :void
  (tensor th-double-tensor-ptr)
  (dimension :int)
  (size (:pointer :long))
  (stride (:pointer :long)))
(cffi:defcfun ("THDoubleTensor_resize1d" th-double-tensor-resize-1d) :void
  (tensor th-double-tensor-ptr)
  (size0 :long))
(cffi:defcfun ("THDoubleTensor_resize2d" th-double-tensor-resize-2d) :void
  (tensor th-double-tensor-ptr)
  (size0 :long)
  (size1 :long))
(cffi:defcfun ("THDoubleTensor_resize3d" th-double-tensor-resize-3d) :void
  (tensor th-double-tensor-ptr)
  (size0 :long)
  (size1 :long)
  (size2 :long))
(cffi:defcfun ("THDoubleTensor_resize4d" th-double-tensor-resize-4d) :void
  (tensor th-double-tensor-ptr)
  (size0 :long)
  (size1 :long)
  (size2 :long)
  (size3 :long))
(cffi:defcfun ("THDoubleTensor_resize5d" th-double-tensor-resize-5d) :void
  (tensor th-double-tensor-ptr)
  (size0 :long)
  (size1 :long)
  (size2 :long)
  (size3 :long)
  (size4 :long))

(cffi:defcfun ("THDoubleTensor_set" th-double-tensor-set) :void
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr))
(cffi:defcfun ("THDoubleTensor_setStorage" th-double-tensor-set-storage) :void
  (tensor th-double-tensor-ptr)
  (storage th-double-storage-ptr)
  (storage-offset :long-long)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
(cffi:defcfun ("THDoubleTensor_setStorageNd" th-double-tensor-set-storage-nd) :void
  (tensor th-double-tensor-ptr)
  (storage th-double-storage-ptr)
  (storage-offset :long-long)
  (dimension :int)
  (size th-long-storage-ptr)
  (stride th-long-storage-ptr))
(cffi:defcfun ("THDoubleTensor_setStorage1d" th-double-tensor-set-storage-1d) :void
  (tensor th-double-tensor-ptr)
  (storage th-double-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long))
(cffi:defcfun ("THDoubleTensor_setStorage2d" th-double-tensor-set-storage-2d) :void
  (tensor th-double-tensor-ptr)
  (storage th-double-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long))
(cffi:defcfun ("THDoubleTensor_setStorage3d" th-double-tensor-set-storage-3d) :void
  (tensor th-double-tensor-ptr)
  (storage th-double-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long)
  (size2 :long)
  (stride2 :long))
(cffi:defcfun ("THDoubleTensor_setStorage4d" th-double-tensor-set-storage-4d) :void
  (tensor th-double-tensor-ptr)
  (storage th-double-storage-ptr)
  (storage-offset :long-long)
  (size0 :long)
  (stride0 :long)
  (size1 :long)
  (stride1 :long)
  (size2 :long)
  (stride2 :long)
  (size3 :long)
  (stride3 :long))

(cffi:defcfun ("THDoubleTensor_narrow" th-double-tensor-narrow) :void
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr)
  (dimension :int)
  (first-index :long)
  (size :long))
(cffi:defcfun ("THDoubleTensor_select" th-double-tensor-select) :void
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr)
  (dimension :int)
  (slice-index :long))
(cffi:defcfun ("THDoubleTensor_transpose" th-double-tensor-transpose) :void
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr)
  (dimension1 :int)
  (dimension2 :int))
(cffi:defcfun ("THDoubleTensor_unfold" th-double-tensor-unfold) :void
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr)
  (dimension :int)
  (size :long)
  (step :long))
(cffi:defcfun ("THDoubleTensor_squeeze" th-double-tensor-squeeze) :void
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr))
(cffi:defcfun ("THDoubleTensor_squeeze1d" th-double-tensor-squeeze-1d) :void
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr)
  (dimension :int))
(cffi:defcfun ("THDoubleTensor_unsqueeze1d" th-double-tensor-unsqueeze-1d) :void
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr)
  (dimension :int))

(cffi:defcfun ("THDoubleTensor_isContiguous" th-double-tensor-is-contiguous) :int
  (tensor th-double-tensor-ptr))
(cffi:defcfun ("THDoubleTensor_isSameSizeAs" th-double-tensor-is-same-size-as) :int
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr))
(cffi:defcfun ("THDoubleTensor_isSetTo" th-double-tensor-is-set-to) :int
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr))
(cffi:defcfun ("THDoubleTensor_isSize" th-double-tensor-is-size) :int
  (tensor th-double-tensor-ptr)
  (dims th-long-storage-ptr))
(cffi:defcfun ("THDoubleTensor_nElement" th-double-tensor-n-element) :long-long
  (tensor th-double-tensor-ptr))

(cffi:defcfun ("THDoubleTensor_retain" th-double-tensor-retain) :void
  (tensor th-double-tensor-ptr))
(cffi:defcfun ("THDoubleTensor_free" th-double-tensor-free) :void
  (tensor th-double-tensor-ptr))
(cffi:defcfun ("THDoubleTensor_freeCopyTo" th-double-tensor-free-copy-to) :void
  (source th-double-tensor-ptr)
  (target th-double-tensor-ptr))

;; slow access methods [check everything]
;; void THTensor_(set1d)(THTensor *tensor, long x0, real value);
(cffi:defcfun ("THDoubleTensor_set1d" th-double-tensor-set-1d) :void
  (tensor th-double-tensor-ptr)
  (index0 :long)
  (value :double))
;; void THTensor_(set2d)(THTensor *tensor, long x0, long x1, real value);
(cffi:defcfun ("THDoubleTensor_set2d" th-double-tensor-set-2d) :void
  (tensor th-double-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (value :double))
;; void THTensor_(set3d)(THTensor *tensor, long x0, long x1, long x2, real value);
(cffi:defcfun ("THDoubleTensor_set3d" th-double-tensor-set-3d) :void
  (tensor th-double-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long)
  (value :double))
;; void THTensor_(set4d)(THTensor *tensor, long x0, long x1, long x2, long x3, real value);
(cffi:defcfun ("THDoubleTensor_set4d" th-double-tensor-set-4d) :void
  (tensor th-double-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long)
  (index3 :long)
  (value :double))

(cffi:defcfun ("THDoubleTensor_get1d" th-double-tensor-get-1d) :double
  (tensor th-double-tensor-ptr)
  (index0 :long))
(cffi:defcfun ("THDoubleTensor_get2d" th-double-tensor-get-2d) :double
  (tensor th-double-tensor-ptr)
  (index0 :long)
  (index1 :long))
(cffi:defcfun ("THDoubleTensor_get3d" th-double-tensor-get-3d) :double
  (tensor th-double-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long))
(cffi:defcfun ("THDoubleTensor_get4d" th-double-tensor-get-4d) :double
  (tensor th-double-tensor-ptr)
  (index0 :long)
  (index1 :long)
  (index2 :long)
  (index3 :long))

;; support for copy betweeb different tensor types
;; void THTensor_(copy)(THTensor *tensor, THTensor *src);
(cffi:defcfun ("THDoubleTensor_copy" th-double-tensor-copy) :void
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr))
;; void THTensor_(copyByte)(THTensor *tensor, struct THByteTensor *src);
(cffi:defcfun ("THDoubleTensor_copyByte" th-double-tensor-copy-byte) :void
  (tensor th-double-tensor-ptr)
  (src th-byte-tensor-ptr))
;; void THTensor_(copyChar)(THTensor *tensor, struct THCharTensor *src);
(cffi:defcfun ("THDoubleTensor_copyChar" th-double-tensor-copy-char) :void
  (tensor th-double-tensor-ptr)
  (src th-char-tensor-ptr))
;; void THTensor_(copyShort)(THTensor *tensor, struct THShortTensor *src);
(cffi:defcfun ("THDoubleTensor_copyShort" th-double-tensor-copy-short) :void
  (tensor th-double-tensor-ptr)
  (src th-short-tensor-ptr))
;; void THTensor_(copyInt)(THTensor *tensor, struct THIntTensor *src);
(cffi:defcfun ("THDoubleTensor_copyInt" th-double-tensor-copy-int) :void
  (tensor th-double-tensor-ptr)
  (src th-int-tensor-ptr))
;; void THTensor_(copyLong)(THTensor *tensor, struct THLongTensor *src);
(cffi:defcfun ("THDoubleTensor_copyLong" th-double-tensor-copy-long) :void
  (tensor th-double-tensor-ptr)
  (src th-long-tensor-ptr))
;; void THTensor_(copyFloat)(THTensor *tensor, struct THFloatTensor *src);
(cffi:defcfun ("THDoubleTensor_copyFloat" th-double-tensor-copy-float) :void
  (tensor th-double-tensor-ptr)
  (src th-float-tensor-ptr))
;; void THTensor_(copyDouble)(THTensor *tensor, struct THDoubleTensor *src);
(cffi:defcfun ("THDoubleTensor_copyDouble" th-double-tensor-copy-double) :void
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr))

;; RANDOM
;; void THTensor_(random)(THTensor *self, THGenerator *_generator);
(cffi:defcfun ("THDoubleTensor_random" th-double-tensor-random) :void
  (tensor th-double-tensor-ptr)
  (generator th-generator-ptr))
;; void THTensor_(clampedRandom)(THTensor *self, THGenerator *_generator, long min, long max)
(cffi:defcfun ("THDoubleTensor_clampedRandom" th-double-tensor-clamped-random) :void
  (tensor th-double-tensor-ptr)
  (genrator th-generator-ptr)
  (min :long)
  (max :long))
;; void THTensor_(cappedRandom)(THTensor *self, THGenerator *_generator, long max);
(cffi:defcfun ("THDoubleTensor_cappedRandom" th-double-tensor-capped-random) :void
  (tensor th-double-tensor-ptr)
  (generator th-generator-ptr)
  (max :long))
;; void THTensor_(geometric)(THTensor *self, THGenerator *_generator, double p);
(cffi:defcfun ("THDoubleTensor_geometric" th-double-tensor-geometric) :void
  (tensor th-double-tensor-ptr)
  (generator th-generator-ptr)
  (p :double))
;; void THTensor_(bernoulli)(THTensor *self, THGenerator *_generator, double p);
(cffi:defcfun ("THDoubleTensor_bernoulli" th-double-tensor-bernoulli) :void
  (tensor th-double-tensor-ptr)
  (generator th-generator-ptr)
  (p :double))
;; void THTensor_(bernoulli_FloatTensor)(THTensor *self, THGenerator *_generator, THFloatTensor *p);
(cffi:defcfun ("THDoubleTensor_bernoulli_FloatTensor" th-double-tensor-bernoulli-float-tensor) :void
  (tensor th-double-tensor-ptr)
  (generator th-generator-ptr)
  (p th-float-tensor-ptr))
;; void THTensor_(bernoulli_DoubleTensor)(THTensor *self, THGenerator *_generator, THDoubleTensor *p);
(cffi:defcfun ("THDoubleTensor_bernoulli_DoubleTensor" th-double-tensor-bernoulli-double-tensor)
    :void
  (tensor th-double-tensor-ptr)
  (generator th-generator-ptr)
  (p th-double-tensor-ptr))

;; #if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
;; void THTensor_(uniform)(THTensor *self, THGenerator *_generator, double a, double b);
(cffi:defcfun ("THDoubleTensor_uniform" th-double-tensor-uniform) :void
  (tensor th-double-tensor-ptr)
  (generator th-generator-ptr)
  (a :double)
  (b :double))
;; void THTensor_(normal)(THTensor *self, THGenerator *_generator, double mean, double stdv);
(cffi:defcfun ("THDoubleTensor_normal" th-double-tensor-normal) :void
  (tensor th-double-tensor-ptr)
  (generator th-generator-ptr)
  (mean :double)
  (stdv :double))
;; void THTensor_(normal_means)(THTensor *self, THGenerator *gen, THTensor *means, double stddev);
(cffi:defcfun ("THDoubleTensor_normal_means" th-double-tensor-normal-means) :void
  (tensor th-double-tensor-ptr)
  (generator th-generator-ptr)
  (means th-double-tensor-ptr)
  (stdv :double))
;; void THTensor_(normal_stddevs)(THTensor *self, THGenerator *gen, double mean, THTensor *stddevs);
(cffi:defcfun ("THDoubleTensor_normal_stddevs" th-double-tensor-normal-stddevs) :void
  (tensor th-double-tensor-ptr)
  (generator th-generator-ptr)
  (mean :double)
  (stddevs th-double-tensor-ptr))
;; void THTensor_(normal_means_stddevs)(THTensor *self, THGenerator *gen, THTensor *means, THTensor *stddevs);
(cffi:defcfun ("THDoubleTensor_normal_means_stddevs" th-double-tensor-normal-means-stddevs) :void
  (tensor th-double-tensor-ptr)
  (generator th-generator-ptr)
  (means th-double-tensor-ptr)
  (stddevs th-double-tensor-ptr))
;; void THTensor_(exponential)(THTensor *self, THGenerator *_generator, double lambda);
(cffi:defcfun ("THDoubleTensor_exponential" th-double-tensor-exponential) :void
  (tensor th-double-tensor-ptr)
  (generator th-generator-ptr)
  (lam :double))
;; void THTensor_(cauchy)(THTensor *self, THGenerator *_generator, double median, double sigma);
(cffi:defcfun ("THDoubleTensor_cauchy" th-double-tensor-cauchy) :void
  (tensor th-double-tensor-ptr)
  (generator th-generator-ptr)
  (median :double)
  (sigma :double))
;; void THTensor_(logNormal)(THTensor *self, THGenerator *_generator, double mean, double stdv);
(cffi:defcfun ("THDoubleTensor_logNormal" th-double-tensor-log-normal) :void
  (tensor th-double-tensor-ptr)
  (generator th-generator-ptr)
  (mean :double)
  (stdv :double))
;; void THTensor_(multinomial)(THLongTensor *self, THGenerator *_generator, THTensor *prob_dist, int n_sample, int with_replacement);
(cffi:defcfun ("THDoubleTensor_multinomial" th-double-tensor-multinomial) :void
  (tensor th-long-tensor-ptr)
  (generator th-generator-ptr)
  (prob-dist th-double-tensor-ptr)
  (n-sample :int)
  (replacement :int))
;; void THTensor_(multinomialAliasSetup)(THTensor *prob_dist, THLongTensor *J, THTensor *q);
(cffi:defcfun ("THDoubleTensor_multinomialAliasSetup" th-double-tensor-multinomial-alias-setup)
    :void
  (prob-dist th-double-tensor-ptr)
  (j th-long-tensor-ptr)
  (q th-double-tensor-ptr))
;; void THTensor_(multinomialAliasDraw)(THLongTensor *self, THGenerator *_generator, THLongTensor *J, THTensor *q);
(cffi:defcfun ("THDoubleTensor_multinomialAliasDraw" th-double-tensor-multinomial-alias-draw)
    :void
  (tensor th-double-tensor-ptr)
  (generator th-generator-ptr)
  (j th-long-tensor-ptr)
  (q th-double-tensor-ptr))
;; #endif

;; #if defined(TH_REAL_IS_BYTE)
;; void THTensor_(getRNGState)(THGenerator *_generator, THTensor *self);
;; void THTensor_(setRNGState)(THGenerator *_generator, THTensor *self);
;; #endif

;; void THTensor_(fill)(THTensor *r_, real value);
(cffi:defcfun ("THDoubleTensor_fill" th-double-tensor-fill) :void
  (tensor th-double-tensor-ptr)
  (value :double))
;; void THTensor_(zero)(THTensor *r_);
(cffi:defcfun ("THDoubleTensor_zero" th-double-tensor-zero) :void
  (tensor th-double-tensor-ptr))

;; void THTensor_(maskedFill)(THTensor *tensor, THByteTensor *mask, real value);
(cffi:defcfun ("THDoubleTensor_maskedFill" th-double-tensor-masked-fill) :void
  (tensor th-double-tensor-ptr)
  (mask th-byte-tensor-ptr)
  (value :double))
;; void THTensor_(maskedCopy)(THTensor *tensor, THByteTensor *mask, THTensor* src);
(cffi:defcfun ("THDoubleTensor_maskedCopy" th-double-tensor-masked-copy) :void
  (tensor th-double-tensor-ptr)
  (mask th-byte-tensor-ptr)
  (src th-double-tensor-ptr))
;; void THTensor_(maskedSelect)(THTensor *tensor, THTensor* src, THByteTensor *mask);
(cffi:defcfun ("THDoubleTensor_maskedSelect" th-double-tensor-masked-select) :void
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr)
  (mask (th-byte-tensor-ptr)))

;; void THTensor_(nonzero)(THLongTensor *subscript, THTensor *tensor);
(cffi:defcfun ("THDoubleTensor_nonzero" th-double-tensor-nonzero) :void
  (subscript th-long-tensor-ptr)
  (tensor th-double-tensor-ptr))

;; void THTensor_(indexSelect)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index)
(cffi:defcfun ("THDoubleTensor_indexSelect" th-double-tensor-index-select) :void
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr))
;; void THTensor_(indexCopy)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THDoubleTensor_indexCopy" th-double-tensor-index-copy) :void
  (tensor th-double-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-double-tensor-ptr))
;; void THTensor_(indexAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THDoubleTensor_indexAdd" th-double-tensor-index-add) :void
  (tensor th-double-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-double-tensor-ptr))
;; void THTensor_(indexFill)(THTensor *tensor, int dim, THLongTensor *index, real val);
(cffi:defcfun ("THDoubleTensor_indexFill" th-double-tensor-index-fill) :void
  (tensor th-double-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (value :double))

;; void THTensor_(gather)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index);
(cffi:defcfun ("THDoubleTensor_gather" th-double-tensor-gather) :void
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr))
;; void THTensor_(scatter)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THDoubleTensor_scatter" th-double-tensor-scatter) :void
  (tensor th-double-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-double-tensor-ptr))
;; void THTensor_(scatterAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
(cffi:defcfun ("THDoubleTensor_scatterAdd" th-double-tensor-scatter-add) :void
  (tensor th-double-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (src th-double-tensor-ptr))
;; void THTensor_(scatterFill)(THTensor *tensor, int dim, THLongTensor *index, real val);
(cffi:defcfun ("THDoubleTensor_scatterFill" th-double-tensor-scatter-fill) :void
  (tensor th-double-tensor-ptr)
  (dim :int)
  (index th-long-tensor-ptr)
  (value :double))

;; accreal THTensor_(dot)(THTensor *t, THTensor *src);
(cffi:defcfun ("THDoubleTensor_dot" th-double-tensor-dot) :double
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr))

;; real THTensor_(minall)(THTensor *t);
(cffi:defcfun ("THDoubleTensor_minall" th-double-tensor-min-all) :double
  (tensor th-double-tensor-ptr))
;; real THTensor_(maxall)(THTensor *t);
(cffi:defcfun ("THDoubleTensor_maxall" th-double-tensor-max-all) :double
  (tensor th-double-tensor-ptr))
;; real THTensor_(medianall)(THTensor *t);
(cffi:defcfun ("THDoubleTensor_medianall" th-double-tensor-median-all) :double
  (tensor th-double-tensor-ptr))
;; accreal THTensor_(sumall)(THTensor *t);
(cffi:defcfun ("THDoubleTensor_sumall" th-double-tensor-sum-all) :double
  (tensor th-double-tensor-ptr))
;; accreal THTensor_(prodall)(THTensor *t);
(cffi:defcfun ("THDoubleTensor_prodall" th-double-tensor-prod-all) :double
  (tensor th-double-tensor-ptr))

;; void THTensor_(neg)(THTensor *self, THTensor *src);
(cffi:defcfun ("THDoubleTensor_neg" th-double-tensor-neg) :void
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr))
;; void THTensor_(cinv)(THTensor *self, THTensor *src);
(cffi:defcfun ("THDoubleTensor_cinv" th-double-tensor-cinv) :void
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr))

;; void THTensor_(add)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THDoubleTensor_add" th-double-tensor-add) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))
;; void THTensor_(sub)(THTensor *self, THTensor *src, real value);
(cffi:defcfun ("THDoubleTensor_sub" th-double-tensor-sub) :void
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr)
  (value :double))
;; void THTensor_(mul)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THDoubleTensor_mul" th-double-tensor-mul) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))
;; void THTensor_(div)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THDoubleTensor_div" th-double-tensor-div) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))
;; void THTensor_(lshift)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THDoubleTensor_lshift" th-double-tensor-lshift) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))
;; void THTensor_(rshift)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THDoubleTensor_rshift" th-double-tensor-rshift) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))
;; void THTensor_(fmod)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THDoubleTensor_fmod" th-double-tensor-fmod) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))
;; void THTensor_(remainder)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THDoubleTensor_remainder" th-double-tensor-remainder) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))
;; void THTensor_(clamp)(THTensor *r_, THTensor *t, real min_value, real max_value);
(cffi:defcfun ("THDoubleTensor_clamp" th-double-tensor-clamp) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (min-value :double)
  (max-value :double))
;; void THTensor_(bitand)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THDoubleTensor_bitand" th-double-tensor-bitand) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))
;; void THTensor_(bitor)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THDoubleTensor_bitor" th-double-tensor-bitor) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))
;; void THTensor_(bitxor)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THDoubleTensor_bitxor" th-double-tensor-bitxor) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))

;; void THTensor_(cadd)(THTensor *r_, THTensor *t, real value, THTensor *src);
(cffi:defcfun ("THDoubleTensor_cadd" th-double-tensor-cadd) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double)
  (src th-double-tensor-ptr))
;; void THTensor_(csub)(THTensor *self, THTensor *src1, real value, THTensor *src2);
(cffi:defcfun ("THDoubleTensor_csub" th-double-tensor-csub) :void
  (tensor th-double-tensor-ptr)
  (src1 th-double-tensor-ptr)
  (value :double)
  (src2 th-double-tensor-ptr))
;; void THTensor_(cmul)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THDoubleTensor_cmul" th-double-tensor-cmul) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr))
;; void THTensor_(cpow)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THDoubleTensor_cpow" th-double-tensor-cpow) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr))
;; void THTensor_(cdiv)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THDoubleTensor_cdiv" th-double-tensor-cdiv) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr))
;; void THTensor_(clshift)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THDoubleTensor_clshift" th-double-tensor-clshift) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr))
;; void THTensor_(crshift)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THDoubleTensor_crshift" th-double-tensor-crshift) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr))
;; void THTensor_(cfmod)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THDoubleTensor_cfmod" th-double-tensor-cfmod) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr))
;; void THTensor_(cremainder)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THDoubleTensor_cremainder" th-double-tensor-cremainder) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr))
;; void THTensor_(cbitand)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THDoubleTensor_cbitand" th-double-tensor-cbitand) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr))
;; void THTensor_(cbitor)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THDoubleTensor_cbitor" th-double-tensor-cbitor) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr))
;; void THTensor_(cbitxor)(THTensor *r_, THTensor *t, THTensor *src);
(cffi:defcfun ("THDoubleTensor_cbitxor" th-double-tensor-cbitxor) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr))

;; void THTensor_(addcmul)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2);
(cffi:defcfun ("THDoubleTensor_addcmul" th-double-tensor-add-cmul) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double)
  (src1 th-double-tensor-ptr)
  (src2 th-double-tensor-ptr))
;; void THTensor_(addcdiv)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2);
(cffi:defcfun ("THDoubleTensor_addcdiv" th-double-tensor-add-cdiv) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double)
  (src1 th-double-tensor-ptr)
  (src2 th-double-tensor-ptr))
;; void THTensor_(addmv)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat,  THTensor *vec);
(cffi:defcfun ("THDoubleTensor_addmv" th-double-tensor-add-mv) :void
  (result th-double-tensor-ptr)
  (beta :double)
  (tensor th-double-tensor-ptr)
  (alpha :double)
  (matrix th-double-tensor-ptr)
  (vector th-double-tensor-ptr))
;; void THTensor_(addmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat1, THTensor *mat2);
(cffi:defcfun ("THDoubleTensor_addmm" th-double-tensor-add-mm) :void
  (result th-double-tensor-ptr)
  (beta :double)
  (tensor th-double-tensor-ptr)
  (alpha :double)
  (matrix1 th-double-tensor-ptr)
  (matrix2 th-double-tensor-ptr))
;; void THTensor_(addr)(THTensor *r_,  real beta, THTensor *t, real alpha, THTensor *vec1, THTensor *vec2);
(cffi:defcfun ("THDoubleTensor_addr" th-double-tensor-add-r) :void
  (result th-double-tensor-ptr)
  (beta :double)
  (tensor th-double-tensor-ptr)
  (alpha :double)
  (vector1 th-double-tensor-ptr)
  (vector2 th-double-tensor-ptr))
;; void THTensor_(addbmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *batch1, THTensor *batch2);
(cffi:defcfun ("THDoubleTensor_addbmm" th-double-tensor-add-bmm) :void
  (result th-double-tensor-ptr)
  (beta :double)
  (tensor th-double-tensor-ptr)
  (alpha :double)
  (batch1 th-double-tensor-ptr)
  (batch2 th-double-tensor-ptr))
;; void THTensor_(baddbmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *batch1, THTensor *batch2);
(cffi:defcfun ("THDoubleTensor_baddbmm" th-double-tensor-badd-bmm) :void
  (result th-double-tensor-ptr)
  (beta :double)
  (tensor th-double-tensor-ptr)
  (alpha :double)
  (batch1 th-double-tensor-ptr)
  (batch2 th-double-tensor-ptr))

;; void THTensor_(match)(THTensor *r_, THTensor *m1, THTensor *m2, real gain);
(cffi:defcfun ("THDoubleTensor_match" th-double-tensor-match) :void
  (result th-double-tensor-ptr)
  (m1 th-double-tensor-ptr)
  (m2 th-double-tensor-ptr)
  (gain :double))

;; ptrdiff_t THTensor_(numel)(THTensor *t);
(cffi:defcfun ("THDoubleTensor_numel" th-double-tensor-numel) :long-long
  (tensor th-double-tensor-ptr))
;; void THTensor_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THDoubleTensor_max" th-double-tensor-max) :void
  (values th-double-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THDoubleTensor_min" th-double-tensor-min) :void
  (values th-double-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(kthvalue)(THTensor *values_, THLongTensor *indices_, THTensor *t, long k, int dimension, int keepdim);
(cffi:defcfun ("THDoubleTensor_kthvalue" th-double-tensor-kth-value) :void
  (values th-double-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (k :long)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(mode)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THDoubleTensor_mode" th-double-tensor-mode) :void
  (values th-double-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(median)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THDoubleTensor_median" th-double-tensor-median) :void
  (values th-double-tensor-ptr)
  (indices th-long-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(sum)(THTensor *r_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THDoubleTensor_sum" th-double-tensor-sum) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(prod)(THTensor *r_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THDoubleTensor_prod" th-double-tensor-prod) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(cumsum)(THTensor *r_, THTensor *t, int dimension);
(cffi:defcfun ("THDoubleTensor_cumsum" th-double-tensor-cum-sum) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (dimension :int))
;; void THTensor_(cumprod)(THTensor *r_, THTensor *t, int dimension);
(cffi:defcfun ("THDoubleTensor_cumprod" th-double-tensor-cum-prod) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (dimension :int))
;; void THTensor_(sign)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THDoubleTensor_sign" th-double-tensor-sign) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr))
;; accreal THTensor_(trace)(THTensor *t);
(cffi:defcfun ("THDoubleTensor_trace" th-double-tensor-trace) :double
  (tensor th-double-tensor-ptr))
;; void THTensor_(cross)(THTensor *r_, THTensor *a, THTensor *b, int dimension);
(cffi:defcfun ("THDoubleTensor_cross" th-double-tensor-cross) :void
  (result th-double-tensor-ptr)
  (a th-double-tensor-ptr)
  (b th-double-tensor-ptr)
  (dimension :int))

;; void THTensor_(cmax)(THTensor *r, THTensor *t, THTensor *src);
(cffi:defcfun ("THDoubleTensor_cmax" th-double-tensor-cmax) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr))
;; void THTensor_(cmin)(THTensor *r, THTensor *t, THTensor *src);
(cffi:defcfun ("THDoubleTensor_cmin" th-double-tensor-cmin) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (src th-double-tensor-ptr))
;; void THTensor_(cmaxValue)(THTensor *r, THTensor *t, real value);
(cffi:defcfun ("THDoubleTensor_cmaxValue" th-double-tensor-cmax-value) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))
;; void THTensor_(cminValue)(THTensor *r, THTensor *t, real value);
(cffi:defcfun ("THDoubleTensor_cminValue" th-double-tensor-cmin-value) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))

;; void THTensor_(zeros)(THTensor *r_, THLongStorage *size);
(cffi:defcfun ("THDoubleTensor_zeros" th-double-tensor-zeros) :void
  (result th-double-tensor-ptr)
  (size th-long-storage-ptr))
;; void THTensor_(zerosLike)(THTensor *r_, THTensor *input);
(cffi:defcfun ("THDoubleTensor_zerosLike" th-double-tensor-zero-like) :void
  (result th-double-tensor-ptr)
  (input th-double-tensor-ptr))
;; void THTensor_(ones)(THTensor *r_, THLongStorage *size);
(cffi:defcfun ("THDoubleTensor_ones" th-double-tensor-ones) :void
  (result th-double-tensor-ptr)
  (size th-long-storage-ptr))
;; void THTensor_(onesLike)(THTensor *r_, THTensor *input);
(cffi:defcfun ("THDoubleTensor_onesLike" th-double-tensor-one-like) :void
  (result th-double-tensor-ptr)
  (input th-double-tensor-ptr))
;; void THTensor_(diag)(THTensor *r_, THTensor *t, int k);
(cffi:defcfun ("THDoubleTensor_diag" th-double-tensor-diag) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (k :int))
;; void THTensor_(eye)(THTensor *r_, long n, long m);
(cffi:defcfun ("THDoubleTensor_eye" th-double-tensor-eye) :void
  (result th-double-tensor-ptr)
  (n :long)
  (m :long))
;; void THTensor_(arange)(THTensor *r_, accreal xmin, accreal xmax, accreal step);
(cffi:defcfun ("THDoubleTensor_arange" th-double-tensor-arange) :void
  (result th-double-tensor-ptr)
  (xmin :double)
  (xmax :double)
  (step :double))
;; void THTensor_(range)(THTensor *r_, accreal xmin, accreal xmax, accreal step);
(cffi:defcfun ("THDoubleTensor_range" th-double-tensor-range) :void
  (result th-double-tensor-ptr)
  (xmin :double)
  (xmax :double)
  (step :double))
;; void THTensor_(randperm)(THTensor *r_, THGenerator *_generator, long n);
(cffi:defcfun ("THDoubleTensor_randperm" th-double-tensor-rand-perm) :void
  (result th-double-tensor-ptr)
  (generator th-generator-ptr)
  (n :long))

;; void THTensor_(reshape)(THTensor *r_, THTensor *t, THLongStorage *size);
(cffi:defcfun ("THDoubleTensor_reshape" th-double-tensor-reshape) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (size th-long-storage-ptr))
;; void THTensor_(sort)(THTensor *rt_, THLongTensor *ri_, THTensor *t, int dimension, int descendingOrder);
(cffi:defcfun ("THDoubleTensor_sort" th-double-tensor-sort) :void
  (result-tensor th-double-tensor-ptr)
  (result-indices th-long-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (dimension :int)
  (discending-order :int))
;; void THTensor_(topk)(THTensor *rt_, THLongTensor *ri_, THTensor *t, long k, int dim, int dir, int sorted);
(cffi:defcfun ("THDoubleTensor_topk" th-double-tensor-topk) :void
  (result-tensor th-double-tensor-ptr)
  (result-indices th-long-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (k :long)
  (dim :int)
  (dir :int)
  (sorted :int))
;; void THTensor_(tril)(THTensor *r_, THTensor *t, long k);
(cffi:defcfun ("THDoubleTensor_tril" th-double-tensor-tril) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (k :long))
;; void THTensor_(triu)(THTensor *r_, THTensor *t, long k);
(cffi:defcfun ("THDoubleTensor_triu" th-double-tensor-triu) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (k :long))
;; void THTensor_(cat)(THTensor *r_, THTensor *ta, THTensor *tb, int dimension);
(cffi:defcfun ("THDoubleTensor_cat" th-double-tensor-cat) :void
  (result th-double-tensor-ptr)
  (tensora th-double-tensor-ptr)
  (tensorb th-double-tensor-ptr)
  (dimension :int))
;; void THTensor_(catArray)(THTensor *result, THTensor **inputs, int numInputs, int dimension);
(cffi:defcfun ("THDoubleTensor_catArray" th-double-tensor-cat-array) :void
  (result th-double-tensor-ptr)
  (inputs (:pointer th-double-tensor-ptr))
  (num-inputs :int)
  (dimension :int))

;; int THTensor_(equal)(THTensor *ta, THTensor *tb);
(cffi:defcfun ("THDoubleTensor_equal" th-double-tensor-equal) :int
  (tensora th-double-tensor-ptr)
  (tensorb th-double-tensor-ptr))

;; void THTensor_(ltValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THDoubleTensor_ltValue" th-double-tensor-lt-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))
;; void THTensor_(leValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THDoubleTensor_leValue" th-double-tensor-le-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))
;; void THTensor_(gtValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THDoubleTensor_gtValue" th-double-tensor-gt-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))
;; void THTensor_(geValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THDoubleTensor_geValue" th-double-tensor-ge-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))
;; void THTensor_(neValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THDoubleTensor_neValue" th-double-tensor-ne-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))
;; void THTensor_(eqValue)(THByteTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THDoubleTensor_eqValue" th-double-tensor-eq-value) :void
  (result th-byte-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))

;; void THTensor_(ltValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THDoubleTensor_ltValueT" th-double-tensor-lt-value-t) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))
;; void THTensor_(leValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THDoubleTensor_leValueT" th-double-tensor-le-value-t) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))
;; void THTensor_(gtValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THDoubleTensor_gtValueT" th-double-tensor-gt-value-t) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))
;; void THTensor_(geValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THDoubleTensor_geValueT" th-double-tensor-ge-value-t) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))
;; void THTensor_(neValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THDoubleTensor_neValueT" th-double-tensor-ne-value-t) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))
;; void THTensor_(eqValueT)(THTensor *r_, THTensor* t, real value);
(cffi:defcfun ("THDoubleTensor_eqValueT" th-double-tensor-eq-value-t) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))

;; void THTensor_(ltTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THDoubleTensor_ltTensor" th-double-tensor-lt-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-double-tensor-ptr)
  (tensorb th-double-tensor-ptr))
;; void THTensor_(leTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THDoubleTensor_leTensor" th-double-tensor-le-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-double-tensor-ptr)
  (tensorb th-double-tensor-ptr))
;; void THTensor_(gtTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THDoubleTensor_gtTensor" th-double-tensor-gt-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-double-tensor-ptr)
  (tensorb th-double-tensor-ptr))
;; void THTensor_(geTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THDoubleTensor_geTensor" th-double-tensor-ge-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-double-tensor-ptr)
  (tensorb th-double-tensor-ptr))
;; void THTensor_(neTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THDoubleTensor_neTensor" th-double-tensor-ne-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-double-tensor-ptr)
  (tensorb th-double-tensor-ptr))
;; void THTensor_(eqTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THDoubleTensor_eqTensor" th-double-tensor-eq-tensor) :void
  (result th-byte-tensor-ptr)
  (tensora th-double-tensor-ptr)
  (tensorb th-double-tensor-ptr))

;; void THTensor_(ltTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THDoubleTensor_ltTensorT" th-double-tensor-lt-tensor-t) :void
  (result th-double-tensor-ptr)
  (tensora th-double-tensor-ptr)
  (tensorb th-double-tensor-ptr))
;; void THTensor_(leTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THDoubleTensor_leTensorT" th-double-tensor-le-tensor-t) :void
  (result th-double-tensor-ptr)
  (tensora th-double-tensor-ptr)
  (tensorb th-double-tensor-ptr))
;; void THTensor_(gtTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THDoubleTensor_gtTensorT" th-double-tensor-gt-tensor-t) :void
  (result th-double-tensor-ptr)
  (tensora th-double-tensor-ptr)
  (tensorb th-double-tensor-ptr))
;; void THTensor_(geTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THDoubleTensor_geTensorT" th-double-tensor-ge-tensor-t) :void
  (result th-double-tensor-ptr)
  (tensora th-double-tensor-ptr)
  (tensorb th-double-tensor-ptr))
;; void THTensor_(neTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THDoubleTensor_neTensorT" th-double-tensor-ne-tensor-t) :void
  (result th-double-tensor-ptr)
  (tensora th-double-tensor-ptr)
  (tensorb th-double-tensor-ptr))
;; void THTensor_(eqTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
(cffi:defcfun ("THDoubleTensor_eqTensorT" th-double-tensor-eq-tensor-t) :void
  (result th-double-tensor-ptr)
  (tensora th-double-tensor-ptr)
  (tensorb th-double-tensor-ptr))

;; #if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

;; void THTensor_(sigmoid)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THDoubleTensor_sigmoid" th-double-tensor-sigmoid) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr))
;; void THTensor_(log)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THDoubleTensor_log" th-double-tensor-log) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr))
;; void THTensor_(lgamma)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THDoubleTensor_lgamma" th-double-tensor-lgamma) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr))
;; void THTensor_(log1p)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THDoubleTensor_log1p" th-double-tensor-log1p) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr))
;; void THTensor_(exp)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THDoubleTensor_exp" th-double-tensor-exp) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr))
;; void THTensor_(cos)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THDoubleTensor_cos" th-double-tensor-cos) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr))
;; void THTensor_(acos)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THDoubleTensor_acos" th-double-tensor-acos) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr))
;; void THTensor_(cosh)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THDoubleTensor_cosh" th-double-tensor-cosh) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr))
;; void THTensor_(sin)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THDoubleTensor_sin" th-double-tensor-sin) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr))
;; void THTensor_(asin)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THDoubleTensor_asin" th-double-tensor-asin) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr))
;; void THTensor_(sinh)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THDoubleTensor_sinh" th-double-tensor-sinh) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr))
;; void THTensor_(tan)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THDoubleTensor_tan" th-double-tensor-tan) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr))
;; void THTensor_(atan)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THDoubleTensor_atan" th-double-tensor-atan) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr))
;; void THTensor_(atan2)(THTensor *r_, THTensor *tx, THTensor *ty);
(cffi:defcfun ("THDoubleTensor_atan2" th-double-tensor-atan2) :void
  (result th-double-tensor-ptr)
  (tensorx th-double-tensor-ptr)
  (tensory th-double-tensor-ptr))
;; void THTensor_(tanh)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THDoubleTensor_tanh" th-double-tensor-tanh) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr))
;; void THTensor_(pow)(THTensor *r_, THTensor *t, real value);
(cffi:defcfun ("THDoubleTensor_pow" th-double-tensor-pow) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double))
;; void THTensor_(tpow)(THTensor *r_, real value, THTensor *t);
(cffi:defcfun ("THDoubleTensor_tpow" th-double-tensor-tpow) :void
  (result th-double-tensor-ptr)
  (value :double)
  (tensor th-double-tensor-ptr))
;; void THTensor_(sqrt)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THDoubleTensor_sqrt" th-double-tensor-sqrt) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr))
;; void THTensor_(rsqrt)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THDoubleTensor_rsqrt" th-double-tensor-rsqrt) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr))
;; void THTensor_(ceil)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THDoubleTensor_ceil" th-double-tensor-ceil) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr))
;; void THTensor_(floor)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THDoubleTensor_floor" th-double-tensor-floor) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr))
;; void THTensor_(round)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THDoubleTensor_round" th-double-tensor-round) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr))
;; void THTensor_(abs)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THDoubleTensor_abs" th-double-tensor-abs) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr))
;; void THTensor_(trunc)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THDoubleTensor_trunc" th-double-tensor-trunc) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr))
;; void THTensor_(frac)(THTensor *r_, THTensor *t);
(cffi:defcfun ("THDoubleTensor_frac" th-double-tensor-frac) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr))
;; void THTensor_(lerp)(THTensor *r_, THTensor *a, THTensor *b, real weight);
(cffi:defcfun ("THDoubleTensor_lerp" th-double-tensor-lerp) :void
  (result th-double-tensor-ptr)
  (a th-double-tensor-ptr)
  (b th-double-tensor-ptr)
  (weight :double))

;; void THTensor_(mean)(THTensor *r_, THTensor *t, int dimension, int keepdim);
(cffi:defcfun ("THDoubleTensor_mean" th-double-tensor-mean) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(std)(THTensor *r_, THTensor *t, int dimension, int biased, int keepdim);
(cffi:defcfun ("THDoubleTensor_std" th-double-tensor-std) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (dimension :int)
  (biased :int)
  (keep-dim :int))
;; void THTensor_(var)(THTensor *r_, THTensor *t, int dimension, int biased, int keepdim);
(cffi:defcfun ("THDoubleTensor_var" th-double-tensor-var) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (dimension :int)
  (biased :int)
  (keep-dim :int))
;; void THTensor_(norm)(THTensor *r_, THTensor *t, real value, int dimension, int keepdim);
(cffi:defcfun ("THDoubleTensor_norm" th-double-tensor-norm) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double)
  (dimension :int)
  (keep-dim :int))
;; void THTensor_(renorm)(THTensor *r_, THTensor *t, real value, int dimension, real maxnorm);
(cffi:defcfun ("THDoubleTensor_renorm" th-double-tensor-renorm) :void
  (result th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (value :double)
  (dimension :int)
  (maxnorm :double))
;; accreal THTensor_(dist)(THTensor *a, THTensor *b, real value);
(cffi:defcfun ("THDoubleTensor_dist" th-double-tensor-dist) :double
  (a th-double-tensor-ptr)
  (b th-double-tensor-ptr)
  (value :double))
;; void THTensor_(histc)(THTensor *hist, THTensor *tensor, long nbins, real minvalue, real maxvalue)
(cffi:defcfun ("THDoubleTensor_histc" th-double-tensor-histc) :void
  (hist th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (nbins :long)
  (min-value :double)
  (max-value :double))
;; void THTensor_(bhistc)(THTensor *hist, THTensor *tensor, long nbins, real minvalue, real maxvalue);
(cffi:defcfun ("THDoubleTensor_bhistc" th-double-tensor-bhistc) :void
  (hist th-double-tensor-ptr)
  (tensor th-double-tensor-ptr)
  (nbins :long)
  (min-value :double)
  (max-value :double))

;; accreal THTensor_(meanall)(THTensor *self);
(cffi:defcfun ("THDoubleTensor_meanall" th-double-tensor-mean-all) :double
  (tensor th-double-tensor-ptr))
;; accreal THTensor_(varall)(THTensor *self, int biased);
(cffi:defcfun ("THDoubleTensor_varall" th-double-tensor-var-all) :double
  (tensor th-double-tensor-ptr)
  (biased :int))
;; accreal THTensor_(stdall)(THTensor *self, int biased);
(cffi:defcfun ("THDoubleTensor_stdall" th-double-tensor-std-all) :double
  (tensor th-double-tensor-ptr)
  (biased :int))
;; accreal THTensor_(normall)(THTensor *t, real value);
(cffi:defcfun ("THDoubleTensor_normall" th-double-tensor-norm-all) :double
  (tensor th-double-tensor-ptr)
  (value :double))

;; void THTensor_(linspace)(THTensor *r_, real a, real b, long n);
(cffi:defcfun ("THDoubleTensor_linspace" th-double-tensor-linspace) :void
  (result th-double-tensor-ptr)
  (a :double)
  (b :double)
  (n :long))
;; void THTensor_(logspace)(THTensor *r_, real a, real b, long n);
(cffi:defcfun ("THDoubleTensor_logspace" th-double-tensor-logspace) :void
  (result th-double-tensor-ptr)
  (a :double)
  (b :double)
  (n :long))
;; void THTensor_(rand)(THTensor *r_, THGenerator *_generator, THLongStorage *size);
(cffi:defcfun ("THDoubleTensor_rand" th-double-tensor-rand) :void
  (result th-double-tensor-ptr)
  (generator th-generator-ptr)
  (size th-long-storage-ptr))
;; void THTensor_(randn)(THTensor *r_, THGenerator *_generator, THLongStorage *size);
(cffi:defcfun ("THDoubleTensor_randn" th-double-tensor-randn) :void
  (result th-double-tensor-ptr)
  (generator th-generator-ptr)
  (size th-long-storage-ptr))

;; #endif /* FLOAT OR DOUBLE */

;; #if defined(TH_REAL_IS_BYTE)

;; int THTensor_(logicalall)(THTensor *self);
;; (cffi:defcfun ("THDoubleTensor_logicalall" th-double-tensor-logical-all) :int
;;   (tensor (:pointer :void)))
;; int THTensor_(logicalany)(THTensor *self);
;; (cffi:defcfun ("THDoubleTensor_logicalany" th-double-tensor-logical-any) :int
;;   (tensor (:pointer :void)))

;; #endif /* TH_REAL_IS_BYTE */


;; void THTensor_(validXCorr2Dptr)(real *r_,
;;                                 real alpha,
;;                                 real *t_, long ir, long ic,
;;                                 real *k_, long kr, long kc,
;;                                 long sr, long sc);
(cffi:defcfun ("THDoubleTensor_validXCorr2Dptr" th-double-tensor-valid-x-corr-2d-ptr) :void
  (res (:pointer :double))
  (alpha :double)
  (ten (:pointer :double))
  (ir :long)
  (ic :long)
  (k (:pointer :double))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(validConv2Dptr)(real *r_,
;;                                real alpha,
;;                                real *t_, long ir, long ic,
;;                                real *k_, long kr, long kc,
;;                                long sr, long sc);
(cffi:defcfun ("THDoubleTensor_validConv2Dptr" th-double-tensor-valid-conv-2d-ptr) :void
  (res (:pointer :double))
  (alpha :double)
  (ten (:pointer :double))
  (ir :long)
  (ic :long)
  (k (:pointer :double))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(fullXCorr2Dptr)(real *r_,
;;                                real alpha,
;;                                real *t_, long ir, long ic,
;;                                real *k_, long kr, long kc,
;;                                long sr, long sc);
(cffi:defcfun ("THDoubleTensor_fullXCorr2Dptr" th-double-tensor-full-x-corr-2d-ptr) :void
  (res (:pointer :double))
  (alpha :double)
  (ten (:pointer :double))
  (ir :long)
  (ic :long)
  (k (:pointer :double))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(fullConv2Dptr)(real *r_,
;;                               real alpha,
;;                               real *t_, long ir, long ic,
;;                               real *k_, long kr, long kc,
;;                               long sr, long sc);
(cffi:defcfun ("THDoubleTensor_fullConv2Dptr" th-double-tensor-full-conv-2d-ptr) :void
  (res (:pointer :double))
  (alpha :double)
  (ten (:pointer :double))
  (ir :long)
  (ic :long)
  (k (:pointer :double))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(validXCorr2DRevptr)(real *r_,
;;                                    real alpha,
;;                                    real *t_, long ir, long ic,
;;                                    real *k_, long kr, long kc,
;;                                    long sr, long sc);
(cffi:defcfun ("THDoubleTensor_validXCorr2DRevptr" th-double-tensor-valid-x-corr-2d-rev-ptr) :void
  (res (:pointer :double))
  (alpha :double)
  (ten (:pointer :double))
  (ir :long)
  (ic :long)
  (k (:pointer :double))
  (kr :long)
  (kc :long)
  (sr :long)
  (sc :long))

;; void THTensor_(conv2DRevger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol);
(cffi:defcfun ("THDoubleTensor_conv2DRevger" th-double-tensor-conv-2d-rev-ger) :void
  (result th-double-tensor-ptr)
  (beta :double)
  (alpha :double)
  (tensor th-double-tensor-ptr)
  (k th-double-tensor-ptr)
  (srow :long)
  (scol :long))
;; void THTensor_(conv2DRevgerm)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol);
(cffi:defcfun ("THDoubleTensor_conv2DRevgerm" th-double-tensor-conv-2d-rev-germ) :void
  (result th-double-tensor-ptr)
  (beta :double)
  (alpha :double)
  (tensor th-double-tensor-ptr)
  (k th-double-tensor-ptr)
  (srow :long)
  (scol :long))
;; void THTensor_(conv2Dger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THDoubleTensor_conv2Dger" th-double-tensor-conv-2d-ger) :void
  (result th-double-tensor-ptr)
  (beta :double)
  (alpha :double)
  (tensor th-double-tensor-ptr)
  (k th-double-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf :string)
  (xc :string))
;; void THTensor_(conv2Dmv)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THDoubleTensor_conv2Dmv" th-double-tensor-conv-2d-mv) :void
  (result th-double-tensor-ptr)
  (beta :double)
  (alpha :double)
  (tensor th-double-tensor-ptr)
  (k th-double-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf :string)
  (xc :string))
;; void THTensor_(conv2Dmm)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THDoubleTensor_conv2Dmm" th-double-tensor-conv-2d-mm) :void
  (result th-double-tensor-ptr)
  (beta :double)
  (alpha :double)
  (tensor th-double-tensor-ptr)
  (k th-double-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf :string)
  (xc :string))
;; void THTensor_(conv2Dmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THDoubleTensor_conv2Dmul" th-double-tensor-conv-2d-mul) :void
  (result th-double-tensor-ptr)
  (beta :double)
  (alpha :double)
  (tensor th-double-tensor-ptr)
  (k th-double-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf :string)
  (xc :string))
;; void THTensor_(conv2Dcmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THDoubleTensor_conv2Dcmul" th-double-tensor-conv-2d-cmul) :void
  (result th-double-tensor-ptr)
  (beta :double)
  (alpha :double)
  (tensor th-double-tensor-ptr)
  (k th-double-tensor-ptr)
  (srow :long)
  (scol :long)
  (vf :string)
  (xc :string))

;; void THTensor_(validXCorr3Dptr)(real *r_,
;;                                 real alpha,
;;                                 real *t_, long it, long ir, long ic,
;;                                 real *k_, long kt, long kr, long kc,
;;                                 long st, long sr, long sc);
(cffi:defcfun ("THDoubleTensor_validXCorr3Dptr" th-double-tensor-valid-x-corr-3d-ptr) :void
  (res (:pointer :double))
  (alpha :double)
  (ten (:pointer :double))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :double))
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
(cffi:defcfun ("THDoubleTensor_validConv3Dptr" th-double-tensor-valid-conv-3d-ptr) :void
  (res (:pointer :double))
  (alpha :double)
  (ten (:pointer :double))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :double))
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
(cffi:defcfun ("THDoubleTensor_fullXCorr3Dptr" th-double-tensor-full-x-corr-3d-ptr) :void
  (res (:pointer :double))
  (alpha :double)
  (ten (:pointer :double))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :double))
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
(cffi:defcfun ("THDoubleTensor_fullConv3Dptr" th-double-tensor-full-conv-3d-ptr) :void
  (res (:pointer :double))
  (alpha :double)
  (ten (:pointer :double))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :double))
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
(cffi:defcfun ("THDoubleTensor_validXCorr3DRevptr" th-double-tensor-valid-x-corr-3d-rev-ptr) :void
  (res (:pointer :double))
  (alpha :double)
  (ten (:pointer :double))
  (it :long)
  (ir :long)
  (ic :long)
  (k (:pointer :double))
  (kt :long)
  (kr :long)
  (kc :long)
  (st :long)
  (sr :long)
  (sc :long))

;; void THTensor_(conv3DRevger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol);
(cffi:defcfun ("THDoubleTensor_conv3DRevger" th-double-tensor-conv-3d-rev-ger) :void
  (result th-double-tensor-ptr)
  (beta :double)
  (alpha :double)
  (tensor th-double-tensor-ptr)
  (k th-double-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long))
;; void THTensor_(conv3Dger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THDoubleTensor_conv3Dger" th-double-tensor-conv-3d-ger) :void
  (result th-double-tensor-ptr)
  (beta :double)
  (alpha :double)
  (tensor th-double-tensor-ptr)
  (k th-double-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf :string)
  (xc :string))
;; void THTensor_(conv3Dmv)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THDoubleTensor_conv3Dmv" th-double-tensor-conv-3d-mv) :void
  (result th-double-tensor-ptr)
  (beta :double)
  (alpha :double)
  (tensor th-double-tensor-ptr)
  (k th-double-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf :string)
  (xc :string))
;; void THTensor_(conv3Dmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THDoubleTensor_conv3Dmul" th-double-tensor-conv-3d-mul) :void
  (result th-double-tensor-ptr)
  (beta :double)
  (alpha :double)
  (tensor th-double-tensor-ptr)
  (k th-double-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf :string)
  (xc :string))
;; void THTensor_(conv3Dcmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
(cffi:defcfun ("THDoubleTensor_conv3Dcmul" th-double-tensor-conv-3d-cmul) :void
  (result th-double-tensor-ptr)
  (beta :double)
  (alpha :double)
  (tensor th-double-tensor-ptr)
  (k th-double-tensor-ptr)
  (sdepth :long)
  (srow :long)
  (scol :long)
  (vf :string)
  (xc :string))

;; /* Level 1 */
;; void THBlas_(swap)(long n, real *x, long incx, real *y, long incy);
(cffi:defcfun ("THDoubleBlas_swap" th-double-blas-swap) :void
  (n :long)
  (x (:pointer :double))
  (incx :long)
  (y (:pointer :double))
  (incy :long))
;; void THBlas_(scal)(long n, real a, real *x, long incx);
(cffi:defcfun ("THDoubleBlas_scal" th-double-blas-scal) :void
  (n :long)
  (a :double)
  (x (:pointer :double))
  (incx :long))
;; void THBlas_(copy)(long n, real *x, long incx, real *y, long incy);
(cffi:defcfun ("THDoubleBlas_copy" th-double-blas-copy) :void
  (n :long)
  (x (:pointer :double))
  (incx :long)
  (y (:pointer :double))
  (incy :long))
;; void THBlas_(axpy)(long n, real a, real *x, long incx, real *y, long incy);
(cffi:defcfun ("THDoubleBlas_axpy" th-double-blas-axpy) :void
  (n :long)
  (a :double)
  (x (:pointer :double))
  (incx :long)
  (y (:pointer :double))
  (incy :long))
;; real THBlas_(dot)(long n, real *x, long incx, real *y, long incy);
(cffi:defcfun ("THDoubleBlas_dot" th-double-blas-dot) :void
  (n :long)
  (x (:pointer :double))
  (incx :long)
  (y (:pointer :double))
  (incy :long))

;; /* Level 2 */
;; void THBlas_(gemv)(char trans, long m, long n, real alpha, real *a, long lda, real *x, long incx, real beta, real *y, long incy);
(cffi:defcfun ("THDoubleBlas_gemv" th-double-blas-gemv) :void
  (trans :char)
  (m :long)
  (n :long)
  (alpha :double)
  (a (:pointer :double))
  (lda :long)
  (x (:pointer :double))
  (incx :long)
  (beta :double)
  (y (:pointer :double))
  (incy :long))
;; void THBlas_(ger)(long m, long n, real alpha, real *x, long incx, real *y, long incy, real *a, long lda);
(cffi:defcfun ("THDoubleBlas_ger" th-double-blas-ger) :void
  (m :long)
  (n :long)
  (alpha :double)
  (x (:pointer :double))
  (incx :long)
  (y (:pointer :double))
  (incy :long)
  (a (:pointer :double))
  (lda :long))

;; /* Level 3 */
;; void THBlas_(gemm)(char transa, char transb, long m, long n, long k, real alpha, real *a, long lda, real *b, long ldb, real beta, real *c, long ldc);
(cffi:defcfun ("THDoubleBlas_gemm" th-double-blas-gemm) :void
  (transa :char)
  (transb :char)
  (m :long)
  (n :long)
  (k :long)
  (alpha :double))

;; void THTensor_(gesv)(THTensor *rb_, THTensor *ra_, THTensor *b_, THTensor *a_);
(cffi:defcfun ("THDoubleTensor_gesv" th-double-tensor-gesv) :void
  (resultb th-double-tensor-ptr)
  (resulta th-double-tensor-ptr)
  (b th-double-tensor-ptr)
  (a th-double-tensor-ptr))
;; void THTensor_(trtrs)(THTensor *rb_, THTensor *ra_, THTensor *b_, THTensor *a_, const char *uplo, const char *trans, const char *diag);
(cffi:defcfun ("THDoubleTensor_trtrs" th-double-tensor-trtrs) :void
  (resultb th-double-tensor-ptr)
  (resulta th-double-tensor-ptr)
  (b th-double-tensor-ptr)
  (a th-double-tensor-ptr)
  (uplo :string)
  (trans :string)
  (diag :string))
;; void THTensor_(gels)(THTensor *rb_, THTensor *ra_, THTensor *b_, THTensor *a_);
(cffi:defcfun ("THDoubleTensor_gels" th-double-tensor-gels) :void
  (resultb th-double-tensor-ptr)
  (resulta th-double-tensor-ptr)
  (b th-double-tensor-ptr)
  (a th-double-tensor-ptr))
;; void THTensor_(syev)(THTensor *re_, THTensor *rv_, THTensor *a_, const char *jobz, const char *uplo);
(cffi:defcfun ("THDoubleTensor_syev" th-double-tensor-syev) :void
  (resulte th-double-tensor-ptr)
  (resultv th-double-tensor-ptr)
  (a th-double-tensor-ptr)
  (jobz :string)
  (uplo :string))
;; void THTensor_(geev)(THTensor *re_, THTensor *rv_, THTensor *a_, const char *jobvr);
(cffi:defcfun ("THDoubleTensor_geev" th-double-tensor-geev) :void
  (resulte th-double-tensor-ptr)
  (resultv th-double-tensor-ptr)
  (a th-double-tensor-ptr)
  (jobvr :string))
;; void THTensor_(gesvd)(THTensor *ru_, THTensor *rs_, THTensor *rv_, THTensor *a, const char *jobu)
(cffi:defcfun ("THDoubleTensor_gesvd" th-double-tensor-gesvd) :void
  (resultu th-double-tensor-ptr)
  (results th-double-tensor-ptr)
  (resultv th-double-tensor-ptr)
  (a th-double-tensor-ptr)
  (jobu :string))
;; void THTensor_(gesvd2)(THTensor *ru_, THTensor *rs_, THTensor *rv_, THTensor *ra_, THTensor *a, const char *jobu);
(cffi:defcfun ("THDoubleTensor_gesvd2" th-double-tensor-gesvd2) :void
  (resultu th-double-tensor-ptr)
  (results th-double-tensor-ptr)
  (resultv th-double-tensor-ptr)
  (resulta th-double-tensor-ptr)
  (a th-double-tensor-ptr)
  (jobu :string))
;; void THTensor_(getri)(THTensor *ra_, THTensor *a);
(cffi:defcfun ("THDoubleTensor_getri" th-double-tensor-getri) :void
  (resulta th-double-tensor-ptr)
  (a th-double-tensor-ptr))
;; void THTensor_(potrf)(THTensor *ra_, THTensor *a, const char *uplo);
(cffi:defcfun ("THDoubleTensor_potrf" th-double-tensor-potrf) :void
  (resulta th-double-tensor-ptr)
  (a th-double-tensor-ptr)
  (uplo :string))
;; void THTensor_(potrs)(THTensor *rb_, THTensor *b_, THTensor *a_,  const char *uplo);
(cffi:defcfun ("THDoubleTensor_potrs" th-double-tensor-potrs) :void
  (resultb th-double-tensor-ptr)
  (b th-double-tensor-ptr)
  (a th-double-tensor-ptr)
  (uplo :string))
;; void THTensor_(potri)(THTensor *ra_, THTensor *a, const char *uplo);
(cffi:defcfun ("THDoubleTensor_potri" th-double-tensor-potri) :void
  (resulta th-double-tensor-ptr)
  (a th-double-tensor-ptr)
  (uplo :string))
;; void THTensor_(qr)(THTensor *rq_, THTensor *rr_, THTensor *a);
(cffi:defcfun ("THDoubleTensor_qr" th-double-tensor-qr) :void
  (resultq th-double-tensor-ptr)
  (resultr th-double-tensor-ptr)
  (a th-double-tensor-ptr))
;; void THTensor_(geqrf)(THTensor *ra_, THTensor *rtau_, THTensor *a);
(cffi:defcfun ("THDoubleTensor_geqrf" th-double-tensor-geqrf) :void
  (resulta th-double-tensor-ptr)
  (resulttau th-double-tensor-ptr)
  (a th-double-tensor-ptr))
;; void THTensor_(orgqr)(THTensor *ra_, THTensor *a, THTensor *tau);
(cffi:defcfun ("THDoubleTensor_orgqr" th-double-tensor-orgqr) :void
  (resulta th-double-tensor-ptr)
  (a th-double-tensor-ptr)
  (tau th-double-tensor-ptr))
;; void THTensor_(ormqr)(THTensor *ra_, THTensor *a, THTensor *tau, THTensor *c, const char *side, const char *trans);
(cffi:defcfun ("THDoubleTensor_ormqr" th-double-tensor-ormqr) :void
  (resulta th-double-tensor-ptr)
  (a th-double-tensor-ptr)
  (tau th-double-tensor-ptr)
  (c th-double-tensor-ptr)
  (side :string)
  (trans :string))
;; void THTensor_(pstrf)(THTensor *ra_, THIntTensor *rpiv_, THTensor*a, const char* uplo, real tol);
(cffi:defcfun ("THDoubleTensor_pstrf" th-double-tensor-pstrf) :void
  (resulta th-double-tensor-ptr)
  (resultpiv th-int-tensor-ptr)
  (a th-double-tensor-ptr)
  (uplo :string)
  (tol :double))

;; void THTensor_(btrifact)(THTensor *ra_, THIntTensor *rpivots_, THIntTensor *rinfo_, int pivot, THTensor *a);
(cffi:defcfun ("THDoubleTensor_btrifact" th-double-tensor-btrifact) :void
  (resulta th-double-tensor-ptr)
  (resultpivots th-int-tensor-ptr)
  (resultinfo th-int-tensor-ptr)
  (pivot :int)
  (a th-double-tensor-ptr))
;; void THTensor_(btrisolve)(THTensor *rb_, THTensor *b, THTensor *atf, THIntTensor *pivots);
(cffi:defcfun ("THDoubleTensor_btrisolve" th-double-tensor-btrisolve) :void
  (resultb th-double-tensor-ptr)
  (b th-double-tensor-ptr)
  (atf th-double-tensor-ptr)
  (pivots th-int-tensor-ptr))
