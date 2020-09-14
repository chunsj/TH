(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defgeneric tensor-coerce (tensor value)
  (:documentation "returns coerced value for given tensor"))

(defgeneric tensor-acoerce (tensor value)
  (:documentation "returns coerced value for given tensor for accumulated type"))

(defgeneric tensor-type (tensor)
  (:documentation "returns type of elements of the tensor"))

(defgeneric tensor-at (tensor location &rest others)
  (:documentation "accessing tensor elements"))

(defgeneric (setf tensor-at) (value tensor location &rest others)
  (:documentation "setf for tensor-at method"))

(defgeneric tensor-storage (tensor)
  (:documentation "returns storage of the tensor"))
(defgeneric tensor-storage-offset (tensor)
  (:documentation "returns the 1st index used in tensor's storage"))
(defgeneric tensor-n-dimension (tensor)
  (:documentation "returns the number of dimensions of given tensor"))
(defgeneric tensor-size (tensor dimension)
  (:documentation "returns dimensions of the tensor"))
(defgeneric tensor-stride (tensor dimension)
  (:documentation "returns the jumps to go from one to next element in tensor"))
(defgeneric tensor-data (tensor)
  (:documentation "returns pointer for tensor data"))

(defgeneric tensor-with-tensor (tensor)
  (:documentation "returns a new tensor which shared the same storage of the given tensor"))
(defgeneric tensor-clone (tensor)
  (:documentation "returns a new cloned tensor"))
(defgeneric tensor-contiguous (tensor)
  (:documentation "returns a new contiguously allocated layout tensor if it's not"))
(defgeneric tensor-with-storage (storage &optional storage-offset size stride)
  (:documentation "returns a new tensor with given storage"))

(defgeneric tensor-new-select (tensor dimension slice-index)
  (:documentation "returns a new tensor slice at the slice-index in the given dimension"))
(defgeneric tensor-new-narrow (tensor dimension first-index size)
  (:documentation "returns a new tensor that is a narrowed version of src tensor"))
(defgeneric tensor-new-transpose (tensor)
  (:documentation "returns a new tensor that is the transposed view of the tensor"))
(defgeneric tensor-new-unfold (tensor dimension size step)
  (:documentation "returns a new tensor which contains all slice of the given size by step"))
(defgeneric tensor-new-view (tensor size)
  (:documentation "returns a new tensor which different dimensions of the same storage"))

(defgeneric tensor-expand (tensor src size)
  (:documentation "returns a new view of the src with singleton dimension expanded"))

(defgeneric tensor-resize-as (tensor src))
(defgeneric tensor-resize (tensor size &optional stride))

(defgeneric tensor-set (tensor src))
(defgeneric tensor-set-storage (tensor storage offset size stride))

(defgeneric tensor-narrow (tensor src dimension first-idx size)
  (:documentation "returns a tensor that is a narrowed version of src tensor"))
(defgeneric tensor-select (tensor src dimension slice-index)
  (:documentation "returns a tensor which is a slice at the given index in the dim"))
(defgeneric tensor-transpose (tensor src dimension0 dimension1)
  (:documentation "returns a transposed tensor between dimensions"))
(defgeneric tensor-unfold (tensor src dimension size step))
(defgeneric tensor-squeeze (tensor src &optional dimension)
  (:documentation "removes all the dimensions with size 1"))
(defgeneric tensor-unsqueeze (tensor src dimension))

(defgeneric tensor-contiguous-p (tensor)
  (:documentation "check whether tensor is contiguous in memory in C order"))
(defgeneric tensor-same-size-p (tensor src))
(defgeneric tensor-set-to-p (tensor src))
(defgeneric tensor-size-p (tensor dims))
(defgeneric tensor-n-element (tensor))

(defgeneric tensor-copy (tensor src)
  (:documentation "returns the already same sized element-wise copied tensor from the source"))

(defgeneric tensor-random (tensor &optional generator))
(defgeneric tensor-geometric (tensor p &optional generator)
  (:documentation "returns geometric(p) element-wise"))
(defgeneric tensor-bernoulli (tensor p &optional generator)
  (:documentation "returns bernoulli(p) element-wise"))

(defgeneric tensor-uniform (tensor a b &optional generator)
  (:documentation "returns tensor filled with number drawn from uniform distribution"))
(defgeneric tensor-normal (tensor mean stdev &optional generator)
  (:documentation "returns tensor filled with numbers drawn from normal distribution"))
(defgeneric tensor-exponential (tensor lam &optional generator)
  (:documentation "returns tensor filled with numbers drawn from exponential distribution"))
(defgeneric tensor-cauchy (tensor median sigma &optional generator)
  (:documentation "returns tensor filled with numbers drawn from cauchy distribution"))
(defgeneric tensor-log-normal (tensor mean stdev &optional generator)
  (:documentation "returns tensor filled with numbers drawn from log normal distribution"))
(defgeneric tensor-multinomial (tensor pdist nsample replacement &optional generator)
  (:documentation "returns tensor filled with numbers drawn from multinomial distribution"))
(defgeneric tensor-multinomial-alias-setup (pdist J q))
(defgeneric tensor-multinomial-alias-draw (tensor J q &optional generator))

(defgeneric tensor-fill (tensor value)
  (:documentation "fills element of the tensor with value"))
(defgeneric tensor-zero (tensor)
  (:documentation "fills element of the tensor with zero"))

(defgeneric tensor-masked-fill (tensor mask value)
  (:documentation "fills element of tensor at the location of one in byte tensor mask"))
(defgeneric tensor-masked-copy (tensor mask src))
(defgeneric tensor-masked-select (tensor mask src)
  (:documentation "returns a vector from elements of src masked by byte mask"))

(defgeneric tensor-non-zero (tensor &optional indices)
  (:documentation "returns indices of non zero elements in tnsor"))

(defgeneric tensor-index-select (tensor src dim index))
(defgeneric tensor-index-copy (tensor src dim index))
(defgeneric tensor-index-add (tensor src dim index))
(defgeneric tensor-index-fill (tensor value dim index))

(defgeneric tensor-gather (tensor src dim index))
(defgeneric tensor-scatter (tensor src dim index))
(defgeneric tensor-scatter-add (tensor src dim index))
(defgeneric tensor-scatter-fill (tensor value dim index))

(defgeneric tensor-dot (tensor1 tensor2)
  (:documentation "returns dot product of two tensors"))

(defgeneric tensor-min-all (tensor))
(defgeneric tensor-max-all (tensor))
(defgeneric tensor-median-all (tensor))
(defgeneric tensor-sum-all (tensor))
(defgeneric tensor-prd-all (tensor))

(defgeneric tensor-neg (tensor src))
(defgeneric tensor-cinv (tensor src))

(defgeneric tensor-add (tensor src value)
  (:documentation "returns element-wise addition of scalar value"))
(defgeneric tensor-sub (tensor src value))
(defgeneric tensor-mul (tensor src value))
(defgeneric tensor-div (tensor src value)
  (:documentation "returns element-wise division of scalar value"))
(defgeneric tensor-mod (tensor src value))
(defgeneric tensor-fmod (tensor src value)
  (:documentation "returns element-wise remainders"))
(defgeneric tensor-clamp (tensor src min max)
  (:documentation "returns tensor with its elements clamped between min and max"))
(defgeneric tensor-lshift (tensor src value))
(defgeneric tensor-rshift (tensor src value))
(defgeneric tensor-bitand (tensor src value))
(defgeneric tensor-bitor (tensor src value))
(defgeneric tensor-bitxor (tensor src value))

(defgeneric tensor-cadd (tensor src1 value src2))
(defgeneric tensor-csub (tensor src1 value src2))
(defgeneric tensor-cmul (tensor src1 src2))
(defgeneric tensor-cpow (tensor src1 src2))
(defgeneric tensor-cdiv (tensor src1 src2))
(defgeneric tensor-clshift (tensor src1 src2))
(defgeneric tensor-crshift (tensor src1 src2))
(defgeneric tensor-cmod (tensor src1 src2))
(defgeneric tensor-cfmod (tensor src1 src2))
(defgeneric tensor-cbitand (tensor src1 src2))
(defgeneric tensor-cbitor (tensor src1 src2))
(defgeneric tensor-cbitxor (tensor src1 src2))

(defgeneric tensor-add-cmul (tensor src1 value src2 src3)
  (:documentation "returns src1 + value * src2 * src3, element-wise"))
(defgeneric tensor-add-cdiv (tensor src1 value src2 src3)
  (:documentation "returns src1 + value * (src2 / src3), element-wise"))
(defgeneric tensor-add-mv (tensor beta y alpha A x)
  (:documentation "returns beta * y + alpha * A @ x"))
(defgeneric tensor-add-mm (tensor beta C alpha A B)
  (:documentation "returns beta * C + alpha * A @ B"))
(defgeneric tensor-add-r (tensor beta A alpha x y)
  (:documentation "returns beta * A + alpha * (x cross y)"))
(defgeneric tensor-add-bmm (tensor beta C alpha batchA batchB)
  (:documentation "returns beta * C + alpha * [sum of batchA @ batch B]"))
(defgeneric tensor-badd-bmm (tensor beta batchC alpha batchA batchB)
  (:documentation "returns beta * batchC(i) + alpha * (batchA(i) @ batchB(i))"))

(defgeneric tensor-match (tensor m1 m2 gain))

(defgeneric tensor-max (vals indices tensor dimension keep-dim)
  (:documentation "returns max values and indices with given dimension axis"))
(defgeneric tensor-min (vals indices tensor dimension keep-dim)
  (:documentation "returns min values and indices with given dimension axis"))
(defgeneric tensor-kth-value (vals indices tensor k dimension keep-dim)
  (:documentation "returns kth smallest element of the given tensor along given dimension"))
(defgeneric tensor-mode (vals indices tensor dimension keep-dim)
  (:documentation "returns the mode value of each row specified by dimension (-1 as default)"))
(defgeneric tensor-median (vals indices tensor dimension keep-dim))
(defgeneric tensor-sum (tensor src dimension keep-dim))
(defgeneric tensor-prd (tensor src dimension keep-dim))
(defgeneric tensor-cum-sum (tensor src dimension)
  (:documentation "returns cumulative sum value along dimension axis"))
(defgeneric tensor-cum-prd (tensor src dimension)
  (:documentation "returns cumulative product value along dimension axis"))
(defgeneric tensor-sign (tensor src))
(defgeneric tensor-trace (tensor))
(defgeneric tensor-cross (tensor A B dimension))

(defgeneric tensor-cmax (tensor src1 src2))
(defgeneric tensor-cmin (tensor src1 src2))

(defgeneric tensor-zeros (tensor size))
(defgeneric tensor-ones (tensor size))
(defgeneric tensor-diag (tensor src k))
(defgeneric tensor-eye (tensor nrow ncol))
(defgeneric tensor-arange (tensor xmin xmax step))
(defgeneric tensor-range (tensor xmin xmax step))
(defgeneric tensor-rand-perm (tensor n &optional generator)
  (:documentation "returns random permutation of integers from 0 to n - 1"))

(defgeneric tensor-reshape (tensor src size)
  (:documentation "reshapes size of the given tensor src"))
(defgeneric tensor-sort (tensor indices src dimension descending))
(defgeneric tensor-top-k (tensor indices src k dim dir sorted)
  (:documentation "returns k smallest/largest(dir) values sorted or not"))
(defgeneric tensor-tri-l (tensor src k))
(defgeneric tensor-tri-u (tensor src k))
(defgeneric tensor-cat (tensor dimension &rest srcs))
(defgeneric tensor-cat2 (tensor dimension src1 src2))
(defgeneric tensor-catn (tensor dimension srcs))
(defgeneric tensor-equal (tensor-a tensor-b)
  (:documentation "check equality of element in tensors"))
(defgeneric tensor-compare (operation tensor src1 src2)) ;; :lt :le :gt :ge :ne :eq

(defgeneric tensor-sigmoid (tensor src)
  (:documentation "returns element-wise sigmoid values"))
(defgeneric tensor-log (tensor src)
  (:documentation "returns element-wise log values"))
(defgeneric tensor-gamma (tensor src))
(defgeneric tensor-lgamma (tensor src))
(defgeneric tensor-erf (tensor src))
(defgeneric tensor-erfc (tensor src))
(defgeneric tensor-log1p (tensor src))
(defgeneric tensor-exp (tensor src)
  (:documentation "returns element-wise exponential values"))
(defgeneric tensor-cos (tensor src)
  (:documentation "returns element-wise cos values"))
(defgeneric tensor-acos (tensor src)
  (:documentation "returns element-wise acos values"))
(defgeneric tensor-cosh (tensor src)
  (:documentation "returns element-wise cosh values"))
(defgeneric tensor-sin (tensor src)
  (:documentation "returns element-wise sin values"))
(defgeneric tensor-asin (tensor src)
  (:documentation "returns element-wise asin values"))
(defgeneric tensor-sinh (tensor src)
  (:documentation "returns element-wise sinh values"))
(defgeneric tensor-tan (tensor src)
  (:documentation "returns element-wise tan values"))
(defgeneric tensor-atan (tensor src)
  (:documentation "returns element-wise atan values"))
(defgeneric tensor-atan2 (tensor srcx srcy)
  (:documentation "returns element-wise atan(x/y)"))
(defgeneric tensor-tanh (tensor src)
  (:documentation "returns element-wise tanh values"))
(defgeneric tensor-pow (tensor src exponent)
  (:documentation "returns element-wise pow values"))
(defgeneric tensor-sqrt (tensor src)
  (:documentation "returns element-wise sqrt values"))
(defgeneric tensor-rsqrt (tensor src))
(defgeneric tensor-ceil (tensor src)
  (:documentation "returns ceil of the elements"))
(defgeneric tensor-floor (tensor src)
  (:documentation "returns element-wise floor values"))
(defgeneric tensor-round (tensor src)
  (:documentation "returns element-wise rounded values"))
(defgeneric tensor-abs (tensor src)
  (:documentation "returns element-wise abs values"))
(defgeneric tensor-trunc (tensor src))
(defgeneric tensor-frac (tensor src)
  (:documentation "returns element-wise fractional values"))
(defgeneric tensor-lerp (tensor tensor-a tensor-b weight)
  (:documentation "performs linear interpolation of tensor a and b; a + weight * (b - a)"))

(defgeneric tensor-mean (tensor src dimension keep-dim))
(defgeneric tensor-sd (tensor src dimension keep-dim &optional biased))
(defgeneric tensor-var (tensor src dimension keep-dim &optional biased))
(defgeneric tensor-norm (tensor src value dimension keep-dim))
(defgeneric tensor-renorm (tensor src value dimension maxnorm)
  (:documentation "returns a tensor whose subtensors along dimension is normalized (lower maxnorm)"))
(defgeneric tensor-dist (tensor-a tensor-b value))
(defgeneric tensor-histc (hist tensor nbins min-value max-value))
(defgeneric tensor-bhistc (hist tensor nbins min-value max-value))

(defgeneric tensor-mean-all (tensor))
(defgeneric tensor-var-all (tensor &optional biased))
(defgeneric tensor-sd-all (tensor &optional biased))
(defgeneric tensor-norm-all (tensor value))

(defgeneric tensor-linspace (tensor a b n))
(defgeneric tensor-logspace (tensor a b n))
(defgeneric tensor-rand (tensor size &optional generator))
(defgeneric tensor-randn (tensor size &optional generator))

(defgeneric tensor-conv-2d-rev-ger (result beta alpha tensor k srow scol))
(defgeneric tensor-conv-2d-rev-germ (result beta alpha tensor k srow scol))
(defgeneric tensor-conv-2d-ger (result beta alpha tensor k srow scol vf xc))
(defgeneric tensor-conv-2d-mv (result beta alpha tensor k srow scol vf xc))
(defgeneric tensor-conv-2d-mm (result beta alpha tensor k srow scol vf xc))
(defgeneric tensor-conv-2d-mul (result beta alpha tensor k srow scol vf xc))
(defgeneric tensor-conv-2d-cmul (result beta alpha tensor k srow scol vf xc))

(defgeneric tensor-conv-3d-rev-ger (result beta alpha tensor k sdepth srow scol))
(defgeneric tensor-conv-3d-ger (result beta alpha tensor k sdepth srow scol vf xc))
(defgeneric tensor-conv-3d-mv (result beta alpha tensor k sdepth srow scol vf xc))
(defgeneric tensor-conv-3d-mul (result beta alpha tensor k sdepth srow scol vf xc))
(defgeneric tensor-conv-3d-cmul (result beta alpha tensor k sdepth srow scol vf xc))

(defun make-tensor-rand (tensor sizes)
  (let ((sz (if (typep ($last sizes) 'generator)
                (butlast sizes)
                sizes))
        (g (if (typep ($last sizes) 'generator)
               ($last sizes)
               *generator*)))
    (tensor-rand tensor sz g)))

(defun tensor.float-rand (&rest sizes) (make-tensor-rand (tensor.float) sizes))
(defun tensor.double-rand (&rest sizes) (make-tensor-rand (tensor.double) sizes))

(defun make-tensor-randn (tensor sizes)
  (let ((sz (if (typep ($last sizes) 'generator)
                (butlast sizes)
                sizes))
        (g (if (typep ($last sizes) 'generator)
               ($last sizes)
               *generator*)))
    (tensor-randn tensor sz g)))

(defun tensor.float-randn (&rest sizes) (make-tensor-randn (tensor.float) sizes))
(defun tensor.double-randn (&rest sizes) (make-tensor-randn (tensor.double) sizes))

(defun byte-zeros (&rest sizes) (-> (apply #'tensor.byte sizes) (tensor-zero)))
(defun char-zeros (&rest sizes) (-> (apply #'tensor.char sizes) (tensor-zero)))
(defun short-zeros (&rest sizes) (-> (apply #'tensor.short sizes) (tensor-zero)))
(defun int-zeros (&rest sizes) (-> (apply #'tensor.int sizes) (tensor-zero)))
(defun long-zeros (&rest sizes) (-> (apply #'tensor.long sizes) (tensor-zero)))
(defun float-zeros (&rest sizes) (-> (apply #'tensor.float sizes) (tensor-zero)))
(defun double-zeros (&rest sizes) (-> (apply #'tensor.double sizes) (tensor-zero)))

(defgeneric tensor-logical-all (tensor))
(defgeneric tensor-logical-any (tensor))

(defgeneric tensor-gesv (rb ra b a))
(defgeneric tensor-trtrs (rb ra b a uplo trans diag))
(defgeneric tensor-gels (rb ra b a))
(defgeneric tensor-syev (re rv a jobz uplo)
  (:documentation "computes eigenvalues and eigenvectors"))
(defgeneric tensor-geev (re rv a jobvr))
(defgeneric tensor-gesvd (ru rs rv a jobu))
(defgeneric tensor-gesvd2 (ru rs rv ra a jobu))
(defgeneric tensor-getri (ra a))
(defgeneric tensor-potrf (ra a uplo)
  (:documentation "computes the cholesky decomposition of symmetric positive-definite matrix a"))
(defgeneric tensor-potrs (rb b a uplo)
  (:documentation "solves a linear system of equations with its cholesky factor u"))
(defgeneric tensor-potri (ra a uplo)
  (:documentation "inverses a positive semidefinite matrix given its cholesky factor u"))
(defgeneric tensor-qr (rq rr a))
(defgeneric tensor-geqrf (ra rtau a))
(defgeneric tensor-orgqr (ra a tau))
(defgeneric tensor-ormqr (ra a tau c side trans))
(defgeneric tensor-pstrf (ra rpiv a uplo tol)
  (:documentation "computes the pivoted cholesky decomposition"))
(defgeneric tensor-btrifact (ra rpivots rinfo a pivot)
  (:documentation "performs batch LU factorization"))
(defgeneric tensor-btrisolve (rb b atf pivots)
  (:documentation "performs batch LU solve; uses LU factorization results"))
