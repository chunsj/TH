(in-package :th)

(defgeneric $validp (generator)
  (:documentation "Returns whether generator is valid one or not."))

(defgeneric $seed (generator)
  (:documentation "Returns current seed of generator."))

(defgeneric $random (generator)
  (:documentation "Returns random number from generator."))

(defgeneric $uniform (object a b)
  (:documentation "Returns uniform random number between a and b."))
(defgeneric $normal (object mean stdv)
  (:documentation "Returns normal random number from N(mean,stdv)."))
(defgeneric $exponential (object lam)
  (:documentation "Returns exponential random number with rate lam."))
(defgeneric $cauchy (object median sigma)
  (:documentation "Returns cauchy random number."))
(defgeneric $lognormal (object mean stdv)
  (:documentation "Returns log normal random number from N(mean,stdv)."))
(defgeneric $geometric (object p)
  (:documentation "Returns geometric random number."))
(defgeneric $bernoulli (object p)
  (:documentation "Returns bernoulli random number."))

(defgeneric $storagep (object)
  (:documentation "Returns whether object is storage or not."))
(defgeneric $tensorp (object)
  (:documentation "Returns whether object is tensor or not."))

(defgeneric $empty (object)
  (:documentation "Returns empty new object from object type."))
(defgeneric $list (object)
  (:documentation "Returns contents of object as a list."))

(defgeneric $handle (object)
  (:documentation "Returns native handle of the object."))
(defgeneric $pointer (object)
  (:documentation "Returns pointer of the data of the object."))
(defgeneric $type (object)
  (:documentation "Returns native type tag of the object."))
(defgeneric $storage (tensor)
  (:documentation "Returns storage of tensor."))
(defgeneric $offset (tensor)
  (:documentation "Returns storage offset."))
(defgeneric $ndim (tensor)
  (:documentation "Returns the number of dimensions of tensor."))
(defgeneric $size (object &optional dimension)
  (:documentation "Returns size of the object along dimension."))
(defgeneric $stride (tensor &optional dimension)
  (:documentation "Returns stride of the tensor along dimension."))

(defgeneric $coerce (object value)
  (:documentation "Returns coerced value for the given object type."))
(defgeneric $acoerce (object value)
  (:documentation "Returns coerced value for accumulation."))

(defgeneric $contiguous (tensor)
  (:documentation "Returns a new contiguously allocated tensor if it's not."))
(defgeneric $contiguousp (tensor)
  (:documentation "Returns whether tensor is contiguously allocated."))

(defgeneric $resize (object size &optional stride)
  (:documentation "Resizes object as size and stride."))
(defgeneric $copy (object source)
  (:documentation "Copies content from source."))
(defgeneric $swap (object1 object2)
  (:documentation "Swaps the contents of objects."))

(defgeneric $fill (object value)
  (:documentation "Returns a new tensor filled with value."))
(defgeneric $fill! (object value)
  (:documentation "Returns a tensor filled with value."))
(defgeneric $zero (tensor)
  (:documentation "Returns a new tensor of whose elements of tensor as zero."))
(defgeneric $zero! (tensor)
  (:documentation "Fills elements of tensor as zero."))
(defgeneric $one (tensor)
  (:documentation "Fills elements of tensor as one."))
(defgeneric $one! (tensor)
  (:documentation "Returns a new tensor of whose elements of tensor as one."))

(defgeneric $clone (tensor)
  (:documentation "Returns a new cloned tensor."))
(defgeneric $sizep (tensor other)
  (:documentation "Checks whether tensor has the same size of other or other dimension."))

(defgeneric $set (tensor source &optional offset size stride)
  (:documentation "Sets the storage contents of source to tensor."))
(defgeneric $setp (tensor source)
  (:documentation "Checks whether tensor is set with source."))

(defgeneric $transpose (tensor &optional dimension0 dimension1)
  (:documentation "Returns a new transposed tensor between dimensions."))
(defgeneric $transpose! (tensor &optional dimension0 dimension1)
  (:documentation "Returns a transposed tensor between dimensions."))

(defgeneric $view (tensor &rest sizes)
  (:documentation "Returns a new tensor which has different dimension of the same storage."))
(defgeneric $subview (tensor &rest index-sizes)
  (:documentation "Returns a new tensor which is a subview of the tensor."))

(defgeneric $select (tensor dimension slice-index)
  (:documentation "Returns a new tensor slice at slice-index along dimension."))
(defgeneric $select! (tensor dimension slice-index)
  (:documentation "Returns a tensor slice at slice-index along dimension"))
(defgeneric $narrow (tensor dimension first-index size)
  (:documentation "Returns a new tensor that is a narrowed."))
(defgeneric $narrow! (tensor dimension first-index size)
  (:documentation "Returns a tensor that is a narrowed."))

(defgeneric $unfold (tensor dimension size step)
  (:documentation "Returns a new tensor with all slices of the given size by step."))
(defgeneric $unfold! (tensor dimension size step)
  (:documentation "Returns a tensor with all slices of the given size by step."))

(defgeneric $index (tensor dimension index)
  (:documentation "Returns a new tensor with contents selected by index along dimension."))

(defgeneric $gather (tensor dimension indices)
  (:documentation "Returns a new tensor by gathering elements from indices along dimension."))
(defgeneric $scatter (tensor dimension indices value)
  (:documentation "Writes value specified by indices along dimension."))

(defgeneric $masked (tensor mask)
  (:documentation "Returns one dimensional tensor with elements selected by mask."))

(defgeneric $repeat (tensor &rest sizes)
  (:documentation "Returns a new tensor with repeated tensors of grid defined by sizes."))

(defgeneric $squeeze (tensor &optional dimension)
  (:documentation "Returns a new tensor with singleton dimension removed."))
(defgeneric $squeeze! (tensor &optional dimension)
  (:documentation "Returns a tensor with singleton dimension removed."))
(defgeneric $unsqueeze (tensor dimension)
  (:documentation "Returns a new tensor with singleton dimension."))
(defgeneric $unsqueeze! (tensor dimension)
  (:documentation "Returns a tensor with singleton dimension."))

(defgeneric $permute (tensor &rest dimensions)
  (:documentation "Returns a new tensor where the dimensions are permuted as specified."))

(defgeneric $split (tensor size &optional dimension)
  (:documentation "Splits tensor by size along dimension."))
(defgeneric $chunk (tensor n &optional dimension)
  (:documentation "Splits tensor by n approximately equal partitions along dimension."))

(defgeneric $cat (dimension tensor &rest tensors)
  (:documentation "Returns a new tensor which is a concatenation of tensors along dimension."))

(defgeneric $reshape (tensor &rest sizes)
  (:documentation "Returns a new tensor of sizes shape, elements copied from tensor."))
(defgeneric $reshape! (tensor &rest sizes)
  (:documentation "Returns a tensor of sizes shape, elements copied from tensor."))

(defgeneric $diag (tensor &optional k)
  (:documentation "Returns a new diagonal matrix from tensor."))
(defgeneric $diag! (tensor &optional k)
  (:documentation "Returns a diagonal matrix from tensor."))

(defgeneric $eye (tensor m &optional n)
  (:documentation "Returns a new identity matrix of size m by n"))
(defgeneric $eye! (tensor m &optional n)
  (:documentation "Returns a identity matrix of size m by n"))

(defgeneric $tril (tensor &optional k)
  (:documentation "Returns a new tensor with lower triangular part."))
(defgeneric $tril! (tensor &optional k)
  (:documentation "Returns a tensor with lower triangular part."))

(defgeneric $triu (tensor &optional k)
  (:documentation "Returns a new tensor with upper triangular part."))
(defgeneric $triu! (tensor &optional k)
  (:documentation "Returns a tensor with upper triangular part."))

(defgeneric $compare (spec a b)
  (:documentation "Returns a byte tensor as boolean for given comparison spec."))
(defgeneric $lt (a b)
  (:documentation "Returns a byte tensor as boolean for elementwise a < b."))
(defgeneric $le (a b)
  (:documentation "Returns a byte tensor as boolean for elementwise a <= b."))
(defgeneric $gt (a b)
  (:documentation "Returns a byte tensor as boolean for elementwise a > b."))
(defgeneric $ge (a b)
  (:documentation "Returns a byte tensor as boolean for elementwise a >= b."))
(defgeneric $eq (a b)
  (:documentation "Returns a byte tensor as boolean for elementwise a = b."))
(defgeneric $ne (a b)
  (:documentation "Returns a byte tensor as boolean for elementwise a ~= b."))

(defgeneric $nonzero (tensor)
  (:documentation "Returns a long tensor which contains indices of nonzero elements."))

(defgeneric $fmap (fn tensor &rest tensors)
  (:documentation "Applies fn elementwise where only non-nil result will be updated, new tensor."))
(defgeneric $fmap! (fn tensor &rest tensors)
  (:documentation "Applies fn elementwise where only non-nil result will be updated on 1st tensor"))

(defgeneric $abs (x) (:documentation "Returns abs of elements of the given tensor."))
(defgeneric $abs! (x) (:documentation "Returns an in-place replaced abs of elements."))

(defgeneric $sign (x) (:documentation "Returns a signum of the elements of given tensor."))
(defgeneric $sign! (x) (:documentation "Returns an in-place replaced signums."))

(defgeneric $acos (x) (:documentation "Returns acos of the elements"))
(defgeneric $acos! (x) (:documentation "Returns an in-place replaced acos values."))

(defgeneric $asin (x) (:documentation "Returns asin of the elements."))
(defgeneric $asin! (x) (:documentation "Returns an in-place replaced asin values."))

(defgeneric $atan (y) (:documentation "Returns atan of the elements."))
(defgeneric $atan! (y) (:documentation "Returns an in-place replaced atan values."))

(defgeneric $atan2 (y x) (:documentation "Returns atan2 of the elements"))
(defgeneric $atan2! (y x) (:documentation "Returns an in-place replaces atan2 values at y."))

(defgeneric $ceil (x) (:documentation "Returns ceil of the elements."))
(defgeneric $ceil! (x) (:documentation "Returns an in-place replaced ceil values."))

(defgeneric $cos (x) (:documentation "Returns cos of the elements."))
(defgeneric $cos! (x) (:documentation "Returns an in-place replaced cos values."))

(defgeneric $cosh (x) (:documentation "Returns cosh of the elements."))
(defgeneric $cosh! (x) (:documentation "Returns an in-place replaced cosh values."))

(defgeneric $exp (x) (:documentation "Returns exp of the elements."))
(defgeneric $exp! (x) (:documentation "Returns an in-place replaced exp values."))

(defgeneric $floor (x) (:documentation "Returns floor of the elements."))
(defgeneric $floor! (x) (:documentation "Returns an in-place replaced floor values."))

(defgeneric $log (x) (:documentation "Returns log of the elements."))
(defgeneric $log! (x) (:documentation "Return an in-place replaced log values."))

(defgeneric $log1p (x) (:documentation "Returns log(1+x) of the elements."))
(defgeneric $log1p! (x) (:documentation "Returns an in-place replaced log(1+x) values."))

(defgeneric $neg (x) (:documentation "Returns negation of the elements."))
(defgeneric $neg! (x) (:documentation "Returns an in-place replaced negated values."))

(defgeneric $cinv (x) (:documentation "Returns 1/x of the elements."))
(defgeneric $cinv! (x) (:documentation "Returns an in-place replaced 1/x values."))

(defgeneric $expt (x n) (:documentation "Returns expt/pow of the elements."))
(defgeneric $expt! (x n) (:documentation "Returns an in-place replaced expt/pow values."))

(defgeneric $round (x) (:documentation "Returns rounded value of the elements."))
(defgeneric $round! (x) (:documentation "Returns an in-place replaced rounded values."))

(defgeneric $sin (x) (:documentation "Returns sin of the elements."))
(defgeneric $sin! (x) (:documentation "Returns an in-place replaced sin values."))

(defgeneric $sinh (x) (:documentation "Returns sinh of the elements."))
(defgeneric $sinh! (x) (:documentation "Returns an in-place replaced sinh values."))

(defgeneric $sqrt (x) (:documentation "Returns sqrt of the elements."))
(defgeneric $sqrt! (x) (:documentation "Returns an in-place replaced sqrt values."))

(defgeneric $rsqrt (x) (:documentation "Returns 1/sqrt(x) of the elements."))
(defgeneric $rsqrt! (x) (:documentation "Returns an in-place replaces 1/sqrt(x) values."))

(defgeneric $tan (x) (:documentation "Returns tan of the elements."))
(defgeneric $tan! (x) (:documentation "Returns an in-place replaced tan values."))

(defgeneric $tanh (x) (:documentation "Returns tanh of the elements."))
(defgeneric $tanh! (x) (:documentation "Returns an in-place replaced tanh values."))

(defgeneric $sigmoid (x) (:documentation "Returns sigmoid of the elements."))
(defgeneric $sigmoid! (x) (:documentation "Returns an in-place replaced sigmoid values."))

(defgeneric $trunc (x) (:documentation "Returns the integral part of the elements."))
(defgeneric $trunc! (x) (:documentation "Returns an in-place replaced integral values"))

(defgeneric $frac (x) (:documentation "Returns the fractional part of the elements."))
(defgeneric $frac! (x) (:documentation "Returns an in-place replaced fractional values."))

(defgeneric $equal (x y) (:documentation "Compares the elements of x and y"))

(defgeneric $clamp (x min max) (:documentation "Returns a new tensor clamped between min and max."))
(defgeneric $clamp! (x min max) (:documentation "Clamps tensor elements between min and max."))

(defgeneric $fmod (x value) (:documentation "Returns fmod of elements."))
(defgeneric $fmod! (x value) (:documentation "Returns in-place replaced fmod values."))

(defgeneric $rem (x value) (:documentation "Returns remainder of elements."))
(defgeneric $rem! (x value) (:documentation "Returns in-place replaces remainder values."))

(defgeneric $axpy (α x y) (:documentation "Returns a new tensor of αx + y."))
(defgeneric $axpy! (α x y) (:documentation "Returns y = αx + y."))

(defgeneric $gemv (α m x β y) (:documentation "Returns α(mx) + βy"))
(defgeneric $gemv! (α m x β y) (:documentation "Returns y = α(mx) + βy"))

(defgeneric $ger (α x y m) (:documentation "Returns α(x@y') + m"))
(defgeneric $ger! (α x y m) (:documentation "Returns m = α(x@y') + m"))

(defgeneric $gemm (α x y β z) (:documentation "Returns α(x@y) + βz"))
(defgeneric $gemm! (α x y β z) (:documentation "Returns z = α(x@y) + βz"))

(defgeneric $add (y x) (:documentation "Returns a new tensor of y + x."))
(defgeneric $add! (y x) (:documentation "Returns y = y + x."))

(defgeneric $sub (y x) (:documentation "Returns a new tensor of y - x."))
(defgeneric $sub! (y x) (:documentation "Returns y = y - x."))

(defgeneric $mul (y x) (:documentation "Returns a new tensor of y * x."))
(defgeneric $mul (y x) (:documentation "Returns y = y * x."))

(defgeneric $div (y x) (:documentation "Returns a new tensor of y / x."))
(defgeneric $div! (y x) (:documentation "Returns y = y / x."))

(defgeneric $dot (x y) (:documentation "Returns x @ y, both as vectors."))

(defgeneric $addmul (z x y &optional α) (:documentation "Returns z + α(x*y)"))
(defgeneric $addmul! (z x y &optional α) (:documentation "Returns z = z + α(x*y)"))

(defgeneric $adddiv (z x y &optional α) (:documentation "Returns z + α(x/y)"))
(defgeneric $adddiv! (z x y &optional α) (:documentation "Returns z = z + α(x/y)"))

(defgeneric $addmv (x m v &optional α β) (:documentation "Returns βx + α(mv)"))
(defgeneric $addmv! (x m v &optional α β) (:documentation "Returns x = βx + α(mv)"))

(defgeneric $addr (m x y &optional α β) (:documentation "Returns βm + α(x@y')"))
(defgeneric $addr! (m x y &optional α β) (:documentation "Returns m = βm + α(x@y')"))

(defgeneric $addmm (z x y &optional α β) (:documentation "Returns α(x@y) + βz"))
(defgeneric $addmm! (z x y &optional α β) (:documentation "Returns z = α(x@y) + βz"))

(defgeneric $addbmm (z bx by &optional α β) (:documentation "Returns α(Σ(x@y)) + βz"))
(defgeneric $addbmm! (z bx by &optional α β) (:documentation "Returns z = α(Σ(x@y)) + βz"))

(defgeneric $baddbmm (bz bx by &optional α β) (:documentation "Returns batched α(x@y) + βz"))
(defgeneric $baddbmm! (bz bx by &optional α β) (:documentation "Returns batched z = α(x@y) + βz"))

(defgeneric $vv (x y) (:documentation "Returns x@y'."))
(defgeneric $vv! (m x y) (:documentation "Returns m = x@y'."))

(defgeneric $mv (m v) (:documentation "Returns m@v."))
(defgeneric $mv! (x m v) (:documentation "Returns x = m@v."))

(defgeneric $mm (x y) (:documentation "Returns x@y."))
(defgeneric $mm! (z x y) (:documentation "Returns z = x@y."))

(defgeneric $bmm (bx by) (:documentation "Returns batched (x@y)."))
(defgeneric $bmm! (bz bx by) (:documentation "Returns batched z = (x@y)."))

(defgeneric $xx (x y &optional dimension) (:documentation "Returns x <cross> y."))
(defgeneric $xx! (z x y &optional dimension) (:documentation "Returns z = x <cross> y."))

(defgeneric $cumprd (x &optional dimension) (:documentation "Returns cumulative products."))
(defgeneric $cumprd! (y x &optional dimension) (:documentation "Returns y = cumprd x."))

(defgeneric $cumsum (x &optional dimension) (:documentation "Returns cumulative sum."))
(defgeneric $cumsum! (y x &optional dimension) (:documentation "Returns y = cumsum x."))

(defgeneric $max (x &optional dimension) (:documentation "Returns (max-vals, max-indices)."))
(defgeneric $max! (vals indices x &optional dimension)
  (:documentation "Returns (max-vals, max-indices)."))

(defgeneric $min (x &optional dimension) (:documentation "Returns (min-vals, min-indices)."))
(defgeneric $min! (vals indices x &optional dimension)
  (:documentation "Returns (min-vals, min-indices)."))

(defgeneric $mean (x &optional dimension) (:documentation "Returns mean values."))
(defgeneric $mean! (m x &optional dimension) (:documentation "Returns m = mean x."))

(defgeneric $cmax (tensor &rest tensors) (:documentation "Elemnentwise maximum."))
(defgeneric $cmax! (tensor &rest tensors) (:documentation "Elemnentwise maximum."))

(defgeneric $cmin (tensor &rest tensors) (:documentation "Elemnentwise minimum."))
(defgeneric $cmin! (tensor &rest tensors) (:documentation "Elemnentwise minimum."))

(defgeneric $median (x &optional dimension)
  (:documentation "Returns (median-vals, median-indices)."))
(defgeneric $median! (vals indices x &optional dimension)
  (:documentation "Returns (median-vals, median-indices)."))

(defgeneric $mode (x &optional dimension)
  (:documentation "Returns (mode-vals, mode-indices)."))
(defgeneric $mode! (vals indices x &optional dimension)
  (:documentation "Returns (mode-vals, mode-indices)."))

(defgeneric $kth (x k &optional dimension) (:documentation "Returns kth smallest (value, index)."))
(defgeneric $kth! (vals indices x k &optional dimension)
  (:documentation "Returns kth smallest (value, index)."))

(defgeneric $topk (x k &optional dimension descendingp sortp)
  (:documentation "Returns top k smallest (values, indices) in unsorted manner."))
(defgeneric $topk! (vals indices x k &optional dimension descendingp sortp)
  (:documentation "Returns top k smallest (values, indices) in unsorted manner."))

(defgeneric $sort (x &optional dimension descendingp)
  (:documentation "Returns sorted (values, indices) along dimension."))
(defgeneric $sort! (vals indices x &optional dimension descendingp)
  (:documentation "Returns sorted (values, indices) along dimension."))

(defgeneric $prd (x &optional dimension)
  (:documentation "Returns products along dimension."))
(defgeneric $prd! (z x &optional dimension)
  (:documentation "Returns z = products along dimension of x."))

(defgeneric $sum (x &optional dimension)
  (:documentation "Returns sums along dimension."))
(defgeneric $sum! (z x &optional dimension)
  (:documentation "Returns z = sums along dimension of x."))

(defgeneric $sd (x &optional dimension biased)
  (:documentation "Returns standard deviation along dimension."))
(defgeneric $sd! (z x &optional dimension biased)
  (:documentation "Returns z = sd(x) along dimension."))

(defgeneric $var (x &optional dimension biased)
  (:documentation "Returns variance along dimension."))
(defgeneric $var! (z x &optional dimension biased)
  (:documentation "Returns z = var(x) along dimension."))

(defgeneric $norm (x &optional p dimension)
  (:documentation "Returns p-norm."))
(defgeneric $norm! (z x &optional p dimension)
  (:documentation "Returns z = p-norm of x."))

(defgeneric $renorm (x p dimension max)
  (:documentation "Renormalizes subtensors along dimension so that they do not exceed norm max."))
(defgeneric $renorm! (z x p dimension max)
  (:documentation "Renormalizes subtensors along dimension so that they do not exceed norm max."))

(defgeneric $dist (x y &optional p)
  (:documentation "Returns p-norm of x - y."))

(defgeneric $trace (x) (:documentation "Returns the trace of x."))

(defgeneric $conv2 (x k &optional type)
  (:documentation "Computes 2D convolution between x and k. By default valid."))
(defgeneric $conv2! (r x k &optional type)
  (:documentation "Computes r = 2D convolution between x and k. By default valid."))

(defgeneric $xcore2 (x k &optional type)
  (:documentation "Computes 2D cross-correlation between x and k. By default valid."))
(defgeneric $xcorr2! (r x k &optional type)
  (:documentation "Computes r = 2D cross-correlation between x and k. By default valid."))

(defgeneric $conv3 (x k &optional type)
  (:documentation "Computes 3D convolution between x and k. By default valid."))
(defgeneric $conv3! (r x k &optional type)
  (:documentation "Computes r = 3D convolution between x and k. By default valid."))

(defgeneric $xcore3 (x k &optional type)
  (:documentation "Computes 3D cross-correlation between x and k. By default valid."))
(defgeneric $xcorr3! (r x k &optional type)
  (:documentation "Computes r = 3D cross-correlation between x and k. By default valid."))

(defgeneric $gesv (b a) (:documentation "Returns (x, lu) for ax = b."))
(defgeneric $gesv! (x lu b a) (:documentation "Returns (x, lu) for ax = b."))

(defgeneric $trtrs (b a &optional up trans unit-diag)
  (:documentation "Returns x for ax = b, a is triangular."))
(defgeneric $trtrs! (x b a &optional up trans unit-diag)
  (:documentation "Returns x for ax = b, a is triangular."))

(defgeneric $potrf (a &optional up)
  (:documentation "Returns cholesky decomposition of a."))
(defgeneric $potrf! (ch a &optional up)
  (:documentation "Returns cholesky decomposition ch of a."))

(defgeneric $pstrf (a &optional up)
  (:documentation "Returns cholesky decomposition with pivot(ch, piv) of a."))
(defgeneric $pstrf! (ch piv a &optional up)
  (:documentation "Returns cholesky decomposition with pivot(ch, piv) of a."))

(defgeneric $potrs (b ch &optional up) (:documentation "Returns x for ax = b, a = ch."))
(defgeneric $potrs! (x b ch &optional up) (:documentation "Returns x for ax = b, a = ch."))

(defgeneric $potri (ch &optional up) (:documentation "Returns inverse of ch."))
(defgeneric $potri! (inv ch &optional up) (:documentation "Returns inverse of ch."))

(defgeneric $gels (b a) (:documentation "Returns solution of least square of a."))
(defgeneric $gels! (x b a) (:documentation "Returns solution of least square of a."))

(defgeneric $syev (a &optional all up)
  (:documentation "Returns (eigenvalues, eigenvectors) of symmetric matrix a."))
(defgeneric $syev! (e v a &optional all up)
  (:documentation "Returns (eigenvalues, eigenvectors) of symmetric matrix a."))

(defgeneric $ev (a &optional all)
  (:documentation "Returns (eigenvalues, eigenvectors) of matrix a."))
(defgeneric $ev! (e v a &optional all)
  (:documentation "Returns (eigenvalues, eigenvectors) of matrix a."))

(defgeneric $svd (a &optional all) (:documentation "Returns (u, s, v) of a."))
(defgeneric $svd! (u s v a &optional all) (:documentation "Returns (u, s, v) of a."))

(defgeneric $inverse (a) (:documentation "Returns inverse of a."))
(defgeneric $inverse! (r a) (:documentation "Returns r = inverse of a."))

(defgeneric $qr (x) (:documentation "Returns qr decomposition (q, r) of a."))
(defgeneric $qr! (q r x) (:documentation "Returns qr decomposition (q, r) of a."))

(defgeneric $fopenedp (file) (:documentation "Check file is opened."))
(defgeneric $fquietp (file) (:documentation "Check file is in quiet mode."))
(defgeneric $fpedanticp (file) (:documentation "Check file is in pedantic mode."))
(defgeneric $freadablep (file) (:documentation "Check file is readable."))
(defgeneric $fwritablep (file) (:documentation "Check file is writable."))
(defgeneric $fbinaryp (file) (:documentation "Check file is binary mode."))
(defgeneric $fasciip (file) (:documentation "Check file is ascii mode."))
(defgeneric $fautospacingp (file) (:documentation "Check file is in auto spacing mode."))
(defgeneric $fnoautospacingp (file) (:documentation "Check file is not in auto spacing mode."))
(defgeneric $ferrorp (file) (:documentation "Check file has errors."))

(defgeneric $freadbyte (file) (:documentation "Reads a scalar byte."))
(defgeneric $freadchar (file) (:documentation "Reads a scalar char."))
(defgeneric $freadshort (file) (:documentation "Reads a scalar short."))
(defgeneric $freadint (file) (:documentation "Reads a scalar int."))
(defgeneric $freadlong (file) (:documentation "Reads a scalar long."))
(defgeneric $freadfloat (file) (:documentation "Reads a scalar float."))
(defgeneric $freaddouble (file) (:documentation "Reads a scalar double."))

(defgeneric $freadbyte! (storage file) (:documentation "Reads a byte storage."))
(defgeneric $freadchar! (storage file) (:documentation "Reads a char storage."))
(defgeneric $freadshort! (storage file) (:documentation "Reads a short storage."))
(defgeneric $freadint! (storage file) (:documentation "Reads a int storage."))
(defgeneric $freadlong! (storage file) (:documentation "Reads a long storage."))
(defgeneric $freadfloat! (storage file) (:documentation "Reads a float storage."))
(defgeneric $freaddouble! (storage file) (:documentation "Reads a double storage."))

(defgeneric $fwritebyte (byte file) (:documentation "Writes a scalar byte."))
(defgeneric $fwritechar (char file) (:documentation "Writes a scalar char."))
(defgeneric $fwriteshort (short file) (:documentation "Writes a scalar short."))
(defgeneric $fwriteint (int file) (:documentation "Writes a scalar int."))
(defgeneric $fwritelong (long file) (:documentation "Writes a scalar long."))
(defgeneric $fwritefloat (float file) (:documentation "Writes a scalar float."))
(defgeneric $fwritedouble (double file) (:documentation "Writes a scalar double."))

(defgeneric $fwrite (storage file) (:documentation "Writes a storage object."))

(defgeneric $fsync (file) (:documentation "Synchronizes file."))
(defgeneric $fseek (file position) (:documentation "Repositions file to position."))
(defgeneric $fseekend (file) (:documentation "Repositions to file end."))
(defgeneric $ftell (file) (:documentation "Returns current position of file."))
(defgeneric $fclose (file) (:documentation "Closes file."))

(defgeneric $fname (file) (:documentation "Returns file name."))
