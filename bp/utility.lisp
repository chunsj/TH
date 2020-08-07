(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defgeneric $xwpb (x w b) (:documentation "Returns x@w + b."))
(defgeneric $affine (x w b) (:documentation "Affine transformation."))
(defgeneric $affine2 (x1 w1 x2 w2 b) (:documentation "Affine transformation."))
(defgeneric $addm2 (x1 w1 x2 w2) (:documentation "x1*w1 + x2*w2"))

(defgeneric $choice (elements probabilities) (:documentation "Random sampling with probabilities."))
(defgeneric $wimb (xwi w) (:documentation "Computes word embedding."))
(defgeneric $wemb (xoh w) (:documentation "Computes word embedding with one hot encoding."))
(defgeneric $embedding (indices weight) (:documentation "Embedding operation using indices."))
(defgeneric $emb (xi wx b) (:documentation "Embedded affine using indices."))

(defgeneric $rn! (tensor &optional µ σ) (:documentation "Fills with random normal."))
(defgeneric $rnt! (tensor &optional µ σ) (:documentation "Fills with truncated random normal."))
(defgeneric $ru! (tensor &optional min max) (:documentation "Fills with random uniform."))
(defgeneric $xavieru! (tensor) (:documentation "Fills with Xavier uniform."))
(defgeneric $xaviern! (tensor) (:documentation "Fills with Xavier normal."))
(defgeneric $heu! (tensor) (:documentation "Fills with He uniform."))
(defgeneric $hen! (tensor) (:documentation "Fills with He normal."))
(defgeneric $lecunu! (tensor) (:documentation "Fills with Lecun uniform."))
(defgeneric $lecunn! (tensor) (:documentation "Fills with Lecun normal."))

(defgeneric $array (collection) (:documentation "Converts collection into array."))

(defgeneric $argmax (x &optional dim) (:documentation "Argument Max."))
(defgeneric $argmin (x &optional dim) (:documentation "Argument Max."))

(defgeneric $scalar (x) (:documentation "Convert to the scalar value if possible."))

(defgeneric $diagflat (x) (:documentation "Diagonal matrix with flattening."))

(defun addmul (x1 w1 x2 w2) ($addmul! ($mul x1 w1) x2 w2))

(defmethod $addm2 ((x1 node) (w1 node) (x2 node) (w2 node))
  (node (addmul ($data x1) ($data w1) ($data x2) ($data w2))
        :name :addm2
        :link (link
                (to x1 ($mul ($data w1) gv))
                (to w1 ($mul ($data x1) gv))
                (to x2 ($mul ($data w2) gv))
                (to w2 ($mul ($data x2) gv)))))

(defmethod $addm2 ((x1 tensor) (w1 node) (x2 tensor) (w2 node))
  (node (addmul x1 ($data w1) x2 ($data w2))
        :name :addm2
        :link (link
                (to w1 ($mul x1 gv))
                (to w2 ($mul x2 gv)))))

(defmethod $addm2 ((x1 node) (w1 tensor) (x2 node) (w2 tensor))
  (node (addmul ($data x1) w1 ($data x2) w2)
        :name :addm2
        :link (link
                (to x1 ($mul w1 gv))
                (to x2 ($mul w2 gv)))))

(defmethod $addm2 ((x1 tensor) (w1 tensor) (x2 node) (w2 node))
  (node (addmul x1 w1 ($data x2) ($data w2))
        :name :addm2
        :link (link
                (to x2 ($mul ($data w2) gv))
                (to w2 ($mul ($data x2) gv)))))

(defmethod $addm2 ((x1 node) (w1 node) (x2 tensor) (w2 tensor))
  (node (addmul ($data x1) ($data w1) x2 w2)
        :name :addm2
        :link (link
                (to x1 ($mul ($data w1) gv))
                (to w1 ($mul ($data x1) gv)))))

(defmethod $addm2 ((x1 node) (w1 node) (x2 node) (w2 tensor))
  (node (addmul ($data x1) ($data w1) ($data x2) w2)
        :name :addm2
        :link (link
                (to x1 ($mul ($data w1) gv))
                (to w1 ($mul ($data x1) gv))
                (to x2 ($mul w2 gv)))))

(defmethod $addm2 ((x1 tensor) (w1 tensor) (x2 tensor) (w2 tensor)) (addmul x1 w1 x2 w2))

(defun affine-without-bias (x w)
  (let ((dim ($ndim x)))
    (cond ((eq dim 1) (let ((output ($zero! ($resize! ($empty x) (list ($size w 1)))))
                            (tw ($transpose w)))
                        ($addmv! output tw x 1 1)
                        output))
          ((eq dim 2) (let* ((nframe ($size x 0))
                             (output ($zero! ($resize! ($empty x)
                                                       (list nframe ($size w 1))))))
                        ($addmm! output x w 1 0)
                        output)))))

(defun affine2-without-bias (x1 w1 x2 w2)
  (let ((dim1 ($ndim x1))
        (dim2 ($ndim x2)))
    (when (eq dim1 dim2)
      (let ((nf1 ($size x1 0))
            (nf2 ($size x2 0)))
        (when (eq nf1 nf2)
          (cond ((eq dim1 1)
                 (let ((output ($zero! ($resize! ($empty x1) (list ($size w1 1)))))
                       (tw1 ($transpose w1))
                       (tw2 ($transpose w2)))
                   ($addmv! output tw1 x1 1 1)
                   ($addmv! output tw2 x2 1 1)
                   output))
                ((eq dim1 2)
                 (let ((output ($zero! ($resize! ($empty x1) (list nf1 ($size w1 1))))))
                   ($addmm! output x1 w1 1 0)
                   ($addmm! output x2 w2 1 1)
                   output))))))))

(defun affine-with-bias (x w b)
  (let ((dim ($ndim x)))
    (cond ((eq dim 1) (let ((output ($copy! ($resize! ($empty x) (list ($size w 1))) b))
                            (tw ($transpose w)))
                        ($addmv! output tw x 1 1)
                        output))
          ((eq dim 2) (let* ((nframe ($size x 0))
                             (output ($zero! ($resize! ($empty x) (list nframe ($size w 1)))))
                             (addbuf (allocate-addbuf nframe)))
                        ($addmm! output x w 1 0)
                        ($addr! output addbuf b 1 1)
                        output)))))

(defun affine2-with-bias (x1 w1 x2 w2 b)
  (let ((dim1 ($ndim x1))
        (dim2 ($ndim x2)))
    (when (eq dim1 dim2)
      (let ((nf1 ($size x1 0))
            (nf2 ($size x2 0)))
        (when (eq nf1 nf2)
          (cond ((eq dim1 1)
                 (let ((output ($copy! ($resize! ($empty x1) (list ($size w1 1))) b))
                       (tw1 ($transpose w1))
                       (tw2 ($transpose w2)))
                   ($addmv! output tw1 x1 1 1)
                   ($addmv! output tw2 x2 1 1)
                   output))
                ((eq dim1 2)
                 (let* ((output ($zero! ($resize! ($empty x1) (list nf1 ($size w1 1)))))
                        (addbuf (allocate-addbuf nf1)))
                   ($addmm! output x1 w1 1 0)
                   ($addmm! output x2 w2 1 1)
                   ($addr! output addbuf b 1 1)
                   output))))))))

(defun daffine-output (x w gv)
  (let ((dx ($zero x))
        (dim ($ndim x)))
    (cond ((eq dim 1) (progn
                        ($addmv! dx w gv 1 0)
                        dx))
          ((eq dim 2) (let ((tw ($transpose w)))
                        ($addmm! dx gv tw 1 0)
                        dx)))))

(defun daffine-weight (x w gv)
  (let ((dw ($zero w))
        (dim ($ndim x)))
    (cond ((eq dim 1) (progn
                        ($addr! dw x gv 1 1)
                        dw))
          ((eq dim 2) (let ((tx ($transpose x)))
                        ($addmm! dw tx gv 1 1)
                        dw)))))

(defun daffine-bias (x b gv)
  (let ((db ($zero b))
        (dim ($ndim x)))
    (cond ((eq dim 1) (progn
                        (tensor-cadd db db 1 gv)
                        db))
          ((eq dim 2) (let* ((nframe ($size x 0))
                             (tgv ($transpose gv))
                             (addbuf (allocate-addbuf nframe)))
                        ($addmv! db tgv addbuf 1 1)
                        db)))))

(defmethod $xwpb ((x tensor) (w tensor) (b tensor))
  (cond ((null b) (affine-without-bias x w))
        (t (affine-with-bias x w b))))

(defmethod $xwpb ((x node) (w node) (b node))
  (node ($xwpb ($data x) ($data w) ($data b))
        :name :xwpb
        :link (link
                (to x (daffine-output ($data x) ($data w) gv))
                (to w (daffine-weight ($data x) ($data w) gv))
                (when b (to b (daffine-bias ($data x) ($data b) gv))))))

(defmethod $xwpb ((x tensor) (w node) (b node))
  (node ($xwpb x ($data w) ($data b))
        :name :xwpb
        :link (link
                (to w (daffine-weight x ($data w) gv))
                (when b (to b (daffine-bias x ($data b) gv))))))

(defmethod $xwpb ((x tensor) (w tensor) (b node))
  (node ($xwpb x w ($data b))
        :name :xwpb
        :link (link
                (when b (to b (daffine-bias x ($data b) gv))))))

(defmethod $affine ((x tensor) (w tensor) (b tensor))
  (affine-with-bias x w b))

(defmethod $affine ((x tensor) (w tensor) (b null))
  (affine-without-bias x w))

(defmethod $affine ((x node) (w node) (b node))
  (node ($affine ($data x) ($data w) ($data b))
        :name :affine
        :link (link
                (to x (daffine-output ($data x) ($data w) gv))
                (to w (daffine-weight ($data x) ($data w) gv))
                (to b (daffine-bias ($data x) ($data b) gv)))))

(defmethod $affine ((x node) (w node) (b null))
  (node ($affine ($data x) ($data w) b)
        :name :affine
        :link (link
                (to x (daffine-output ($data x) ($data w) gv))
                (to w (daffine-weight ($data x) ($data w) gv)))))

(defmethod $affine ((x tensor) (w node) (b node))
  (node ($affine x ($data w) ($data b))
        :name :affine
        :link (link
                (to w (daffine-weight x ($data w) gv))
                (to b (daffine-bias x ($data b) gv)))))

(defmethod $affine ((x tensor) (w node) (b null))
  (node ($affine x ($data w) b)
        :name :affine
        :link (link
                (to w (daffine-weight x ($data w) gv)))))

(defmethod $affine ((x tensor) (w tensor) (b node))
  (node ($affine x w ($data b))
        :name :affine
        :link (link (to b (daffine-bias x ($data b) gv)))))

(defmethod $affine2 ((x1 tensor) (w1 tensor) (x2 tensor) (w2 tensor) (b tensor))
  (affine2-with-bias x1 w1 x2 w2 b))

(defmethod $affine2 ((x1 tensor) (w1 tensor) (x2 tensor) (w2 tensor) (b null))
  (affine2-without-bias x1 w1 x2 w2))

(defmethod $affine2 ((x1 node) (w1 node) (x2 node) (w2 node) (b node))
  (node ($affine2 ($data x1) ($data w1) ($data x2) ($data w2) ($data b))
        :name :affine2
        :link (link
                (to x1 (daffine-output ($data x1) ($data w1) gv))
                (to w1 (daffine-weight ($data x1) ($data w1) gv))
                (to x2 (daffine-output ($data x2) ($data w2) gv))
                (to w2 (daffine-weight ($data x2) ($data w2) gv))
                (to b (daffine-bias ($data x1) ($data b) gv)))))

(defmethod $affine2 ((x1 node) (w1 node) (x2 node) (w2 node) (b null))
  (node ($affine2 ($data x1) ($data w1) ($data x2) ($data w2) b)
        :name :affine2
        :link (link
                (to x1 (daffine-output ($data x1) ($data w1) gv))
                (to w1 (daffine-weight ($data x1) ($data w1) gv))
                (to x2 (daffine-output ($data x2) ($data w2) gv))
                (to w2 (daffine-weight ($data x2) ($data w2) gv)))))

(defmethod $affine2 ((x1 tensor) (w1 node) (x2 node) (w2 node) (b node))
  (node ($affine2 x1 ($data w1) ($data x2) ($data w2) ($data b))
        :name :affine2
        :link (link
                (to w1 (daffine-weight x1 ($data w1) gv))
                (to x2 (daffine-output ($data x2) ($data w2) gv))
                (to w2 (daffine-weight ($data x2) ($data w2) gv))
                (to b (daffine-bias x1 ($data b) gv)))))

(defmethod $affine2 ((x1 tensor) (w1 node) (x2 node) (w2 node) (b null))
  (node ($affine2 x1 ($data w1) ($data x2) ($data w2) b)
        :name :affine2
        :link (link
                (to w1 (daffine-weight x1 ($data w1) gv))
                (to x2 (daffine-output ($data x2) ($data w2) gv))
                (to w2 (daffine-weight ($data x2) ($data w2) gv)))))

(defmethod $affine2 ((x1 node) (w1 node) (x2 tensor) (w2 node) (b node))
  (node ($affine2 ($data x1) ($data w1) x2 ($data w2) ($data b))
        :name :affine2
        :link (link
                (to x1 (daffine-output ($data x1) ($data w1) gv))
                (to w1 (daffine-weight ($data x1) ($data w1) gv))
                (to w2 (daffine-weight x2 ($data w2) gv))
                (to b (daffine-bias ($data x1) ($data b) gv)))))

(defmethod $affine2 ((x1 node) (w1 node) (x2 tensor) (w2 node) (b null))
  (node ($affine2 ($data x1) ($data w1) x2 ($data w2) b)
        :name :affine2
        :link (link
                (to x1 (daffine-output ($data x1) ($data w1) gv))
                (to w1 (daffine-weight ($data x1) ($data w1) gv))
                (to w2 (daffine-weight x2 ($data w2) gv)))))

(defmethod $affine2 ((x1 tensor) (w1 node) (x2 tensor) (w2 node) (b node))
  (node ($affine2 x1 ($data w1) x2 ($data w2) ($data b))
        :name :affine2
        :link (link
                (to w1 (daffine-weight x1 ($data w1) gv))
                (to w2 (daffine-weight x2 ($data w2) gv))
                (to b (daffine-bias x1 ($data b) gv)))))

(defmethod $affine2 ((x1 tensor) (w1 node) (x2 tensor) (w2 node) (b null))
  (node ($affine2 x1 ($data w1) x2 ($data w2) b)
        :name :affine2
        :link (link
                (to w1 (daffine-weight x1 ($data w1) gv))
                (to w2 (daffine-weight x2 ($data w2) gv)))))

(defmethod $affine2 ((x1 tensor) (w1 tensor) (x2 node) (w2 node) (b node))
  (node ($affine2 x1 w1 ($data x2) ($data w2) ($data b))
        :name :affine2
        :link (link
                (to x2 (daffine-output ($data x2) ($data w2) gv))
                (to w2 (daffine-weight ($data x2) ($data w2) gv))
                (to b (daffine-bias x1 ($data b) gv)))))

(defmethod $affine2 ((x1 tensor) (w1 tensor) (x2 node) (w2 node) (b null))
  (node ($affine2 x1 w1 ($data x2) ($data w2) b)
        :name :affine2
        :link (link
                (to x2 (daffine-output ($data x2) ($data w2) gv))
                (to w2 (daffine-weight ($data x2) ($data w2) gv)))))

(defmethod $affine2 ((x1 node) (w1 node) (x2 tensor) (w2 tensor) (b node))
  (node ($affine2 ($data x1) ($data w1) x2 w2 ($data b))
        :name :affine2
        :link (link
                (to x1 (daffine-output ($data x1) ($data w1) gv))
                (to w1 (daffine-weight ($data x1) ($data w1) gv))
                (to b (daffine-bias ($data x1) ($data b) gv)))))

(defmethod $affine2 ((x1 tensor) (w1 tensor) (x2 tensor) (w2 tensor) (b node))
  (node ($affine2 x1 w1 x2 w2 ($data b))
        :name :affine2
        :link (link (to b (daffine-bias x1 ($data b) gv)))))

(defmethod $wimb ((xwi list) (w tensor)) ($sum ($index w 0 xwi) 0))
(defmethod $wimb ((xwi tensor.int) (w tensor)) ($sum ($index w 0 xwi) 0))
(defmethod $wimb ((xwi tensor.long) (w tensor)) ($sum ($index w 0 xwi) 0))

(defmethod $wimb ((xwi list) (w node)) ($sum ($index w 0 xwi) 0))
(defmethod $wimb ((xwi tensor.int) (w node)) ($sum ($index w 0 xwi) 0))
(defmethod $wimb ((xwi tensor.long) (w node)) ($sum ($index w 0 xwi) 0))

(defmethod $wemb ((xoh tensor) (w tensor)) ($wimb ($select ($nonzero xoh) 1 1) w))
(defmethod $wemb ((xoh tensor) (w node)) ($wimb ($select ($nonzero xoh) 1 1) w))

(defmethod $embedding ((indices tensor.long) (weight tensor))
  (cond ((eq 1 ($ndim indices)) ($index weight 0 indices))
        (T (let* ((sz ($size indices))
                  (nz ($count indices))
                  (nsz (append sz (list ($size weight 1)))))
             (apply #'$view ($index weight 0 ($reshape indices nz)) nsz)))))

(defmethod $embedding ((indices tensor.int) (weight tensor))
  (cond ((eq 1 ($ndim indices)) ($index weight 0 indices))
        (T (let* ((sz ($size indices))
                  (nz ($count indices))
                  (nsz (append sz (list ($size weight 1)))))
             (apply #'$view ($index weight 0 ($reshape indices nz)) nsz)))))

(defmethod $embedding ((indices list) (weight tensor))
  ($embedding (tensor.long indices) weight))

(defmethod $embedding ((indices tensor.long) (weight node))
  (cond ((eq 1 ($ndim indices)) ($index weight 0 indices))
        (T (let* ((sz ($size indices))
                  (nz ($count indices))
                  (nsz (append sz (list ($size weight 1)))))
             (apply #'$view ($index weight 0 ($reshape indices nz)) nsz)))))

(defmethod $embedding ((indices tensor.int) (weight node))
  (cond ((eq 1 ($ndim indices)) ($index weight 0 indices))
        (T (let* ((sz ($size indices))
                  (nz ($count indices))
                  (nsz (append sz (list ($size weight 1)))))
             (apply #'$view ($index weight 0 ($reshape indices nz)) nsz)))))

(defmethod $embedding ((indices list) (weight node))
  ($embedding (tensor.long indices) weight))

(defmethod $emb ((xi tensor.long) (wx tensor) b)
  (if b
      (let* ((xp ($index wx 0 xi))
             (hp ($vv (allocate-addbuf ($size xp 0)) b)))
        ($+ xp hp))
      ($index wx 0 xi)))

(defmethod $emb ((xi tensor.int) (wx tensor) b)
  (if b
      (let* ((xp ($index wx 0 xi))
             (hp ($vv (allocate-addbuf ($size xp 0)) b)))
        ($+ xp hp))
      ($index wx 0 xi)))

(defmethod $emb ((xi list) (wx tensor) b)
  (if b
      (let* ((xp ($index wx 0 xi))
             (hp ($vv (allocate-addbuf ($size xp 0)) b)))
        ($+ xp hp))
      ($index wx 0 xi)))

(defmethod $emb ((xi tensor.long) (wx node) b)
  (if b
      (let* ((xp ($index wx 0 xi))
             (hp ($vv (allocate-addbuf ($size xp 0)) b)))
        ($+ xp hp))
      ($index wx 0 xi)))

(defmethod $emb ((xi tensor.int) (wx node) b)
  (if b
      (let* ((xp ($index wx 0 xi))
             (hp ($vv (allocate-addbuf ($size xp 0)) b)))
        ($+ xp hp))
      ($index wx 0 xi)))

(defmethod $emb ((xi list) (wx node) b)
  (if b
      (let* ((xp ($index wx 0 xi))
             (hp ($vv (allocate-addbuf ($size xp 0)) b)))
        ($+ xp hp))
      ($index wx 0 xi)))

(defmethod $rn! ((tensor tensor) &optional (mean 0) (sd 0.05))
  ($add! ($mul! (tensor-randn tensor ($size tensor)) sd) mean)
  tensor)

(defmethod $rn! ((node node) &optional (mean 0) (sd 0.05))
  ($rn! ($data node) mean sd)
  node)

(defmethod $ru! ((tensor tensor) &optional (min -0.05) (max 0.05))
  (let ((width (- max min)))
    ($add! ($mul! (tensor-rand tensor ($size tensor)) width) min)
    tensor))

(defmethod $ru! ((node node) &optional (min -0.05) (max 0.05))
  ($ru! ($data node) min max)
  node)

(defmethod $rnt! ((tensor tensor) &optional (mean 0) (sd 0.05))
  (tensor-clamp tensor ($add! ($mul! (tensor-randn tensor ($size tensor)) sd) mean)
                (- mean (* 2 sd)) (+ mean (* 2 sd)))
  tensor)

(defmethod $rnt! ((node node) &optional (mean 0) (sd 0.05))
  ($rnt! ($data node) mean sd)
  node)

(defmethod $xavieru! ((tensor tensor))
  (let* ((in ($size tensor 0))
         (out ($size tensor 1))
         (lmt (sqrt (/ 6 (+ in out)))))
    ($ru! tensor (- lmt) lmt)))

(defmethod $xaviern! ((tensor tensor))
  (let* ((in ($size tensor 0))
         (out ($size tensor 1))
         (sd (sqrt (/ 2 (+ in out)))))
    ($rnt! tensor 0 sd)))

(defmethod $xavieru! ((node node))
  ($xavieru! ($data node))
  node)

(defmethod $variern! ((node node))
  ($xaviern! ($data node))
  node)

(defmethod $heu! ((tensor tensor))
  (let* ((in ($size tensor 0))
         (lmt (sqrt (/ 6 in))))
    ($ru! tensor (- lmt) lmt)))

(defmethod $hen! ((tensor tensor))
  (let* ((in ($size tensor 0))
         (sd (sqrt (/ 2 in))))
    ($rnt! tensor 0 sd)))

(defmethod $heu! ((node node))
  ($heu! ($data node))
  node)

(defmethod $hen! ((node node))
  ($hen! ($data node))
  node)

(defmethod $lecunu! ((tensor tensor))
  (let* ((in ($size tensor 0))
         (lmt (sqrt (/ 3 in))))
    ($ru! tensor (- lmt) lmt)))

(defmethod $lecunn! ((tensor tensor))
  (let* ((in ($size tensor 0))
         (sd (sqrt (/ 1 in))))
    ($rnt! tensor 0 sd)))

(defmethod $lecunu! ((node node))
  ($lecunu! ($data node))
  node)

(defmethod $lecunn! ((node node))
  ($lecunn! ($data node))
  node)

(defun vrn (sizes &optional (µ 0) (σ 0.05))
  (let ((tensor (apply #'tensor sizes)))
    ($rn! tensor µ σ)
    ($parameter tensor)))

(defun vru (sizes &optional (min -0.05) (max 0.05))
  (let ((tensor (apply #'tensor sizes)))
    ($ru! tensor min max)
    ($parameter tensor)))

(defun vrnt (sizes &optional (µ 0) (σ 0.05))
  (let ((tensor (apply #'tensor sizes)))
    ($rnt! tensor µ σ)
    ($parameter tensor)))

(defun vxavier (sizes &optional (dist :normal))
  (let ((tensor (apply #'tensor sizes)))
    (if (eq dist :normal)
        ($xaviern! tensor)
        ($xavieru! tensor))
    ($parameter tensor)))

(defun vhe (sizes &optional (dist :normal))
  (let ((tensor (apply #'tensor sizes)))
    (if (eq dist :normal)
        ($hen! tensor)
        ($heu! tensor))
    ($parameter tensor)))

(defun vlecun (sizes &optional (dist :normal))
  (let ((tensor (apply #'tensor sizes)))
    (if (eq dist :normal)
        ($lecunn! tensor)
        ($lecunu! tensor))
    ($parameter tensor)))

(defun vselu (sizes &optional (dist :normal))
  "https://towardsdatascience.com/selu-make-fnns-great-again-snn-8d61526802a9"
  (let ((tensor (apply #'tensor sizes)))
    (if (eq dist :normal)
        ($parameter ($rn! tensor 0 (/ 1 (sqrt ($size tensor 0)))))
        ($parameter ($ru! tensor (/ -1 (sqrt ($size tensor 0))) (/ 1 (sqrt ($size tensor 0))))))))

(defmethod $choice ((elements tensor) (probabilities tensor))
  (when (and (eq 1 ($ndim probabilities)) (eq ($count elements) ($count probabilities)))
    (let ((choice ($multinomial ($div probabilities ($sum probabilities)) 1)))
      ($ elements ($ ($reshape! choice ($count probabilities)) 0)))))

(defmethod $choice ((elements tensor) (probabilities list))
  ($choice elements (tensor probabilities)))

(defmethod $choice ((elements tensor) (probabilities vector))
  ($choice elements (tensor (coerce probabilities 'list))))

(defmethod $choice ((elements list) (probabilities tensor))
  (when (and (eq 1 ($ndim probabilities)) (eq ($count elements) ($count probabilities)))
    (let ((choice ($multinomial ($div probabilities ($sum probabilities)) 1)))
      ($ elements ($ ($reshape! choice ($count probabilities)) 0)))))

(defmethod $choice ((elements list) (probabilities list))
  ($choice elements (tensor probabilities)))

(defmethod $choice ((elements list) (probabilities vector))
  ($choice elements (tensor (coerce probabilities 'list))))

(defmethod $choice ((elements vector) (probabilities tensor))
  (when (and (eq 1 ($ndim probabilities)) (eq ($count elements) ($count probabilities)))
    (let ((choice ($multinomial ($div probabilities ($sum probabilities)) 1)))
      ($ elements ($ ($reshape! choice ($count probabilities)) 0)))))

(defmethod $choice ((elements vector) (probabilities list))
  ($choice elements (tensor probabilities)))

(defmethod $choice ((elements vector) (probabilities vector))
  ($choice elements (tensor (coerce probabilities 'list))))

(defmethod $array ((list list)) (coerce list 'vector))

(defmethod $argmax ((x tensor) &optional (dim -1))
  (if (< dim 0)
      (if (eq ($ndim x) 1)
          (let ((res (cadr ($max* x 0))))
            (if (eq 1 ($count res))
                ($ ($squeeze res) 0)
                res))
          (error "cannot find argmax for ndim > 1"))
      (let ((res (cadr ($max* x dim))))
        res)))

(defmethod $argmin ((x tensor) &optional (dim -1))
  (if (< dim 0)
      (if (eq ($ndim x) 1)
          (let ((res (cadr ($min* x 0))))
            (if (eq 1 ($count res))
                ($ ($squeeze res) 0)
                res))
          (error "cannot find argmin for ndim > 1"))
      (let ((res (cadr ($min* x dim))))
        res)))

(defun random-normals (means sds)
  (let ((samples (tensor ($count means))))
    (loop :for i :from 0 :below ($count means)
          :do (setf ($ samples i) ($normal *generator*
                                           ($ means i)
                                           ($ sds i))))
    samples))

(defmethod $scalar ((x tensor))
  (when (eq 1 ($count x))
    ($ ($storage x) 0)))

(defmethod $scalar ((x node))
  (when (eq 1 ($count ($data x)))
    ($ ($storage ($data x)) 0)))

(defmethod $scalar ((x T)) x)

(defmacro $incf (place &optional (delta 1)) `(setf ,place ($+ ,place ,delta)))
(defmacro $decf (place &optional (delta 1)) `(setf ,place ($- ,place ,delta)))

(defmethod $diagflat ((x tensor))
  ($diag ($reshape x ($count x))))

(defmethod $diagflat ((x list))
  ($diagflat (tensor x)))
