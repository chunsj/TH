(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defgeneric $xwpb (x w b &optional ones) (:documentation "Returns x@w + b."))
(defgeneric $affine (x w b &optional ones) (:documentation "Affine transformation."))

(defgeneric $wimb (xwi w) (:documentation "Computes word embedding."))

(defgeneric $rn! (tensor &optional µ σ) (:documentation "Fills with random normal."))
(defgeneric $rnt! (tensor &optional µ σ) (:documentation "Fills with truncated random normal."))
(defgeneric $ru! (tensor &optional min max) (:documentation "Fills with random uniform."))
(defgeneric $xavieru! (tensor) (:documentation "Fills with Xavier uniform."))
(defgeneric $xaviern! (tensor) (:documentation "Fills with Xavier normal."))
(defgeneric $heu! (tensor) (:documentation "Fills with He uniform."))
(defgeneric $hen! (tensor) (:documentation "Fills with He normal."))
(defgeneric $lecunu! (tensor) (:documentation "Fills with Lecun uniform."))
(defgeneric $lecunn! (tensor) (:documentation "Fills with Lecun normal."))

(defun allocate-addbuf (nframe)
  (let ((tensor (make-instance *default-tensor-class*)))
    (allocate-tensor-handle tensor (list nframe))
    ($one! tensor)))

(defun affine-without-bias (x w)
  (let ((dim ($ndim x)))
    (cond ((eq dim 1) (let ((output ($zero! ($resize! ($empty x) (list ($size w 1)))))
                            (tw (allocate-transpose w)))
                        ($addmv! output tw x 1 1)
                        (deallocate-tensor-handle tw)
                        output))
          ((eq dim 2) (let* ((nframe ($size x 0))
                             (output ($zero! ($resize! ($empty x)
                                                       (list nframe ($size w 1))))))
                        ($addmm! output x w 1 0)
                        output)))))

(defun affine-with-bias (x w b os)
  (let ((dim ($ndim x)))
    (cond ((eq dim 1) (let ((output ($copy! ($resize! ($empty x) (list ($size w 1))) b))
                            (tw (allocate-transpose w)))
                        ($addmv! output tw x 1 1)
                        (deallocate-tensor-handle tw)
                        output))
          ((eq dim 2) (let* ((nframe ($size x 0))
                             (output ($zero! ($resize! ($empty x) (list nframe ($size w 1)))))
                             (addbuf (or os (allocate-addbuf nframe))))
                        ($addmm! output x w 1 0)
                        ($addr! output addbuf b 1 1)
                        (when (null os) (deallocate-tensor-handle addbuf))
                        output)))))

(defun daffine-output (x w gv)
  (let ((dx ($zero x))
        (dim ($ndim x)))
    (cond ((eq dim 1) (progn
                        ($addmv! dx w gv 1 0)
                        dx))
          ((eq dim 2) (let ((tw (allocate-transpose w)))
                        ($addmm! dx gv tw 1 0)
                        (deallocate-tensor-handle tw)
                        dx)))))

(defun daffine-weight (x w gv)
  (let ((dw ($zero w))
        (dim ($ndim x)))
    (cond ((eq dim 1) (progn
                        ($addr! dw x gv 1 1)
                        dw))
          ((eq dim 2) (let ((tx (allocate-transpose x)))
                        ($addmm! dw tx gv 1 1)
                        (deallocate-tensor-handle tx)
                        dw)))))

(defun daffine-bias (x b gv os)
  (let ((db ($zero b))
        (dim ($ndim x)))
    (cond ((eq dim 1) (progn
                        (tensor-cadd db db 1 gv)
                        db))
          ((eq dim 2) (let* ((nframe ($size x 0))
                             (tgv (allocate-transpose gv))
                             (addbuf (or os (allocate-addbuf nframe))))
                        ($addmv! db tgv addbuf 1 1)
                        (when (null os) (deallocate-tensor-handle addbuf))
                        (deallocate-tensor-handle tgv)
                        db)))))

(defmethod $xwpb ((x tensor) (w tensor) (b tensor) &optional ones)
  (cond ((null b) (affine-without-bias x w))
        (t (affine-with-bias x w b ones))))

(defmethod $xwpb ((x node) (w node) (b node) &optional ones)
  (node ($xwpb ($data x) ($data w) (when b ($data b)) ones)
        :name :xwpb
        :link (link
                (to x (daffine-output ($data x) ($data w) gv))
                (to w (daffine-weight ($data x) ($data w) gv))
                (when b (to b (daffine-bias ($data x) ($data b) gv ones))))))

(defmethod $xwpb ((x tensor) (w node) (b node) &optional ones)
  (node ($xwpb x ($data w) (when b ($data b)) ones)
        :name :xwpb
        :link (link
                (to w (daffine-weight x ($data w) gv))
                (when b (to b (daffine-bias x ($data b) gv ones))))))

(defmethod $xwpb ((x tensor) (w tensor) (b node) &optional ones)
  (node ($xwpb x w (when b ($data b)) ones)
        :name :xwpb
        :link (link
                (when b (to b (daffine-bias x ($data b) gv ones))))))

;; (defmethod $xwpb ((x tensor) (w tensor) (b tensor) &optional ones)
;;   (let ((o (or ones (ones (if (eq 1 ($ndim x)) 1 ($size x 0))))))
;;     ($add! ($mm x w) ($vv o b))))

;; (defmethod $xwpb ((x node) (w node) (b node) &optional ones)
;;   (let ((o (or ones (ones (if (eq 1 ($ndim x)) 1 ($size x 0))))))
;;     ($add ($mm x w) ($vv o b))))

;; (defmethod $xwpb ((x tensor) (w node) (b node) &optional ones)
;;   (let ((o (or ones (ones (if (eq 1 ($ndim x)) 1 ($size x 0))))))
;;     ($add ($mm x w) ($vv o b))))

(defmethod $affine ((x tensor) (w tensor) (b tensor) &optional ones)
  (cond ((null b) (affine-without-bias x w))
        (t (affine-with-bias x w b ones))))

(defmethod $affine ((x node) (w node) (b node) &optional ones)
  (node ($xwpb ($data x) ($data w) (when b ($data b)) ones)
        :name :xwpb
        :link (link
                (to x (daffine-output ($data x) ($data w) gv))
                (to w (daffine-weight ($data x) ($data w) gv))
                (when b (to b (daffine-bias ($data x) ($data b) gv ones))))))

(defmethod $affine ((x tensor) (w node) (b node) &optional ones)
  (node ($xwpb x ($data w) (when b ($data b)) ones)
        :name :xwpb
        :link (link
                (to w (daffine-weight x ($data w) gv))
                (when b (to b (daffine-bias x ($data b) gv ones))))))

(defmethod $affine ((x tensor) (w tensor) (b node) &optional ones)
  (node ($xwpb x w (when b ($data b)) ones)
        :name :xwpb
        :link (link
                (when b (to b (daffine-bias x ($data b) gv ones))))))

;; (defmethod $affine ((x tensor) (w tensor) (b tensor) &optional ones)
;;   (let ((o (or ones (ones ($size x 0) 1))))
;;     ($add! ($mm x w) ($mm o b))))

;; (defmethod $affine ((x node) (w node) (b node) &optional ones)
;;   (let ((o (or ones (ones ($size x 0) 1))))
;;     ($add ($mm x w) ($mm o b))))

;; (defmethod $affine ((x tensor) (w node) (b node) &optional ones)
;;   (let ((o (or ones (ones ($size x 0) 1))))
;;     ($add ($mm x w) ($mm o b))))

(defmethod $wimb ((xwi list) (w tensor)) ($sum ($index w 0 xwi) 0))
(defmethod $wimb ((xwi tensor.int) (w tensor)) ($sum ($index w 0 xwi) 0))
(defmethod $wimb ((xwi tensor.long) (w tensor)) ($sum ($index w 0 xwi) 0))

(defmethod $wimb ((xwi list) (w node)) ($sum ($index w 0 xwi) 0))
(defmethod $wimb ((xwi tensor.int) (w node)) ($sum ($index w 0 xwi) 0))
(defmethod $wimb ((xwi tensor.long) (w node)) ($sum ($index w 0 xwi) 0))

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
        ($rn! tensor 0 (/ 1 (sqrt ($size tensor 0))))
        ($ru! tensor (/ -1 (sqrt ($size tensor 0))) (/ 1 (sqrt ($size tensor 0)))))))
