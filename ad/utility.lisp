(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defgeneric $xwpb (x w b &optional ones))

(defmethod $xwpb ((x tensor) (w tensor) (b tensor) &optional ones)
  (let ((o (or ones (ones (if (eq 1 ($ndim x)) 1 ($size x 0))))))
    ($add! ($mm x w) ($vv o b))))

(defmethod $xwpb ((x node) (w node) (b node) &optional ones)
  (let ((o (or ones ($constant (ones (if (eq 1 ($ndim x)) 1 ($size x 0)))))))
    ($add ($mm x w) ($vv o b))))

(defgeneric $wimb (xwi w))

(defmethod $wimb ((xwi list) (w tensor)) ($sum ($index w 0 xwi) 0))
(defmethod $wimb ((xwi tensor.int) (w tensor)) ($sum ($index w 0 xwi) 0))
(defmethod $wimb ((xwi tensor.long) (w tensor)) ($sum ($index w 0 xwi) 0))

(defmethod $wimb ((xwi list) (w node)) ($sum ($index w 0 xwi) 0))
(defmethod $wimb ((xwi tensor.int) (w node)) ($sum ($index w 0 xwi) 0))
(defmethod $wimb ((xwi tensor.long) (w node)) ($sum ($index w 0 xwi) 0))

(defgeneric $rn! (tensor &optional µ σ) (:documentation "Fills with random normal."))
(defgeneric $rnt! (tensor &optional µ σ) (:documentation "Fills with truncated random normal."))
(defgeneric $ru! (tensor &optional mean max) (:documentation "Fills with random uniform."))
(defgeneric $xavieru! (tensor) (:documentation "Fills with Xavier uniform."))
(defgeneric $xaviern! (tensor) (:documentation "Fills with Xavier normal."))
(defgeneric $heu! (tensor) (:documentation "Fills with He uniform."))
(defgeneric $hen! (tensor) (:documentation "Fills with He normal."))
(defgeneric $lecunu! (tensor) (:documentation "Fills with Lecun uniform."))
(defgeneric $lecunn! (tensor) (:documentation "Fills with Lecun normal."))

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
