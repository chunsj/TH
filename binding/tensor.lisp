(in-package :th)

(defmethod $storage ((tensor tensor)) (tensor-storage tensor))
(defmethod $offset ((tensor tensor)) (tensor-storage-offset tensor))

(defmethod $ndim ((tensor tensor)) (tensor-n-dimension tensor))
(defmethod $ndim ((tensor null)) 0)
(defmethod $ndim ((tensor t)) 0)

(defmethod $coerce ((tensor tensor.byte) value) (coerce value 'unsigned-byte))
(defmethod $coerce ((tensor tensor.char) value) (coerce value 'signed-byte))
(defmethod $coerce ((tensor tensor.short) value) (coerce value 'integer))
(defmethod $coerce ((tensor tensor.int) value) (coerce value 'integer))
(defmethod $coerce ((tensor tensor.long) value) (coerce value 'integer))
(defmethod $coerce ((tensor tensor.float) value) (coerce value 'single-float))
(defmethod $coerce ((tensor tensor.double) value) (coerce value 'double-float))

(defmethod $acoerce ((tensor tensor.byte) value) (coerce value 'integer))
(defmethod $acoerce ((tensor tensor.char) value) (coerce value 'integer))
(defmethod $acoerce ((tensor tensor.short) value) (coerce value 'integer))
(defmethod $acoerce ((tensor tensor.int) value) (coerce value 'integer))
(defmethod $acoerce ((tensor tensor.long) value) (coerce value 'integer))
(defmethod $acoerce ((tensor tensor.float) value) (coerce value 'double-float))
(defmethod $acoerce ((tensor tensor.double) value) (coerce value 'double-float))

(defmethod $type ((tensor tensor.byte)) :unsigned-char)
(defmethod $type ((tensor tensor.char)) :char)
(defmethod $type ((tensor tensor.short)) :short)
(defmethod $type ((tensor tensor.int)) :int)
(defmethod $type ((tensor tensor.long)) :long)
(defmethod $type ((tensor tensor.float)) :float)
(defmethod $type ((tensor tensor.double)) :double)

(defmethod $copy ((tensor tensor) (source tensor))
  (tensor-copy tensor source)
  tensor)
(defmethod $copy ((tensor tensor) (source list))
  (tensor-copy tensor (tensor source))
  tensor)

(defmethod $fill ((tensor tensor) (value number))
  (tensor-fill tensor value)
  tensor)

(defmethod $size ((tensor tensor) &optional dimension)
  (cond ((null dimension) (let ((ndim (tensor-n-dimension tensor)))
                            (when (> ndim 0)
                              (mapcar (lambda (i) (tensor-size tensor i))
                                      (loop :for i :from 0 :below ndim :collect i)))))
        (t (tensor-size tensor dimension))))
(defmethod $size ((tensor null) &optional dimension) (declare (ignore dimension)))
(defmethod $size ((tensor t) &optional dimension) (declare (ignore dimension)))

(defmethod $stride ((tensor tensor) &optional dimension)
  (cond ((null dimension) (let ((ndim (tensor-n-dimension tensor)))
                            (when (> ndim 0)
                              (mapcar (lambda (i) (tensor-stride tensor i))
                                      (loop :for i :from 0 :below ndim :collect i)))))
        (t (tensor-stride tensor dimension))))
(defmethod $stride ((tensor null) &optional dimension) (declare (ignore dimension)))
(defmethod $stride ((tensor t) &optional dimension) (declare (ignore dimension)))

(defmethod $pointer ((tensor tensor)) (tensor-data tensor))

(defmethod $clone ((tensor tensor)) (tensor-clone tensor))

(defmethod $contiguous ((tensor tensor))
  (if ($contiguousp tensor)
      tensor
      (tensor-contiguous tensor)))

(defmethod $contiguousp ((tensor tensor)) (tensor-contiguous-p tensor))

(defmethod $select ((tensor tensor) dimension slice-index)
  (tensor-new-select tensor dimension slice-index))

(defmethod $select! ((tensor tensor) dimension slice-index)
  (tensor-select tensor tensor dimension slice-index)
  tensor)

(defmethod (setf $select) (value (tensor tensor) dimension index)
  (let ((x ($select tensor dimension index)))
    ($copy x value)
    value))

(defmethod $narrow ((tensor tensor) dimension first-index size)
  (tensor-new-narrow tensor dimension first-index size))

(defmethod $narrow! ((tensor tensor) dimension first-index size)
  (tensor-narrow tensor tensor dimension first-index size)
  tensor)

(defmethod (setf $narrow) (value (tensor tensor) dimension first-index size)
  (let ((x ($narrow tensor dimension first-index size)))
    ($copy x value)
    value))

(defmethod $transpose ((tensor tensor) &optional dimension0 dimension1)
  (cond ((and (null dimension0) (null dimension1)) (tensor-new-transpose tensor))
        (t (let ((result ($empty tensor)))
             (tensor-transpose result tensor dimension0 dimension1)
             result))))

(defmethod $transpose! ((tensor tensor) &optional dimension0 dimension1)
  (tensor-transpose tensor tensor (or dimension0 0) (or dimension1 1))
  tensor)

(defmethod $unfold ((tensor tensor) dimension size step)
  (tensor-new-unfold tensor dimension size step))

(defmethod $unfold! ((tensor tensor) dimension size step)
  (tensor-unfold tensor tensor dimension size step)
  tensor)

(defmethod $view ((tensor tensor) &rest sizes)
  (tensor-new-view tensor sizes))

(defmethod $expand ((tensor tensor) &rest sizes)
  (tensor-new-expand tensor sizes))

(defmethod $expand! ((tensor tensor) &rest sizes)
  (tensor-expand tensor tensor sizes)
  tensor)

(defmethod $set ((tensor tensor) (source tensor) &optional offset size stride)
  (declare (ignore offset size stride))
  (tensor-set tensor source)
  tensor)

(defmethod $set ((tensor tensor) (source storage) &optional offset size stride)
  (tensorn-set-storage tensor source offset size stride)
  tensor)

(defmethod $setp ((tensor tensor) (source tensor))
  (tensor-set-to-p tensor source))

(defmethod $sizep ((tensor tensor) (other tensor))
  (tensor-same-size-p tensor other))

(defmethod $sizep ((tensor tensor) (other list))
  (tensor-size-p tensor other))

(defmethod $sizep ((tensor tensor) (other storage.long))
  (tensor-size-p tensor other))

(defmethod $resize ((tensor tensor) (size list) &optional stride)
  (tensor-resize tensor size stride)
  tensor)

(defmethod $resize ((tensor tensor) (other tensor) &optional stride)
  (declare (ignore stride))
  (tensor-resize-as tensor other)
  tensor)

(defmethod $resize ((tensor tensor) (size storage.long) &optional stride)
  (tensor-resize tensor ($list size) ($list stride))
  tensor)

(defmethod $zero ((tensor tensor))
  (let ((nt ($empty tensor)))
    ($resize nt tensor)
    (tensor-zero nt)
    nt))

(defmethod $zero! ((tensor tensor))
  (tensor-zero tensor)
  tensor)

(defmethod $one ((tensor tensor))
  (let ((nt ($empty tensor)))
    ($resize nt tensor)
    (tensor-fill nt 1)
    nt))

(defmethod $one! ((tensor tensor))
  (tensor-fill tensor 1)
  tensor)

(defmethod $subview ((tensor tensor) &rest index-sizes)
  (let ((cx ($narrow tensor 0 (car index-sizes) (cadr index-sizes)))
        (dim 1))
    (loop :for (i s) :on (cddr index-sizes) :by #'cddr
          :do (progn ($narrow! cx dim i s)
                     (incf dim)))
    cx))

(defmethod (setf $subview) (value (tensor tensor) &rest index-sizes)
  (let ((x (apply #'$subview tensor index-sizes)))
    ($copy x value)
    value))

(defun zeros (&rest sizes)
  "Returns a new tensor filled with zero with given sizes"
  (let ((x (make-tensor *default-tensor-class* sizes)))
    ($zero! x)
    x))

(defun ones (&rest sizes)
  "Returns a new tensor filled with one with given sizes"
  (let ((x (make-tensor *default-tensor-class* sizes)))
    ($one! x)
    x))

(defun filled (value &rest sizes)
  "Returns a new tensor filled with value with given sizes"
  (let ((x (make-tensor *default-tensor-class* (or sizes '(1)))))
    ($fill x value)
    x))

(defun rnd (&rest sizes)
  (let ((tensor (tensor)))
    (tensor-randn tensor sizes)
    tensor))

(defun rndn (&rest sizes)
  (let ((tensor (tensor)))
    (tensor-randn tensor sizes)
    tensor))

(defun range (from to &optional (step 1) type)
  (let* ((type (or type *default-tensor-class*))
         (tensor (make-tensor type)))
    (tensor-range tensor from to step)
    tensor))

(defun arange (from to &optional (step 1) type)
  (let* ((type (or type *default-tensor-class*))
         (tensor (make-tensor type)))
    (tensor-arange tensor from to step)
    tensor))

(defmethod $list ((tensor tensor)) ($list ($storage tensor)))

(defmethod $compare (spec (a tensor) (b tensor))
  (let ((result (tensor.byte)))
    (tensor-compare spec result a b)
    result))

(defmethod $compare (spec (a tensor) (b number))
  (let ((result (tensor.byte)))
    (tensor-compare spec result a b)
    result))

(defmethod $lt ((a tensor) (b tensor)) ($compare :lt a b))
(defmethod $lt ((a tensor) (b number)) ($compare :lt a b))
(defmethod $lt ((a number) (b tensor)) ($compare :ge b a))

(defmethod $le ((a tensor) (b tensor)) ($compare :le a b))
(defmethod $le ((a tensor) (b number)) ($compare :le a b))
(defmethod $le ((a number) (b tensor)) ($compare :gt b a))

(defmethod $gt ((a tensor) (b tensor)) ($compare :gt a b))
(defmethod $gt ((a tensor) (b number)) ($compare :gt a b))
(defmethod $gt ((a number) (b tensor)) ($compare :le b a))

(defmethod $ge ((a tensor) (b tensor)) ($compare :ge a b))
(defmethod $ge ((a tensor) (b number)) ($compare :ge a b))
(defmethod $ge ((a number) (b tensor)) ($compare :lt b a))

(defmethod $eq ((a tensor) (b tensor)) ($compare :eq a b))
(defmethod $eq ((a tensor) (b number)) ($compare :eq a b))
(defmethod $eq ((a number) (b tensor)) ($compare :eq b a))

(defmethod $ne ((a tensor) (b tensor)) ($compare :ne a b))
(defmethod $ne ((a tensor) (b number)) ($compare :ne a b))
(defmethod $ne ((a number) (b tensor)) ($compare :ne b a))

(defmethod $index ((tensor tensor) dimension (indices list))
  (let ((result ($empty tensor)))
    (tensor-index-select result tensor dimension (tensor.long indices))
    result))

(defmethod $index ((tensor tensor) dimension (indices tensor.long))
  (let ((result ($empty tensor)))
    (tensor-index-select result tensor dimension indices)
    result))

(defmethod (setf $index) ((value number) (tensor tensor) dimension (indices list))
  (tensor-index-fill tensor value dimension (tensor.long indices))
  value)

(defmethod (setf $index) ((value number) (tensor tensor) dimension (indices tensor.long))
  (tensor-index-fill tensor value dimension indices)
  value)

(defmethod (setf $index) ((value tensor) (tensor tensor) dimension (indices list))
  (tensor-index-copy tensor value dimension (tensor.long indices))
  value)

(defmethod (setf $index) ((value tensor) (tensor tensor) dimension (indices tensor.long))
  (tensor-index-copy tensor value dimension indices)
  value)

(defmethod (setf $index) ((value list) (tensor tensor) dimension (indices list))
  (tensor-index-copy tensor (make-tensor-args (type-of tensor) (list value))
                     dimension (storage.long indices))
  value)

(defmethod (setf $index) ((value list) (tensor tensor) dimension (indices tensor.long))
  (tensor-index-copy tensor (make-tensor-args (type-of tensor) (list value))
                     dimension indices)
  value)

(defmethod $gather ((tensor tensor) dimension (indices list))
  (let ((result ($empty tensor))
        (indices (tensor.long indices)))
    ($resize result indices)
    (tensor-gather result tensor dimension indices)
    result))

(defmethod $gather ((tensor tensor) dimension (indices tensor))
  (let ((result ($empty tensor)))
    ($resize result indices)
    (tensor-gather result tensor dimension indices)
    result))

(defmethod $scatter ((tensor tensor) dimension (indices list) (value tensor))
  (let ((indices (tensor.long indices)))
    (tensor-scatter tensor value dimension indices)
    tensor))

(defmethod $scatter ((tensor tensor) dimension (indices list) (value list))
  (let ((indices (tensor.long indices)))
    (tensor-scatter tensor (make-tensor-args (type-of tensor) (list value)) dimension indices)
    tensor))

(defmethod $scatter ((tensor tensor) dimension (indices list) (value number))
  (let ((indices (tensor.long indices)))
    (tensor-scatter-fill tensor value dimension indices)
    tensor))

(defmethod $scatter ((tensor tensor) dimension (indices tensor.long) (value tensor))
  (tensor-scatter tensor value dimension indices)
  tensor)

(defmethod $scatter ((tensor tensor) dimension (indices tensor.long) (value list))
  (tensor-scatter tensor (make-tensor-args (type-of tensor) (list value)) dimension indices)
  tensor)

(defmethod $scatter ((tensor tensor) dimension (indices tensor.long) (value number))
  (tensor-scatter-fill tensor value dimension indices)
  tensor)

(defmethod $masked ((tensor tensor) (mask tensor.byte))
  (let ((result ($empty tensor)))
    (tensor-masked-select result mask tensor)
    result))

(defmethod $masked ((tensor tensor) (mask list))
  (let ((result ($empty tensor)))
    (tensor-masked-select result (make-tensor-args 'tensor.byte (list mask)) tensor)
    result))

(defmethod (setf $masked) ((value number) (tensor tensor) (mask tensor.byte))
  (tensor-masked-fill tensor mask value)
  value)

(defmethod (setf $masked) ((value tensor) (tensor tensor) (mask tensor.byte))
  (tensor-masked-copy tensor mask value)
  value)

(defmethod (setf $masked) ((value list) (tensor tensor) (mask tensor.byte))
  (tensor-masked-copy tensor mask (make-tensor-args (type-of tensor) (list value)))
  value)

(defmethod $nonzero ((tensor tensor)) (tensor-non-zero tensor))

(defmethod $ ((tensor tensor) (location number) &rest others-and-default)
  (let ((locs (cons location others-and-default)))
    (cond ((= ($ndim tensor) ($count locs))
           (let ((idx (+ ($offset tensor) (loop :for i :from 0 :below ($count locs)
                                                :sum (* ($ locs i) ($stride tensor i))))))
             ($ ($storage tensor) idx)))
          ((> ($ndim tensor) ($count locs))
           (let ((cx (tensor-new-select tensor 0 ($ locs 0))))
             (loop :for i :from 1 :below ($count locs)
                   :do (tensor-select cx cx 0 ($ locs i)))
             cx)))))

(defmethod $ ((tensor tensor) (location list) &rest others-and-default)
  (labels ((flattenl (l) (cond ((null l) nil)
                               ((atom (car l)) (cons (car l) (flattenl (cdr l))))
                               (t (append (flattenl (car l)) (flattenl (cdr l)))))))
    (apply #'$subview tensor (flattenl (append (list location) others-and-default)))))

(defmethod $ ((tensor tensor) (location tensor.byte) &rest others-and-default)
  (declare (ignore others-and-default))
  (let ((result ($empty tensor)))
    (tensor-masked-select result location tensor)
    result))

(defmethod (setf $) ((value number) (tensor tensor) (location number) &rest others)
  (let ((locs (cons location others)))
    (cond ((eq ($count locs) ($ndim tensor))
           (let ((s ($storage tensor)))
             (setf ($ s (+ ($offset tensor)
                           (loop :for i :from 0 :below ($count locs)
                                 :sum (* ($ locs i) ($stride tensor i)))))
                   value)))
          ((> ($ndim tensor) ($count locs))
           (let ((cx (tensor-new-select tensor 0 ($ locs 0))))
             (loop :for i :from 1 :below ($count locs)
                   :do (tensor-select cx cx 0 ($ locs i)))
             (tensor-fill cx value))))
    value))

(defmethod (setf $) ((value tensor) (tensor tensor) (location number) &rest others)
  (let ((locs (cons location others)))
    (when (> ($ndim tensor) ($count locs))
      (let ((cx (tensor-new-select tensor 0 ($ locs 0))))
        (loop :for i :from 1 :below ($count locs)
              :do (tensor-select cx cx 0 ($ locs i)))
        (tensor-copy cx value)
        value))))

(defmethod (setf $) ((value list) (tensor tensor) (location number) &rest others)
  (let ((locs (cons location others)))
    (when (> ($ndim tensor) ($count locs))
      (let ((cx (tensor-new-select tensor 0 ($ locs 0))))
        (loop :for i :from 1 :below ($count locs)
              :do (tensor-select cx cx 0 ($ locs i)))
        (tensor-copy cx (make-tensor-args (type-of tensor) (list value)))
        value))))

(defmethod (setf $) ((value number) (tensor tensor) (location list) &rest others)
  (labels ((flattenl (l) (cond ((null l) nil)
                               ((atom (car l)) (cons (car l) (flattenl (cdr l))))
                               (t (append (flattenl (car l)) (flattenl (cdr l)))))))
    (let ((x (apply #'$subview tensor (flattenl (append (list location) others)))))
      (tensor-fill x value))
    value))

(defmethod (setf $) ((value tensor) (tensor tensor) (location list) &rest others)
  (labels ((flattenl (l) (cond ((null l) nil)
                               ((atom (car l)) (cons (car l) (flattenl (cdr l))))
                               (t (append (flattenl (car l)) (flattenl (cdr l)))))))
    (let ((x (apply #'$subview tensor (flattenl (append (list location) others)))))
      (tensor-copy x value))
    value))

(defmethod (setf $) ((value list) (tensor tensor) (location list) &rest others)
  (labels ((flattenl (l) (cond ((null l) nil)
                               ((atom (car l)) (cons (car l) (flattenl (cdr l))))
                               (t (append (flattenl (car l)) (flattenl (cdr l)))))))
    (let ((x (apply #'$subview tensor (flattenl (append (list location) others)))))
      (tensor-copy x (make-tensor-args (type-of tensor) (list value))))
    value))

(defmethod (setf $) ((value number) (tensor tensor) (location tensor.byte) &rest others)
  (declare (ignore others))
  (tensor-masked-fill tensor location value)
  value)

(defmethod (setf $) ((value tensor) (tensor tensor) (location tensor.byte) &rest others)
  (declare (ignore others))
  (tensor-masked-copy tensor location value)
  value)

(defmethod (setf $) ((value list) (tensor tensor) (location tensor.byte) &rest others)
  (declare (ignore others))
  (tensor-masked-copy tensor location (make-tensor-args (type-of tensor) (list value)))
  value)

(defmethod $repeat ((tensor tensor) &rest sizes)
  (let* ((result ($empty tensor))
         (tensor (if ($contiguousp tensor) tensor ($clone tensor)))
         (size (if (eq 1 ($count sizes))
                   (append sizes '(1))
                   (copy-list sizes)))
         (xtensor (tensor tensor))
         (xsize ($size xtensor))
         (xsize (progn (loop :for i :from 0 :below (- ($count size) ($ndim tensor))
                             :do (push 1 xsize))
                       xsize))
         (size-tensor (tensor-cmul (tensor.long) (tensor.long xsize) (tensor.long size)))
         (size (loop :for i :from 0 :below ($count size-tensor)
                     :collect ($ ($storage size-tensor) i)))
         (result ($resize result size))
         (urtensor (tensor result))
         (xtensor ($resize xtensor xsize)))
    (loop :for i :from 0 :below ($ndim xtensor)
          :for xs = ($size xtensor i)
          :do (setf urtensor ($unfold urtensor i xs xs)))
    (loop :for i :from 0 :below (- ($ndim urtensor) ($ndim xtensor))
          :do (push 1 xsize))
    ($resize xtensor xsize)
    ($copy urtensor (apply #'$expand xtensor ($size urtensor)))
    result))

(defmethod $squeeze ((tensor tensor) &optional (dimension -1))
  (let ((result ($empty tensor)))
    (tensor-squeeze result tensor dimension)
    result))

(defmethod $squeeze! ((tensor tensor) &optional (dimension -1))
  (tensor-squeeze tensor tensor dimension)
  tensor)

(defmethod $unsqueeze ((tensor tensor) dimension)
  (let ((result ($empty tensor)))
    (tensor-unsqueeze result tensor dimension)
    result))

(defmethod $unsqueeze! ((tensor tensor) dimension)
  (tensor-unsqueeze tensor tensor dimension)
  tensor)

(defmethod $permute ((tensor tensor) &rest dimensions)
  (let ((out (tensor tensor))
        (perms (copy-list dimensions)))
    (loop :for i :from 0 :below ($ndim tensor)
          :for p = ($ perms i)
          :when (and (not (eq i p)) (not (eq p -1)))
            :do (let ((j i))
                  (loop :until (eq i ($ perms j))
                        :do (progn (setf out ($transpose out j ($ perms j)))
                                   (let ((oldj j))
                                     (setf j ($ perms j)
                                           ($ perms oldj) -1))))
                  (setf ($ perms j) j)))
    out))

(defmethod $fmap (fn (tensor tensor) &rest tensors)
  (let ((s ($storage tensor)))
    (loop :for i :from 0 :below ($count s)
          :for x = ($ s i)
          :for ys = (mapcar (lambda (aten) ($ aten i)) tensors)
          :do (let ((v (apply #'funcall fn x ys)))
                (when v (setf ($ s i) v))))
    tensor))

(defmethod $split ((tensor tensor) size &optional (dimension 0))
  (let* ((dlen ($size tensor dimension))
         (pcnt (floor (/ dlen size)))
         (remlen (- dlen (* pcnt size))))
    (append (loop :for i :from 0 :below pcnt
                  :for si = (* size i)
                  :collect ($narrow tensor dimension si size))
            (when (> remlen 0)
              (list ($narrow tensor dimension (* pcnt size) remlen))))))

(defmethod $chunk ((tensor tensor) n &optional (dimension 0))
  ($split tensor (ceiling (/ ($size tensor dimension) n)) dimension))

(defmethod $cat ((dimension integer) (tensor tensor) &rest tensors)
  (let ((tensors (cons tensor tensors))
        (result ($empty tensor)))
    (apply #'tensor-cat result dimension tensors)
    result))

(defmethod $diag ((tensor tensor) &optional (k 0))
  (let ((result ($empty tensor)))
    (tensor-diag result tensor k)))

(defmethod $diag! ((tensor tensor) &optional (k 0))
  (tensor-diag tensor tensor k))

(defmethod $eye ((tensor tensor) m &optional n)
  (let ((result ($empty tensor))
        (n (or n m)))
    (tensor-eye result m n)
    result))

(defmethod $eye! ((tensor tensor) m &optional n)
  (let ((n (or n m)))
    (tensor-eye tensor m n)
    tensor))

(defun eye (m &optional n) ($eye! (tensor) m n))

(defun linspace (a b &optional (n 100))
  "Returns a one-dimensional tensor with equally spaced points between a and b."
  (tensor-linspace (tensor) a b n))

(defun logspace (a b &optional (n 100))
  "Returns a one-dimensional tensor with logarithmically equally spaced points between a and b."
  (tensor-logspace (tensor) a b n))
