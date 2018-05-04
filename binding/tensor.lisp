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

(defmethod $select ((tensor tensor) dimension slice-index)
  (tensor-new-select tensor dimension slice-index))

(defmethod $select! ((tensor tensor) dimension slice-index)
  (tensor-select tensor tensor dimension slice-index)
  tensor)

(defmethod $narrow ((tensor tensor) dimension first-index size)
  (tensor-new-narrow tensor dimension first-index size))

(defmethod $narrow! ((tensor tensor) dimension first-index size)
  (tensor-narrow tensor tensor dimension first-index size)
  tensor)

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

(defmethod $set ((tensor tensor) (source tensor))
  (tensor-set tensor source)
  tensor)

(defmethod $setp ((tensor tensor) (source tensor))
  (tensor-set-to-p tensor source))
