(in-package :th)

(defgeneric $conv2d (x k &optional b dw dh pw ph)
  (:documentation "Performs convolution of x with kernel k and bias b which is optional."))
(defgeneric $maxpool2d (x kw kh &optional dw dh pw ph ceilp)
  (:documentation "Performs max pooling over x."))
(defgeneric $avgpool2d (x kw kh &optional dw dh pw ph ceilp count-pad-p)
  (:documentation "Performs average pooling over x."))

(defmethod $conv2d ((x tensor) (k tensor) &optional b (dw 1) (dh 1) (pw 0) (ph 0))
  (let ((out ($empty x))
        (f ($empty x))
        (df ($empty x)))
    (nn-spatial-convolution-mm-update-output x out k b f df ($size k 2) ($size k 3)
                                             dw dh pw ph)
    out))

(defun conv2d-backprop (node gradient f df dw dh pw ph)
  (setf ($gradient node) gradient)
  (setf ($children node) (when ($children node)
                           (if (eq 3 ($count ($children node)))
                               (let* ((x ($c0 node))
                                      (k ($c1 node))
                                      (b ($c2 node))
                                      (dx ($empty ($data x)))
                                      (dk (apply #'zeros ($size k)))
                                      (db (apply #'zeros ($size b))))
                                 (if (or ($gradientp x) ($gradientp k) ($gradientp b))
                                     (nn-spatial-convolution-mm-update-grad-input ($data x)
                                                                                  gradient
                                                                                  dx
                                                                                  ($data k)
                                                                                  f
                                                                                  df
                                                                                  ($size k 2)
                                                                                  ($size k 3)
                                                                                  dw dh pw ph))
                                 (if (or ($gradientp k) ($gradientp b))
                                     (nn-spatial-convolution-mm-acc-grad-parameters ($data x)
                                                                                    gradient
                                                                                    dk
                                                                                    db
                                                                                    f
                                                                                    df
                                                                                    ($size k 2)
                                                                                    ($size k 3)
                                                                                    dw dh pw ph
                                                                                    1))
                                 (list (if ($gradientp x)
                                           ($bp! x dx)
                                           x)
                                       (if ($gradientp k)
                                           ($bp! k dk)
                                           k)
                                       (if ($gradientp b)
                                           ($bp! b db)
                                           b)))
                               (let* ((x ($c0 node))
                                      (k ($c1 node))
                                      (dx ($empty ($data x)))
                                      (dk (apply #'zeros ($size k))))
                                 (nn-spatial-convolution-mm-update-grad-input ($data x)
                                                                              gradient
                                                                              dx
                                                                              ($data k)
                                                                              f
                                                                              df
                                                                              ($size k 2)
                                                                              ($size k 3)
                                                                              dw dh pw ph)
                                 (nn-spatial-convolution-mm-acc-grad-parameters ($data x)
                                                                                gradient
                                                                                dk
                                                                                nil
                                                                                f
                                                                                df
                                                                                ($size k 2)
                                                                                ($size k 3)
                                                                                dw dh pw ph
                                                                                1)
                                 (list (if ($gradientp x)
                                           ($bp! x dx)
                                           x)
                                       (if ($gradientp k)
                                           ($bp! k dk)
                                           k))))))
  node)

(defmethod $conv2d ((x node) (k node) &optional b (dw 1) (dh 1) (pw 0) (ph 0))
  (let ((f ($empty ($data x)))
        (df ($empty ($data x)))
        (out ($empty ($data x))))
    (nn-spatial-convolution-mm-update-output ($data x) out ($data k) (if b ($data b) b)
                                             f df ($size k 2) ($size k 3) dw dh pw ph)
    (let ((result (th::node out)))
      (setf ($children result) (if b
                                   (list x k b)
                                   (list x k)))
      (setf ($gradientp result) (or ($gradientp x) ($gradientp k)
                                    (if b ($gradientp b) nil)))
      (setf ($bpfn result) (lambda (node gradient)
                             (conv2d-backprop node gradient f df dw dh pw ph)))
      result)))

(defmethod $maxpool2d ((x tensor) kw kh &optional (dw 1) (dh 1) (pw 0) (ph 0) ceilp)
  (let ((out ($empty x))
        (indices (tensor.long)))
    (nn-spatial-max-pooling-update-output x out indices kw kh dw dh pw ph ceilp)
    out))

(defun maxpool2d-backprop (node gradient indices kw kh dw dh pw ph ceilp)
  (setf ($gradient node) gradient)
  (setf ($children node) (when ($children node)
                           (let* ((x ($c0 node))
                                  (dx ($empty ($data x))))
                             (nn-spatial-max-pooling-update-grad-input ($data x)
                                                                       gradient
                                                                       dx
                                                                       indices
                                                                       kw kh dw dh pw ph
                                                                       ceilp)
                             (list (if ($gradientp x)
                                       ($bp! x dx)
                                       x)))))
  node)

(defmethod $maxpool2d ((x node) kw kh &optional (dw 1) (dh 1) (pw 0) (ph 0) ceilp)
  (let ((out ($empty ($data x)))
        (indices (tensor.long)))
    (nn-spatial-max-pooling-update-output ($data x) out indices kw kh dw dh pw ph ceilp)
    (let ((result (node out)))
      (setf ($children result) (list x))
      (setf ($gradientp result) ($gradientp x))
      (setf ($bpfn result) (lambda (node gradient)
                             (maxpool2d-backprop node gradient indices kw kh dw dh pw ph ceilp)))
      result)))

(defmethod $avgpool2d ((x tensor) kw kh &optional (dw 1) (dh 1) (pw 0) (ph 0) ceilp (count-pad-p t))
  (let ((out ($empty x)))
    (nn-spatial-average-pooling-update-output x out kw kh dw dh pw ph ceilp count-pad-p)
    out))

(defun avgpool2d-backprop (node gradient kw kh dw dh pw ph ceilp count-pad-p)
  (setf ($gradient node) gradient)
  (setf ($children node) (when ($children node)
                           (let* ((x ($c0 node))
                                  (dx ($empty ($data x))))
                             (nn-spatial-average-pooling-update-grad-input ($data x)
                                                                           gradient
                                                                           dx
                                                                           kw kh dw dh pw ph
                                                                           ceilp
                                                                           count-pad-p)
                             (list (if ($gradientp x)
                                       ($bp! x dx)
                                       x)))))
  node)

(defmethod $avgpool2d ((x node) kw kh &optional (dw 1) (dh 1) (pw 0) (ph 0) ceilp (count-pad-p t))
  (let ((out ($empty ($data x))))
    (nn-spatial-average-pooling-update-output ($data x) out kw kh dw dh pw ph ceilp count-pad-p)
    (let ((result (node out)))
      (setf ($children result) (list x))
      (setf ($gradientp result) ($gradientp x))
      (setf ($bpfn result) (lambda (node gradient)
                             (avgpool2d-backprop node gradient kw kh dw dh pw ph
                                                 ceilp count-pad-p)))
      result)))

(defun conv2-backprop (node gradient type)
  (declare (ignore type))
  (setf ($gradient node) gradient)
  (setf ($children node) (when ($children node)
                           (let ((x ($c0 node))
                                 (k ($c1 node)))
                             (list (if ($gradientp x)
                                       ($bp! x ($xcorr2 gradient ($data k) :full))
                                       x)
                                   (if ($gradientp k)
                                       ($bp! k ($conv2 ($data x) gradient))
                                       k)))))
  node)

(defmethod $conv2 ((x node) (k node) &optional (type :valid))
  (let ((result (node ($conv2 ($data x) ($data k) type))))
    (setf ($children result) (list x k))
    (setf ($gradientp result) (or ($gradientp x) ($gradientp k)))
    (setf ($bpfn result) (lambda (node gradient)
                           (conv2-backprop node gradient type)))
    result))
