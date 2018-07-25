(declaim (optimize (speed 3) (debug 1) (safety 0)))

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

(defmethod $conv2d ((x node) (k node) &optional b (dw 1) (dh 1) (pw 0) (ph 0))
  (let ((f ($empty ($data x)))
        (df ($empty ($data x)))
        (out ($empty ($data x))))
    (nn-spatial-convolution-mm-update-output ($data x) out ($data k) (if b ($data b) b)
                                             f df ($size k 2) ($size k 3) dw dh pw ph)
    (let ((result (node out)))
      (setf ($name result) "CONV2D")
      (if b
          ($gp! result x k b)
          ($gp! result x k))
      (if b
          (let* ((dx nil)
                 (dk nil)
                 (db nil)
                 (gfn (lambda ()
                        (unless (and dx dk db)
                          (setf dx ($empty ($data x)))
                          (setf dk (apply #'zeros ($size k)))
                          (setf db (apply #'zeros ($size b)))
                          (if (or ($gradientp x) ($gradientp k) ($gradientp b))
                              (nn-spatial-convolution-mm-update-grad-input ($data x)
                                                                           ($gradient result)
                                                                           dx
                                                                           ($data k)
                                                                           f
                                                                           df
                                                                           ($size k 2)
                                                                           ($size k 3)
                                                                           dw dh pw ph))
                          (if (or ($gradientp k) ($gradientp b))
                              (nn-spatial-convolution-mm-acc-grad-parameters ($data x)
                                                                             ($gradient result)
                                                                             dk
                                                                             db
                                                                             f
                                                                             df
                                                                             ($size k 2)
                                                                             ($size k 3)
                                                                             dw dh pw ph
                                                                             1))))))
            ($pfn! x (lambda ()
                       (funcall gfn)
                       dx))
            ($pfn! k (lambda ()
                       (funcall gfn)
                       dk))
            ($pfn! b (lambda ()
                       (funcall gfn)
                       db)))
          (let* ((dx nil)
                 (dk nil)
                 (gfn (lambda ()
                        (unless (and dx dk)
                          (setf dx ($empty ($data x)))
                          (setf dk (apply #'zeros ($size k)))
                          (nn-spatial-convolution-mm-update-grad-input ($data x)
                                                                       ($gradient result)
                                                                       dx
                                                                       ($data k)
                                                                       f
                                                                       df
                                                                       ($size k 2)
                                                                       ($size k 3)
                                                                       dw dh pw ph)
                          (nn-spatial-convolution-mm-acc-grad-parameters ($data x)
                                                                         ($gradient result)
                                                                         dk
                                                                         nil
                                                                         f
                                                                         df
                                                                         ($size k 2)
                                                                         ($size k 3)
                                                                         dw dh pw ph
                                                                         1)))))
            ($pfn! x (lambda ()
                       (funcall gfn)
                       dx))
            ($pfn! k (lambda ()
                       (funcall gfn)
                       dk))))
      result)))

(defmethod $maxpool2d ((x tensor) kw kh &optional (dw 1) (dh 1) (pw 0) (ph 0) ceilp)
  (let ((out ($empty x))
        (indices (tensor.long)))
    (nn-spatial-max-pooling-update-output x out indices kw kh dw dh pw ph ceilp)
    out))

(defmethod $maxpool2d ((x node) kw kh &optional (dw 1) (dh 1) (pw 0) (ph 0) ceilp)
  (let ((out ($empty ($data x)))
        (indices (tensor.long)))
    (nn-spatial-max-pooling-update-output ($data x) out indices kw kh dw dh pw ph ceilp)
    (let ((result (node out)))
      (setf ($name result) "MAXPOOL2D")
      ($gp! result x)
      ($pfn! x (lambda ()
                 (let ((dx ($empty ($data x))))
                   (nn-spatial-max-pooling-update-grad-input ($data x)
                                                             ($gradient result)
                                                             dx
                                                             indices
                                                             kw kh dw dh pw ph
                                                             ceilp)
                   dx)))
      result)))

(defmethod $avgpool2d ((x tensor) kw kh &optional (dw 1) (dh 1) (pw 0) (ph 0) ceilp (count-pad-p t))
  (let ((out ($empty x)))
    (nn-spatial-average-pooling-update-output x out kw kh dw dh pw ph ceilp count-pad-p)
    out))

(defmethod $avgpool2d ((x node) kw kh &optional (dw 1) (dh 1) (pw 0) (ph 0) ceilp (count-pad-p t))
  (let ((out ($empty ($data x))))
    (nn-spatial-average-pooling-update-output ($data x) out kw kh dw dh pw ph ceilp count-pad-p)
    (let ((result (node out)))
      (setf ($name result) "AVGPOOL2D")
      ($gp! result x)
      ($pfn! x (lambda ()
                 (let ((dx ($empty ($data x))))
                   (nn-spatial-average-pooling-update-grad-input ($data x)
                                                                 ($gradient result)
                                                                 dx
                                                                 kw kh dw dh pw ph
                                                                 ceilp
                                                                 count-pad-p)
                   dx)))
      result)))

(defmethod $conv2 ((x node) (k node) &optional (type :valid))
  (let ((result (node ($conv2 ($data x) ($data k) type))))
    (setf ($name result) "CONV2")
    ($gp! result x k)
    ($pfn! x (lambda () ($xcorr2 ($gradient result) ($data k) :full)))
    ($pfn! k (lambda () ($conv2 ($data x) ($gradient result))))
    result))
