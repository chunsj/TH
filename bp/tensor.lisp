(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defmethod $ ((x parameter) location &rest others-and-default)
  ($operation (apply #'$ ($data x) (cons location others-and-default))
              :creators (list x)
              :name :get
              :bfn (lambda (self gradient xd)
                     ($bp! x
                           (let ((z ($zero xd))
                                 (locs (cons location others-and-default)))
                             (if (= ($count locs) ($ndim z))
                                 (setf (apply #'$ z locs) gradient)
                                 ($copy! (apply #'$ z (cons location others-and-default)) gradient))
                             z)
                           self))))

(defmethod (setf $) (value (x parameter) location &rest others)
  ($operation (let ((nx ($clone ($data x))))
                (setf (apply #'$ nx (cons location others)) value)
                nx)
              :creators (list x)
              :name :set
              :bfn (lambda (self gradient xd)
                     ($bp! x
                           (let ((ng ($clone gradient)))
                             (setf (apply #'$ ng (cons location others)) 0)
                             ng)
                           self)
                     (when (typep value 'parameter)
                       ($bp! value
                             (let ((gk (apply #'$ gradient (cons location others))))
                               (if (numberp gk)
                                   gk
                                   ($clone gk)))
                             self)))))

(defmethod $index ((x parameter) dimension (indices list))
  ($operation ($index ($data x) dimension indices)
              :creators (list x)
              :name :index
              :bfn (lambda (self gradient xd)
                     ($bp! x
                           (let* ((out ($zero xd))
                                  (outs ($index out dimension indices)))
                             (setf ($index out dimension indices)
                                   (apply #'$reshape gradient ($size outs)))
                             out)
                           self))))

(defmethod $index ((x parameter) dimension (index number))
  ($operation ($index ($data x) dimension index)
              :creators (list x)
              :name :index
              :bfn (lambda (self gradient xd)
                     ($bp! x
                           (let* ((indices (list index))
                                  (out ($zero xd))
                                  (outs ($index out dimension indices)))
                             (setf ($index out dimension indices)
                                   (apply #'$reshape gradient ($size outs)))
                             out)
                           self))))

(defmethod $view ((a parameter) &rest sizes)
  ($operation (apply #'$view ($data a) sizes)
              :creators (list a)
              :name :view
              :bfn (lambda (self gradient ad)
                     ($bp! a ($view gradient ad) self))))

(defmethod $reshape ((x parameter) &rest sizes)
  ($operation (apply #'$reshape ($data x) sizes)
              :creators (list x)
              :name :reshape
              :bfn (lambda (self gradient xd)
                     ($bp! x ($view gradient xd) self))))

(defmethod $expand ((a parameter) size)
  ($operation ($expand ($data a) size)
              :creators (list a)
              :name :expand
              :bfn (lambda (self gradient ad)
                     ($bp! a
                           (let ((asize ($size ad))
                                 (out gradient))
                             (loop :for dim :from 0 :below ($count asize)
                                   :for sz = ($ asize dim)
                                   :do (when (eq sz 1)
                                         (setf out ($sum out dim))))
                             out)
                           self))))

(defmethod $transpose ((x parameter) &optional dimension0 dimension1)
  ($operation ($transpose ($data x) dimension0 dimension1)
              :creators (list x)
              :name :transpose
              :bfn (lambda (self gradient xd)
                     (declare (ignore xd))
                     ($bp! x ($transpose gradient dimension0 dimension1) self))))

(defmethod $inverse ((a parameter))
  ($operation ($inverse ($data a))
              :creators (list a)
              :name :inverse
              :bfn (lambda (self gradient ad)
                     (declare (ignore ad))
                     (let ((tnode ($transpose ($data self))))
                       ($bp! a ($neg! ($mm ($mm tnode gradient) tnode)) self)))))

(defmethod $clone ((x parameter))
  ($operation ($clone ($data x))
              :creators (list x)
              :name :clone
              :bfn (lambda (self gradient xd)
                     (declare (ignore xd))
                     ($bp! x gradient self))))

(defmethod $cat ((x parameter) (y parameter) &optional (dimension 0))
  ($operation ($cat ($data x) ($data y) dimension)
              :creators (list x y)
              :name :cat
              :bfn (lambda (self gradient xd yd)
                     ($bp! x ($narrow gradient dimension 0 ($size xd 1)) self)
                     ($bp! y ($narrow gradient dimension ($size xd 1) ($size yd 1)) self))))

(defmethod $cat ((x parameter) (y tensor) &optional (dimension 0))
  ($operation ($cat ($data x) y dimension)
              :creators (list x)
              :name :cat
              :bfn (lambda (self gradient xd)
                     ($bp! x ($narrow gradient dimension 0 ($size xd 1)) self))))

(defmethod $cat ((x tensor) (y parameter) &optional (dimension 0))
  ($operation ($cat x ($data y) dimension)
              :creators (list y)
              :name :cat
              :bfn (lambda (self gradient yd)
                     ($bp! y ($narrow gradient dimension ($size x 1) ($size yd 1)) self))))

(defmethod $concat ((x parameter) xs &rest others)
  (let ((pd ($last others)))
    (if (numberp pd)
        (let ((dimension pd)
              (xs (cons x (cons xs (butlast others)))))
          (reduce (lambda (r n) ($cat r n dimension)) xs))
        (let ((dimension 0)
              (xs (cons x (cons xs others))))
          (reduce (lambda (r n) ($cat r n dimension)) xs)))))

(defgeneric $conv2d (x k &optional b dw dh pw ph)
  (:documentation "Performs convolution of x with kernel k and bias b which is optional."))
(defgeneric $maxpool2d (x kw kh &optional dw dh pw ph ceilp)
  (:documentation "Performs max pooling over x."))
(defgeneric $avgpool2d (x kw kh &optional dw dh pw ph ceilp count-pad-p)
  (:documentation "Performs average pooling over x."))

(defgeneric $dlconv2d (x k &optional b dw dh pw ph dlw dlh)
  (:documentation "Performs dilated convoution of x with kernel k and bias b which is optional."))
(defgeneric $dlmaxpool2d (x kw kh &optional dw dh pw ph dlw dlh ceilp)
  (:documentation "Performs dilated max pooling over x."))

(defgeneric $dconv2d (x w &optional b dw dh pw ph aw ah)
  (:documentation "Performs deconvolution using w weight, b bias and others."))

(defmethod $conv2d ((x tensor) (k tensor) &optional b (dw 1) (dh 1) (pw 0) (ph 0))
  (let ((out ($empty x))
        (f ($empty x))
        (df ($empty x)))
    (if b
        (nn-spatial-convolution-mm-update-output x out k b f df ($size k 2) ($size k 3)
                                                 dw dh pw ph)
        (nn-spatial-convolution-mm-update-output x out k nil f df ($size k 2) ($size k 3)
                                                 dw dh pw ph))
    out))
(defun conv2d-with-b (x xp k kp b bp dw dh pw ph)
  (let* ((xd (if xp ($data x) x))
         (kd (if kp ($data k) k))
         (bd (if bp ($data b) b))
         (out ($empty xd))
         (f ($empty xd))
         (df ($empty xd)))
    (nn-spatial-convolution-mm-update-output xd out kd bd f df ($size k 2) ($size k 3)
                                             dw dh pw ph)
    ($operation out
                :creators (append '()
                                  (when xp (list x))
                                  (when kp (list k))
                                  (when bp (list b)))
                :name :conv2d
                :bfn (lambda (self gradient &rest ignored)
                       (declare (ignore ignored))
                       (let ((dx ($empty xd))
                             (dk ($zero kd))
                             (db ($zero bd)))
                         (nn-spatial-convolution-mm-update-grad-input xd gradient
                                                                      dx kd f df
                                                                      ($size k 2) ($size k 3)
                                                                      dw dh pw ph)
                         (nn-spatial-convolution-mm-acc-grad-parameters xd gradient
                                                                        dk db f df
                                                                        ($size k 2) ($size k 3)
                                                                        dw dh pw ph 1D0)
                         (when xp ($bp! x dx self))
                         (when kp ($bp! k dk self))
                         (when bp ($bp! b db self)))))))
(defun conv2d-without-b (x xp k kp dw dh pw ph)
  (let* ((xd (if xp ($data x) x))
         (kd (if kp ($data k) k))
         (out ($empty xd))
         (f ($empty xd))
         (df ($empty xd)))
    (nn-spatial-convolution-mm-update-output xd out kd nil f df ($size k 2) ($size k 3)
                                             dw dh pw ph)
    ($operation out
                :creators (append '()
                                  (when xp (list x))
                                  (when kp (list k)))
                :name :conv2d
                :bfn (lambda (self gradient &rest ignored)
                       (declare (ignore ignored))
                       (let ((dx ($empty xd))
                             (dk ($zero kd)))
                         (nn-spatial-convolution-mm-update-grad-input xd gradient
                                                                      dx kd f df
                                                                      ($size k 2) ($size k 3)
                                                                      dw dh pw ph)
                         (nn-spatial-convolution-mm-acc-grad-parameters xd gradient
                                                                        dk nil f df
                                                                        ($size k 2) ($size k 3)
                                                                        dw dh pw ph 1D0)
                         (when xp ($bp! x dx self))
                         (when kp ($bp! k dk self)))))))
(defmethod $conv2d ((x parameter) (k parameter) &optional b (dw 1) (dh 1) (pw 0) (ph 0))
  (if b
      (conv2d-with-b x T k T b (typep b 'parameter) dw dh pw ph)
      (conv2d-without-b x T k T dw dh pw ph)))
(defmethod $conv2d ((x tensor) (k parameter) &optional b (dw 1) (dh 1) (pw 0) (ph 0))
  (if b
      (conv2d-with-b x nil k T b (typep b 'parameter) dw dh pw ph)
      (conv2d-without-b x nil k T dw dh pw ph)))
(defmethod $conv2d ((x parameter) (k tensor) &optional b (dw 1) (dh 1) (pw 0) (ph 0))
  (if b
      (conv2d-with-b x T k nil b (typep b 'parameter) dw dh pw ph)
      (conv2d-without-b x T k nil dw dh pw ph)))

(defmethod $maxpool2d ((x tensor) kw kh &optional (dw 1) (dh 1) (pw 0) (ph 0) ceilp)
  (let ((out ($empty x))
        (indices (tensor.long)))
    (nn-spatial-max-pooling-update-output x out indices kw kh dw dh pw ph ceilp)
    out))
(defmethod $maxpool2d ((x parameter) kw kh &optional (dw 1) (dh 1) (pw 0) (ph 0) ceilp)
  (let ((out ($empty ($data x)))
        (indices (tensor.long)))
    (nn-spatial-max-pooling-update-output ($data x) out indices kw kh dw dh pw ph ceilp)
    ($operation out
                :creators (list x)
                :name :maxpool2d
                :bfn (lambda (self gradient xd)
                       ($bp! x
                             (let ((dx ($empty xd)))
                               (nn-spatial-max-pooling-update-grad-input xd gradient dx
                                                                         indices
                                                                         kw kh dw dh pw ph
                                                                         ceilp)
                               dx)
                             self)))))

(defmethod $avgpool2d ((x tensor) kw kh &optional (dw 1) (dh 1) (pw 0) (ph 0) ceilp (count-pad-p t))
  (let ((out ($empty x)))
    (nn-spatial-average-pooling-update-output x out kw kh dw dh pw ph ceilp count-pad-p)
    out))
(defmethod $avgpool2d ((x parameter) kw kh &optional (dw 1) (dh 1) (pw 0) (ph 0) ceilp
                                             (count-pad-p t))
  (let ((out ($empty ($data x))))
    (nn-spatial-average-pooling-update-output x out kw kh dw dh pw ph ceilp count-pad-p)
    ($operation out
                :creators (list x)
                :name :avgpool2d
                :bfn (lambda (self gradient xd)
                       ($bp! x
                             (let ((dx ($empty xd)))
                               (nn-spatial-average-pooling-update-grad-input xd gradient dx
                                                                             kw kh dw dh pw ph
                                                                             ceilp count-pad-p)
                               dx)
                             self)))))
