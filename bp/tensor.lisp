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
