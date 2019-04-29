(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defmethod $transpose ((x parameter) &optional dimension0 dimension1)
  ($operation ($transpose ($data x) dimension0 dimension1)
              :creators (list x)
              :name :transpose
              :bfn (lambda (self gradient)
                     ($bp! x ($transpose gradient dimension0 dimension1) self))))

(defmethod $view ((a parameter) &rest sizes)
  ($operation (apply #'$view ($data a) sizes)
              :creators (list a)
              :name :view
              :bfn (lambda (self gradient)
                     ($bp! a ($view gradient a) self))))

(defmethod $expand ((a parameter) size)
  ($operation ($expand ($data a) size)
              :creators (list a)
              :name :expand
              :bfn (lambda (self gradient)
                     ($bp! a
                           (let ((asize ($size a))
                                 (out gradient))
                             (loop :for dim :from 0 :below ($count asize)
                                   :for sz = ($ asize dim)
                                   :do (when (eq sz 1)
                                         (setf out ($sum out dim))))
                             out)
                           self))))

(defmethod $reshape ((x parameter) &rest sizes)
  ($operation (apply #'$reshape ($data x) sizes)
              :creators (list x)
              :name :reshape
              :bfn (lambda (self gradient)
                     ($bp! x ($view gradient x) self))))

(defmethod $index ((x parameter) dimension (indices list))
  ($operation ($index ($data x) dimension indices)
              :creators (list x)
              :name :index
              :bfn (lambda (self gradient)
                     ($bp! x
                           (let* ((out ($zero ($data x)))
                                  (outs ($index out dimension indices)))
                             (setf ($index out dimension indices)
                                   (apply #'$reshape gradient ($size outs)))
                             out)
                           self))))

(defmethod $index ((x parameter) dimension (index number))
  ($operation ($index ($data x) dimension index)
              :creators (list x)
              :name :index
              :bfn (lambda (self gradient)
                     ($bp! x
                           (let* ((indices (list index))
                                  (out ($zero ($data x)))
                                  (outs ($index out dimension indices)))
                             (setf ($index out dimension indices)
                                   (apply #'$reshape gradient ($size outs)))
                             out)
                           self))))

(defmethod $inverse ((a parameter))
  ($operation ($inverse ($data a))
              :creators (list a)
              :name :inverse
              :bfn (lambda (self gradient)
                     (let ((tnode ($transpose self)))
                       ($bp! a ($mm ($mm tnode ($neg gradient)) tnode) self)))))

(defmethod $clone ((x parameter))
  ($operation ($clone ($data x))
              :creators (list x)
              :name :clone
              :bfn (lambda (self gradient) ($bp! x gradient self))))

(defmethod $cat ((x parameter) (y parameter) &optional (dimension 0))
  ($operation ($cat ($data x) ($data y) dimension)
              :creators (list x y)
              :name :cat
              :bfn (lambda (self gradient)
                     ($bp! x ($narrow gradient dimension 0 ($size x 1)) self)
                     ($bp! y ($narrow gradient dimension ($size x 1) ($size y 1)) self))))

(defmethod $cat ((x parameter) (y tensor) &optional (dimension 0))
  ($operation ($cat ($data x) y dimension)
              :creators (list x)
              :name :cat
              :bfn (lambda (self gradient)
                     ($bp! x ($narrow gradient dimension 0 ($size x 1)) self))))

(defmethod $cat ((x tensor) (y parameter) &optional (dimension 0))
  ($operation ($cat x ($data y) dimension)
              :creators (list y)
              :name :cat
              :bfn (lambda (self gradient)
                     ($bp! y ($narrow gradient dimension ($size x 1) ($size y 1)) self))))

(defmethod $concat ((x parameter) xs &rest others)
  (let ((pd ($last others)))
    (if (numberp pd)
        (let ((dimension pd)
              (xs (cons x (cons xs (butlast others)))))
          (reduce (lambda (r n) ($cat r n dimension)) xs))
        (let ((dimension 0)
              (xs (cons x (cons xs others))))
          (reduce (lambda (r n) ($cat r n dimension)) xs)))))
