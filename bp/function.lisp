(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defmethod $abs ((x parameter))
  (let ((out ($empty ($data x))))
    (nn-abs-update-output ($data x) out)
    ($operation out
                :creators (list x)
                :name :abs
                :bfn (lambda (self gradient xd)
                       ($bp! x
                             (let ((d ($empty xd)))
                               (nn-abs-update-grad-input xd gradient d)
                               d)
                             self)))))

(defmethod $acos ((x parameter))
  ($operation ($acos ($data x))
              :creators (list x)
              :name :acos
              :bfn (lambda (self gradient xd)
                     ($bp! x ($mul! ($div -1D0 ($sqrt! ($sub 1D0 ($expt xd 2D0)))) gradient) self))))

(defmethod $asin ((x parameter))
  ($operation ($asin ($data x))
              :creators (list x)
              :name :asin
              :bfn (lambda (self gradient xd)
                     ($bp! x ($mul! ($div 1D0 ($sqrt! ($sub 1D0 ($expt xd 2D0)))) gradient) self))))

(defmethod $atan ((x parameter))
  ($operation ($atan ($data x))
              :creators (list x)
              :name :atan
              :bfn (lambda (self gradient xd)
                     ($bp! x ($mul! ($div 1D0 ($add 1D0 ($expt xd 2D0))) gradient) self))))

(defmethod $atan2 ((y parameter) (x parameter))
  ($operation ($atan2 ($data y) ($data x))
              :creators (list y x)
              :name :atan2
              :bfn (lambda (self gradient yd xd)
                     ($bp! y ($mul! ($div xd ($add! ($expt xd 2D0) ($expt yd 2D0))) gradient) self)
                     ($bp! x ($mul! ($div ($neg yd) ($add! ($expt xd 2D0) ($expt yd 2D0))) gradient)
                           self))))

(defmethod $atan2 ((y t) (x parameter))
  ($operation ($atan2 y ($data x))
              :creators (list x)
              :name :atan2
              :bfn (lambda (self gradient xd)
                     ($bp! x ($mul! ($div ($neg y) ($add! ($expt xd 2D0) ($expt y 2D0))) gradient)
                           self))))

(defmethod $atan2 ((y parameter) (x t))
  ($operation ($atan2 ($data y) x)
              :creators (list y)
              :name :atan2
              :bfn (lambda (self gradient yd)
                     ($bp! y ($mul! ($div x ($add! ($expt x 2D0) ($expt yd 2D0))) gradient) self))))

(defmethod $cos ((x parameter))
  ($operation ($cos ($data x))
              :creators (list x)
              :name :cos
              :bfn (lambda (self gradient xd) ($bp! x ($mul! ($neg! ($sin xd)) gradient) self))))

(defmethod $cosh ((x parameter))
  ($operation ($cosh ($data x))
              :creators (list x)
              :name :cosh
              :bfn (lambda (self gradient xd) ($bp! x ($mul! ($sinh xd) gradient) self))))

(defmethod $exp ((x parameter))
  ($operation ($exp ($data x))
              :creators (list x)
              :name :exp
              :bfn (lambda (self gradient xd)
                     (declare (ignore xd))
                     ($bp! x ($mul ($data self) gradient) self))))

(defmethod $expt ((a parameter) (b parameter))
  ($operation ($expt ($data a) ($data b))
              :creators (list a b)
              :name :expt
              :bfn (lambda (self gradient ad bd)
                     ($bp! a ($mul! ($mul gradient bd) ($expt ad ($- bd 1D0))) self)
                     ($bp! b ($mul! ($mul! ($log ad) ($expt ad bd)) gradient) self))))

(defmethod $expt ((a t) (b parameter))
  ($operation ($expt a ($data b))
              :creators (list b)
              :name :expt
              :bfn (lambda (self gradient bd)
                     ($bp! b ($mul! ($mul! ($log a) ($expt a bd)) gradient) self))))

(defmethod $expt ((a parameter) (b t))
  ($operation ($expt ($data a) b)
              :creators (list a)
              :name :expt
              :bfn (lambda (self gradient ad)
                     ($bp! a ($mul! ($mul gradient b) ($expt ad ($- b 1D0))) self))))

(defmethod $log ((x parameter))
  ($operation ($log ($data x))
              :creators (list x)
              :name :log
              :bfn (lambda (self gradient xd) ($bp! x ($mul! ($div 1D0 xd) gradient) self))))

(defmethod $sin ((x parameter))
  ($operation ($sin ($data x))
              :creators (list x)
              :name :sin
              :bfn (lambda (self gradient xd) ($bp! x ($mul! ($cos xd) gradient) self))))

(defun dsigmoid (s) ($mul! ($sub 1D0 s) s))

(defmethod $sigmoid ((x parameter))
  ($operation ($sigmoid ($data x))
              :creators (list x)
              :name :sigmoid
              :bfn (lambda (self gradient xd)
                     (declare (ignore xd))
                     ($bp! x ($mul! (dsigmoid ($data self)) gradient) self))))

(defmethod $sinh ((x parameter))
  ($operation ($sinh ($data x))
              :creators (list x)
              :name :sinh
              :bfn (lambda (self gradient xd)
                     ($bp! x ($mul! ($cosh xd) gradient) self))))

(defmethod $sqrt ((x parameter))
  ($operation ($sqrt ($data x))
              :creators (list x)
              :name :sqrt
              :bfn (lambda (self gradient xd)
                     (declare (ignore xd))
                     ($bp! x ($div! ($mul gradient 0.5D0) ($data self)) self))))

(defmethod $tan ((x parameter))
  ($operation ($tan ($data x))
              :creators (list x)
              :name :tan
              :bfn (lambda (self gradient xd)
                     ($bp! x ($mul! ($expt ($cos xd) 2D0) gradiennt) self))))

(defun dtanh (s) ($sub 1D0 ($mul s s)))

(defmethod $tanh ((x parameter))
  ($operation ($tanh ($data x))
              :creators (list x)
              :name :tanh
              :bfn (lambda (self gradient xd)
                     (declare (ignore xd))
                     ($bp! x ($mul! (dtanh ($data self)) gradient) self))))

(defmethod $sign ((x parameter))
  ($operation ($sign ($data x))
              :creators (list x)
              :name :sign
              :bfn (lambda (self gradient xd)
                     (declare (ignore xd))
                     ($bp! x ($zero gradient) self))))

(defgeneric $relu (x))
(defgeneric $lrelu (x &optional nv))
(defgeneric $elu (x &optional α))
(defgeneric $selu (x))
(defgeneric $softmax (x))
(defgeneric $lsoftmax (x))

(defmethod $relu ((x number)) (max 0D0 x))
(defmethod $relu ((x tensor))
  (let ((output ($empty x)))
    (nn-threshold-update-output x output 0D0 0D0 nil)
    output))
(defun drelu (input gradient)
  (let ((dinput ($empty input)))
    (nn-threshold-update-grad-input input gradient dinput 0D0 0D0 nil)
    dinput))
(defmethod $relu ((x parameter))
  ($operation ($relu ($data x))
              :creators (list x)
              :name :relu
              :bfn (lambda (self gradient xd)
                     ($bp! x (drelu xd gradient) self))))

(defmethod $lrelu ((x number) &optional (nv 0.01D0)) (max (* nv x) x))
(defmethod $lrelu ((x tensor) &optional (nv 0.01D0))
  (let ((output ($empty x)))
    (nn-leaky-relu-update-output x output nv nil)
    output))
(defun dlrelu (input gradient nv)
  (let ((dinput ($empty input)))
    (nn-leaky-relu-update-grad-input input gradient dinput nv nil)
    dinput))
(defmethod $lrelu ((x parameter) &optional (nv 0.01D0))
  ($operation ($relu ($data x))
              :creators (list x)
              :name :lrelu
              :bfn (lambda (self gradient xd)
                     ($bp! x (dlrelu xd gradient nv) self))))

(defmethod $elu ((x number) &optional (α 1D0))
  (if (<= x 0D0)
      (* α (- (exp x) 1D0))
      x))
(defmethod $elu ((x tensor) &optional (α 1D0))
  (let ((output ($empty x)))
    (nn-elu-update-output x output α nil)
    output))
(defun delu (input output gradient α)
  (let ((dinput ($empty output)))
    (nn-elu-update-grad-input input gradient dinput output α nil)
    dinput))
(defmethod $elu ((x parameter) &optional (α 1))
  ($operation ($relu ($data x))
              :creators (list x)
              :name :elu
              :bfn (lambda (self gradient xd)
                     ($bp! x (delu xd ($data self) gradient α)self))))

(defmethod $selu ((x number))
  (let ((alpha 1.6732632423543772848170429916717)
        (scale 1.0507009873554804934193349852946))
    (* ($elu x alpha) scale)))
(defmethod $selu ((x tensor))
  (let ((alpha 1.6732632423543772848170429916717)
        (scale 1.0507009873554804934193349852946))
    ($mul! ($elu x alpha) scale)))
(defmethod $selu ((x parameter))
  (let ((alpha 1.6732632423543772848170429916717)
        (scale 1.0507009873554804934193349852946))
    ($mul! ($elu x alpha) scale)))

(defmethod $softmax ((x tensor))
  (let ((output ($empty x)))
    (nn-softmax-update-output x output)
    output))
(defun dsoftmax (input output gradient)
  (let ((dinput ($empty input)))
    (nn-softmax-update-grad-input input gradient dinput output)
    dinput))
(defmethod $softmax ((x parameter))
  ($operation ($softmax ($data x))
              :creators (list x)
              :name :softmax
              :bfn (lambda (self gradient xd)
                     ($bp! x (dsoftmax xd ($data self) gradient) self))))

(defmethod $lsoftmax ((x tensor))
  (let ((output ($empty x)))
    (nn-log-softmax-update-output x output)
    output))
(defun dlsoftmax (input output gradient)
  (let ((dinput ($empty input)))
    (nn-log-softmax-update-grad-input input gradient dinput output)
    dinput))
(defmethod $lsoftmax ((x parameter))
  ($operation ($lsoftmax ($data x))
              :creators (list x)
              :name :lsoftmax
              :bfn (lambda (self gradient xd)
                     ($bp! x (dlsoftmax xd ($data self) gradient) self))))
