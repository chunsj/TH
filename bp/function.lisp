(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defmethod $abs ((x node))
  (let ((out ($empty ($data x))))
    (nn-abs-update-output ($data x) out)
    (node out
          :name :abs
          :link (link (to x (let ((d ($empty ($data x))))
                                      (nn-abs-update-grad-input ($data x) gv d)
                                      d))))))

(defmethod $acos ((x node))
  (node ($acos ($data x))
        :name :acos
        :link (link (to x ($mul! ($div -1 ($sqrt! ($sub 1 ($expt ($data x) 2)))) gv)))))

(defmethod $asin ((x node))
  (node ($asin ($data x))
        :name :asin
        :link (link
                (to x ($mul! ($neg! ($sqrt! ($cinv! ($neg! ($sub! ($expt ($data x) 2) 1)))))
                                 gv)))))

(defmethod $atan ((x node))
  (node ($atan ($data x))
        :name :atan
        :link (link (to x ($mul! ($cinv! ($add! ($expt ($data x) 2) 1)) gv)))))

(defmethod $atan2 ((y node) (x node))
  (node ($atan2 ($data y) ($data x))
        :name :atan2
        :link (link
                (to y (let ((xd ($data x))
                                (yd ($data y)))
                            ($mul! ($mul! ($cinv! ($add! ($expt xd 2) ($expt yd 2))) xd) gv)))
                (to x (let ((xd ($data x))
                                (yd ($data y)))
                            ($neg! ($mul! ($mul! ($cinv! ($add! ($expt xd 2) ($expt yd 2))) yd)
                                          gv)))))))

(defmethod $cos ((x node))
  (node ($cos ($data x))
        :name :cos
        :link (link (to x ($mul! ($neg! ($sin ($data x))) gv)))))

(defmethod $cosh ((x node))
  (node ($cosh ($data x))
        :name :cosh
        :link (link (to x ($mul! ($sinh ($data x)) gv)))))

(defmethod $exp ((x node))
  (node ($exp ($data x))
        :name :exp
        :link (link (to x ($mul dv gv)))))

(defmethod $expt ((a node) (b node))
  (node ($expt ($data a) ($data b))
        :name :expt
        :link (link
                (to a ($mul! ($mul gv ($data b)) ($expt ($data a) ($- ($data b) 1))))
                (to b ($mul! ($mul! ($log ($data a)) ($expt ($data a) ($data b))) gv)))))

(defmethod $expt ((a node) (b tensor))
  (node ($expt ($data a) b)
        :name :expt
        :link (link (to a ($mul! ($mul! ($expt ($data a) ($- b 1)) gv) b)))))

(defmethod $expt ((a node) (b number))
  (node ($expt ($data a) b)
        :name :expt
        :link (link (to a ($mul! ($mul! ($expt ($data a) (- b 1)) gv) b)))))

(defmethod $expt ((a tensor) (b node))
  (node ($expt a ($data b))
        :name :expt
        :link (link (to b ($mul! ($mul! ($log a) ($expt a ($data b))) gv)))))

(defmethod $expt ((a number) (b node))
  (node ($expt a ($data b))
        :name :expt
        :link (link (to b ($mul! ($mul! ($expt a ($data b)) (log a)) gv)))))

(defun dlog (x) ($cinv x))

(defmethod $log ((x node))
  (node ($log ($data x))
        :name :log
        :link (link (to x ($mul! (dlog ($data x)) gv)))))

(defmethod $sin ((x node))
  (node ($sin ($data x))
        :name :sin
        :link (link (to x ($mul! ($cos ($data x)) gv)))))

(defun dsigmoid (s) ($mul! ($sub 1 s) s))

(defmethod $sigmoid ((x node))
  (node ($sigmoid ($data x))
        :name :sigmoid
        :link (link (to x ($mul! (dsigmoid dv) gv)))))

(defmethod $sinh ((x node))
  (node ($sinh ($data x))
        :name :sinh
        :link (link (to x ($mul! ($cosh ($data x)) gv)))))

(defmethod $sqrt ((x node))
  (node ($sqrt ($data x))
        :name :sqrt
        :link (link (to x ($div! ($mul gv 0.5) dv)))))

(defmethod $tan ((x node))
  (node ($tan ($data x))
        :name :tan
        :link (link (to x ($mul! ($expt! ($cos ($data x)) 2) gv)))))

(defun dtanh (s) ($neg! ($sub! ($mul s s) 1)))

(defmethod $tanh ((x node))
  (node ($tanh ($data x))
        :name :tanh
        :link (link (to x ($mul! (dtanh dv) gv)))))

(defmethod $sign ((x node))
  (node ($sign ($data x))
        :name :sign
        :link (link (to x ($zero gv)))))

(defgeneric $relu (x) (:documentation "RELU activation function."))
(defgeneric $lrelu (x &optional nv) (:documentation "Leaky RELU activation function."))
(defgeneric $elu (x &optional α) (:documentation "Exponential LU activation function."))
(defgeneric $selu (x) (:documentation "Scaled exponential LU activiation function."))
(defgeneric $softmax (x) (:documentation "Softmax function."))
(defgeneric $logsoftmax (x) (:documentation "Log softmax function."))

(defmethod $relu ((x number)) (max 0 x))

(defmethod $relu ((x tensor))
  (let ((output ($empty x)))
    (nn-threshold-update-output x output 0 0 nil)
    output))

(defun drelu (input gradient)
  (let ((dinput ($empty input)))
    (nn-threshold-update-grad-input input gradient dinput 0 0 nil)
    dinput))

(defmethod $relu ((x node))
  (node ($relu ($data x))
        :name :relu
        :link (link (to x (drelu ($data x) gv)))))

(defmethod $lrelu ((x number) &optional (nv 0.01)) (max (* nv x) x))

(defmethod $lrelu ((x tensor) &optional (nv 0.01))
  (let ((output ($empty x)))
    (nn-leaky-relu-update-output x output nv nil)
    output))

(defun dlrelu (input gradient &optional (nv 0.01))
  (let ((dinput ($empty input)))
    (nn-leaky-relu-update-grad-input input gradient dinput nv nil)
    dinput))

(defmethod $lrelu ((x node) &optional (nv 0.01))
  (node ($lrelu ($data x) nv)
        :name :lrelu
        :link (link (to x (dlrelu ($data x) gv nv)))))

(defmethod $elu ((x number) &optional (α 1))
  (if (<= x 0)
      (* α (- (exp x) 1))
      x))

(defmethod $elu ((x tensor) &optional (α 1))
  (let ((output ($empty x)))
    (nn-elu-update-output x output α nil)
    output))

(defun delu (input output gradient &optional (alpha 1))
  (let ((dinput ($empty output)))
    (nn-elu-update-grad-input input gradient dinput output alpha nil)
    dinput))

(defmethod $elu ((x node) &optional (α 1))
  (node ($elu ($data x) α)
        :name :elu
        :link (link (to x (delu ($data x) dv gv α)))))

(defmethod $selu ((x number))
  (let ((alpha 1.6732632423543772848170429916717)
        (scale 1.0507009873554804934193349852946))
    (* ($elu x alpha) scale)))

(defmethod $selu ((x tensor))
  (let ((alpha 1.6732632423543772848170429916717)
        (scale 1.0507009873554804934193349852946))
    ($* ($elu x alpha) scale)))

(defmethod $selu ((x node))
  (let ((alpha 1.6732632423543772848170429916717)
        (scale 1.0507009873554804934193349852946))
    ($* ($elu x alpha) scale)))

(defmethod $softmax ((x tensor))
  (let ((output ($empty x)))
    (nn-softmax-update-output x output)
    output))

(defun dsoftmax (input output gradient)
  (let ((dinput ($empty input)))
    (nn-softmax-update-grad-input input gradient dinput output)
    dinput))

(defmethod $softmax ((x node))
  (node ($softmax ($data x))
        :name :softmax
        :link (link (to x (dsoftmax ($data x) dv gv)))))

(defmethod $logsoftmax ((x tensor))
  (let ((output ($empty x)))
    (nn-log-softmax-update-output x output)
    output))

(defun dlogsoftmax (input output gradient)
  (let ((dinput ($empty input)))
    (nn-log-softmax-update-grad-input input gradient dinput output)
    dinput))

(defmethod $logsoftmax ((x node))
  (node ($logsoftmax ($data x))
        :name :logsoftmax
        :link (link (to x (dlogsoftmax ($data x) dv gv)))))
