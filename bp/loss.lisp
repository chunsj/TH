(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defgeneric $bce (a b) (:documentation "Binary cross entropy loss function."))
(defgeneric $mse (a b) (:documentation "Mean squared error loss function."))
(defgeneric $cee (a b) (:documentation "Cross entroy loss function."))
(defgeneric $cnll (a b) (:documentation "Class negative log likelihood loss function."))
(defgeneric $cec (a b) (:documentation "Class negative log likelihood over log softmax."))

(defmethod $bce ((a tensor) (b tensor))
  (let ((output ($resize! ($empty a) '(1))))
    (nn-bce-criterion-update-output a b output t nil)
    ($ output 0)))

(defun dbce (input target)
  (let ((dinput ($empty input)))
    (nn-bce-criterion-update-grad-input input target dinput t nil)
    dinput))

(defmethod $bce ((a node) (b node))
  (node ($bce ($data a) ($data b))
        :name :bce
        :bps (bps a (lambda (dv gv)
                      (declare (ignore dv gv))
                      (dbce ($data a) ($data b)))
                  b (lambda (dv gv)
                      (declare (ignore dv gv))
                      (dbce ($data b) ($data a))))))

(defmethod $bce ((a node) (b tensor))
  (node ($bce ($data a) b)
        :name :bce
        :bps (bps a (lambda (dv gv)
                      (declare (ignore dv gv))
                      (dbce ($data a) b)))))

(defmethod $bce ((a tensor) (b node))
  (node ($bce a ($data b))
        :name :bce
        :bps (bps b (lambda (dv gv)
                      (declare (ignore dv gv))
                      (dbce ($data b) a)))))

(defmethod $mse ((a tensor) (b tensor))
  (let ((output ($empty a)))
    (nn-mse-criterion-update-output a b output t)
    output))

(defun dmse (input target gradient)
  (let ((dinput ($empty input)))
    (nn-mse-criterion-update-grad-input input target gradient dinput t)
    dinput))

(defmethod $mse ((a node) (b node))
  (node ($mse ($data a) ($data b))
        :name :mse
        :bps (bps a (lambda (dv gv)
                      (declare (ignore dv))
                      (dmse ($data a) ($data b) gv))
                  b (lambda (dv gv)
                      (declare (ignore dv))
                      (dmse ($data b) ($data a) gv)))))

(defmethod $mse ((a node) (b tensor))
  (node ($mse ($data a) b)
        :name :mse
        :bps (bps a (lambda (dv gv)
                      (declare (ignore dv))
                      (dmse ($data a) b gv)))))

(defmethod $mse ((a tensor) (b node))
  (node ($mse a ($data b))
        :name :mse
        :bps (bps b (lambda (dv gv)
                      (declare (ignore dv))
                      (dmse ($data b) a gv)))))

(defmethod $cee ((a tensor) (b tensor))
  (let ((tiny 1D-7)
        (nbatch (if (eq 1 ($ndim a)) 1 ($size a 0))))
    (/ (- ($sum ($mul! ($log! ($add a tiny)) b))) nbatch)))

(defmethod $cee ((a node) (b node))
  (let ((tiny 1D-7)
        (nbatch (if (eq 1 ($ndim a)) 1 ($size a 0))))
    ($div ($neg ($sum ($mul ($log ($add a tiny)) b))) nbatch)))

(defmethod $cee ((a node) (b tensor))
  (let ((tiny 1D-7)
        (nbatch (if (eq 1 ($ndim a)) 1 ($size a 0))))
    ($div ($neg ($sum ($mul ($log ($add a tiny)) b))) nbatch)))

(defmethod $cee ((a tensor) (b node))
  (let ((tiny 1D-7)
        (nbatch (if (eq 1 ($ndim a)) 1 ($size a 0))))
    ($div ($neg ($sum ($mul ($log ($add a tiny)) b))) nbatch)))

;; b should be 1-d
(defmethod $cnll ((a tensor) (b tensor))
  (let ((result (zeros 1))
        (tw (ones 1)))
    (nn-class-nll-criterion-update-output a (tensor.long ($reshape b ($count b)))
                                          result t nil tw -100)
    ($ result 0)))

(defun dcnll (input target)
  (let ((dinput ($zero input)))
    (nn-class-nll-criterion-update-grad-input input (tensor.long ($reshape target ($count target)))
                                              dinput t nil
                                              (ones 1) -100)
    dinput))

(defmethod $cnll ((a node) (b tensor))
  (node ($cnll ($data a) ($data b))
        :name :cnll
        :bps (bps a (lambda (dv gv)
                      (declare (ignore dv gv))
                      (dcnll ($data a) b)))))

(defmethod $cec ((a tensor) (b tensor)) ($cnll ($logsoftmax a) b))
(defmethod $cec ((a node) (b tensor)) ($cnll ($logsoftmax a) b))
