(defpackage :pymc-like
  (:use #:common-lisp
        #:mu
        #:th
        #:th.distributions))

(in-package :pymc-like)

(defvar *disasters* (tensor.int '(4 5 4 0 1 4 3 4 0 6 3 3 4 0 2 6
                                  3 3 5 4 5 3 1 4 4 1 5 5 3 4 2 5
                                  2 2 3 4 2 1 3 2 2 1 1 1 1 3 0 0
                                  1 0 1 1 0 0 3 1 0 3 2 2 0 1 1 1
                                  0 1 0 1 0 0 0 2 1 0 0 0 1 1 0 2
                                  3 3 1 1 2 1 1 1 1 2 4 2 0 0 1 4
                                  0 0 0 1 0 0 0 0 0 1 0 0 1 0 1)))

(defgeneric $value (rv))
(defgeneric $logp (rv))

(defmethod $value ((rv T)) rv)

(defclass r/variable ()
  ((value :initform nil)
   (observedp :initform nil)))

(defmethod $value ((rv r/variable))
  (with-slots (value) rv
    value))

(defmethod print-object ((rv r/variable) stream)
  (with-slots (observedp value) rv
    (format stream "~A~A" (if (not observedp) "?" "") value)))

(defclass r/discrete-uniform (r/variable)
  ((lower :initform 0D0)
   (upper :initform 1D0)))

(defun r/discrete-uniform (&key (lower 0D0) (upper 1D0))
  (let ((l lower)
        (u upper)
        (n (make-instance 'r/discrete-uniform)))
    (with-slots (lower upper value) n
      (setf lower l
            upper u
            value (+ lower (1- ($sample/dice 1 (1+ (- upper lower)) )))))
    n))

(defmethod $logp ((rv r/discrete-uniform))
  (with-slots (value lower upper) rv
    ($ll/uniform value lower upper)))

(defclass r/exponential (r/variable)
  ((rate :initform 1D0)))

(defun r/exponential (&key (rate 1D0))
  (let ((r rate)
        (n (make-instance 'r/exponential)))
    (with-slots (rate value) n
      (setf rate r
            value ($sample/exponential 1 rate)))
    n))

(defmethod $logp ((rv r/exponential))
  (with-slots (value rate) rv
    ($ll/exponential value rate)))

(defclass r/fn (r/variable)
  ((fn :initform nil)))

(defun r/fn (function)
  (let ((n (make-instance 'r/fn)))
    (with-slots (fn value) n
      (setf fn function
            value (funcall fn)))
    n))

(defclass r/poisson (r/variable)
  ((rate :initform 1D0)))

(defun r/poisson (&key (rate 1D0) observation)
  (let ((r rate)
        (n (make-instance 'r/poisson)))
    (with-slots (rate value observedp) n
      (setf rate r
            value (if observation
                      observation
                      ($sample/poisson 1 rate))
            observedp (if observation T nil)))
    n))

(defmethod $logp ((rv r/poisson))
  (with-slots (value rate) rv
    (let* ((rates ($value rate))
           (logps (tensor ($count rates))))
      (loop :for i :from 0 :below ($count rates)
            :for r = ($ rates i)
            :for v = ($ value i)
            :do (setf ($ logps i) ($ll/poisson v r)))
      ($sum logps))))

(let ((switch-point (r/discrete-uniform :lower 0 :upper (1- ($count *disasters*))))
      (early-mean (r/exponential :rate 1))
      (late-mean (r/exponential :rate 1)))
  (let ((rate (r/fn (lambda ()
                      (let ((rates (tensor ($count *disasters*))))
                        (loop :for i :from 0 :below ($value switch-point)
                              :do (setf ($ rates i) ($value early-mean)))
                        (loop :for i :from ($value switch-point) :below ($count rates)
                              :do (setf ($ rates i) ($value late-mean)))
                        rates)))))
    (let ((disasters (r/poisson :rate rate :observation *disasters*)))
      ($logp disasters))))

(let ((switch-point (r/discrete-uniform :lower 0 :upper (1- ($count *disasters*))))
      (early-mean (r/exponential :rate 1))
      (late-mean (r/exponential :rate 1)))
  (let ((rate (r/fn (lambda ()
                      (let ((rates (tensor ($count *disasters*))))
                        (loop :for i :from 0 :below switch-point
                              :do (setf ($ rates i) early-mean))
                        (loop :for i :from s :below ($count rates)
                              :do (setf ($ rate i) late-mean))
                        rates)))))
    (let ((disasters (r/poisson :rate rate :observation *disasters*)))
      disasters)))

(let ((switch-point (r/discrete-uniform :lower 0 :upper (1- ($count *disasters*))))
      (early-mean (r/exponential :rate 1))
      (late-mean (r/exponential :rate 1)))
  (let ((disasters-early (subseq *disasters* 0 ($value switch-point)))
        (disasters-late (subseq *disasters* (1- ($value switch-point)))))
    (let ((d1 (r/poisson :rate early-mean :observation disasters-early))
          (d2 (r/poisson :rate late-mean :observation disasters-late)))
      (list d1 d2))))
