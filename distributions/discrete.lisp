(in-package :th.distributions)

(defclass distribution/dice (distribution)
  ((n :initform 6)))

(defun distribution/dice (&optional (n 6))
  (let ((d (make-instance 'distribution/dice))
        (nin n))
    (with-slots (n) d
      (setf n nin))
    d))

(defmethod $parameters ((d distribution/dice))
  (with-slots (n) d
    (if ($parameterp n)
        (list n)
        '())))

(defmethod $parameter-names ((d distribution/dice))
  (list :n))

(defmethod $ ((d distribution/dice) name &rest ig)
  (declare (ignore ig))
  (when (eq name :n)
    (with-slots (n) d
      n)))

(defmethod (setf $) (value (d distribution/dice) name &rest ig)
  (declare (ignore ig))
  (when (eq name :n)
    (with-slots (n) d
      (setf n value))
    value))

(defmethod $sample ((d distribution/dice) &optional (nn 1))
  (when (> nn 0)
    (with-slots (n) d
      (let ((nv (round ($scalar n))))
        (cond ((eq nn 1) (1+ (random nv)))
              (T (tensor.int (loop :repeat nn :collect (1+ (random nv))))))))))

(defmethod $ll ((d distribution/dice) data)
  (with-slots (n) d
    ($ll/dice data n)))

(defclass distribution/bernoulli (distribution)
  ((p :initform 0.5)))

(defun distribution/bernoulli (&optional (p 0.5D0))
  (let ((dist (make-instance 'distribution/bernoulli))
        (pin p))
    (with-slots (p) dist
      (setf p pin))
    dist))

(defmethod $parameters ((d distribution/bernoulli))
  (with-slots (p) d
    (if ($parameterp p)
        (list p)
        '())))

(defmethod $parameter-names ((d distribution/bernoulli))
  (list :p))

(defmethod $ ((d distribution/bernoulli) name &rest others-and-default)
  (declare (ignore others-and-default))
  (when (eq name :p)
    (with-slots (p) d
      p)))

(defmethod (setf $) (value (d distribution/bernoulli) name &rest others)
  (declare (ignore others))
  (when (eq name :p)
    (with-slots (p) d
      (setf p value)
      value)))

(defmethod $sample ((d distribution/bernoulli) &optional (n 1))
  (when (> n 0)
    (with-slots (p) d
      (cond ((eq n 1) (random/bernoulli (pv p)))
            (T ($bernoulli (tensor.byte n) (pv p)))))))

(defmethod $ll ((d distribution/bernoulli) data)
  (with-slots (p) d
    ($ll/bernoulli data p)))

(defclass distribution/binomial (distribution)
  ((n :initform 1)
   (p :initform 0.5)))

(defun distribution/binomial (&optional (n 1) (p 0.5D0))
  (let ((dist (make-instance 'distribution/binomial))
        (nin n)
        (pin p))
    (with-slots (n p) dist
      (setf n nin
            p pin))
    dist))

(defmethod $parameters ((d distribution/binomial))
  (with-slots (n p) d
    (let ((ps '()))
      (when ($parameterp p) (push p ps))
      (when ($parameterp n) (push n ps))
      ps)))

(defmethod $parameter-names ((d distribution/binomial))
  (list :n :p))

(defmethod $ ((d distribution/binomial) name &rest others-and-default)
  (declare (ignore others-and-default))
  (with-slots (n p) d
    (cond ((eq name :n) n)
          ((eq name :p) p))))

(defmethod (setf $) (value (d distribution/binomial) name &rest others)
  (declare (ignore others))
  (with-slots (n p) d
    (cond ((eq name :n) (setf n value))
          ((eq name :p) (Setf p value))))
  value)

(defmethod $sample ((d distribution/binomial) &optional (n 1))
  (let ((nin n))
    (when (> nin 0)
      (with-slots (n p) d
        (cond ((eq nin 1) (random/binomial ($scalar n) ($scalar p)))
              (T ($binomial (tensor.int nin) ($scalar n) ($scalar p))))))))

(defmethod $ll ((d distribution/binomial) data)
  (with-slots (n p) d
    ($ll/binomial data p n)))

(defclass distribution/poisson (distribution)
  ((rate :initform 1.0)))

(defun distribution/poisson (&optional (rate 1D0))
  (let ((dist (make-instance 'distribution/poisson))
        (lin rate))
    (with-slots (rate) dist
      (setf rate lin))
    dist))

(defmethod $parameters ((d distribution/poisson))
  (with-slots (rate) d
    (when ($parameterp rate)
      (list rate))))

(defmethod $parameter-names ((d distribution/poisson))
  (list :rate))

(defmethod $ ((d distribution/poisson) name &rest others-and-default)
  (declare (ignore others-and-default))
  (when (eq name :rate)
    (with-slots (rate) d
      rate)))

(defmethod (setf $) (value (d distribution/poisson) name &rest others)
  (declare (ignore others))
  (when (eq name :rate)
    (with-slots (rate) d
      (setf rate value)))
  value)

(defmethod $sample ((d distribution/poisson) &optional (n 1))
  (when (> n 0)
    (with-slots (rate) d
      (cond ((eq n 1) (random/poisson ($scalar rate)))
            (T ($poisson (tensor.int n) ($scalar rate)))))))

(defmethod $ll ((d distribution/poisson) data)
  (with-slots (rate) d
    ($ll/poisson data rate)))
