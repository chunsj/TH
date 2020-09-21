(in-package :th.distributions)

(defclass distribution/beta (distribution)
  ((a :initform 1.0)
   (b :initform 1.0)))

(defun distribution/beta (&optional (a 1D0) (b 1D0))
  (let ((dist (make-instance 'distribution/beta))
        (ain a)
        (bin b))
    (with-slots (a b) dist
      (setf a ain
            b bin))
    dist))

(defmethod $parameters ((d distribution/beta))
  (with-slots (a b) d
    (let ((ps '()))
      (when ($parameterp b) (push b ps))
      (when ($parameterp a) (push a ps))
      ps)))

(defmethod $parameter-names ((d distribution/beta))
  (list :a :b))

(defmethod $ ((d distribution/beta) name &rest others-and-default)
  (declare (ignore others-and-default))
  (with-slots (a b) d
    (cond ((eq name :a) a)
          ((eq name :b) b))))

(defmethod (setf $) (value (d distribution/beta) name &rest others)
  (declare (ignore others))
  (with-slots (a b) d
    (cond ((eq name :a) (setf a value))
          ((eq name :b) (setf b value))))
  value)

(defmethod $sample ((d distribution/beta) &optional (n 1))
  (when (> n 0)
    (with-slots (a b) d
      (cond ((eq n 1) (random/beta ($scalar a) ($scalar b)))
            (T ($beta (tensor n) ($scalar a) ($scalar b)))))))

(defmethod $score ((d distribution/beta) (data number))
  (with-slots (a b) d
    (if (and (> data 0) (< data 1))
        ($+ ($mul ($sub a 1) ($log data))
            ($mul ($sub b 1) ($log ($- 1 data)))
            ($neg ($lbetaf a b)))
        most-negative-single-float)))

(defmethod $score ((d distribution/beta) (data list))
  ($score d (tensor data)))

(defmethod $score ((d distribution/beta) (data tensor))
  (let ((nz ($sum ($le data 0)))
        (no ($sum ($ge data 1))))
    (if (and (zerop nz) (zerop no))
        (with-slots (a b) d
          ($+ ($mul ($sub a 1) ($log data))
              ($mul ($sub b 1) ($log ($- 1 data)))
              ($neg ($lbetaf a b))))
        most-negative-single-float)))

(defclass distribution/exponential (distribution)
  ((l :initform 1.0)))

(defun distribution/exponential (&optional (l 1D0))
  (let ((dist (make-instance 'distribution/exponential))
        (lin l))
    (with-slots (l) dist
      (setf l lin))
    dist))

(defmethod $parameters ((d distribution/exponential))
  (with-slots (l) d
    (if ($parameterp l)
        (list l)
        nil)))

(defmethod $parameter-names ((d distribution/exponential))
  (list :l))

(defmethod $ ((d distribution/exponential) name &rest others-and-default)
  (declare (ignore others-and-default))
  (with-slots (l) d
    (when (eq name :l)
      l)))

(defmethod (setf $) (value (d distribution/exponential) name &rest others)
  (declare (ignore others))
  (with-slots (l) d
    (when (eq name :l)
      (setf l value)))
  value)

(defmethod $sample ((d distribution/exponential) &optional (n 1))
  (when (> n 0)
    (with-slots (l) d
      (cond ((eq n 1) (random/exponential ($scalar l)))
            (T ($exponential (tensor n) ($scalar l)))))))

(defmethod $score ((d distribution/exponential) (data number))
  (if (> data 0)
      (with-slots (l) d
        ($sub ($log l) ($mul l data)))
      most-negative-single-float))

(defmethod $score ((d distribution/exponential) (data list))
  ($score d (tensor data)))

(defmethod $score ((d distribution/exponential) (data tensor))
  (let ((nn ($sum ($lt data 0))))
    (if (zerop nn)
        (with-slots (l) d
          ($sum ($sub ($log l) ($mul l data))))
        most-negative-single-float)))

(defmethod $score ((d distribution/exponential) (data node))
  (let ((nn ($sum ($lt (if ($parameterp data) ($data data) data) 0))))
    (if (zerop nn)
        (with-slots (l) d
          ($sum ($sub ($log l) ($mul l data))))
        most-negative-single-float)))

(defclass distribution/uniform (distribution)
  ((a :initform 0.0)
   (b :initform 1.0)))

(defun distribution/uniform (&optional (a 0D0) (b 1D0))
  (let ((dist (make-instance 'distribution/uniform))
        (ain a)
        (bin b))
    (with-slots (a b) dist
      (setf a ain
            b bin))
    dist))

(defmethod $parameters ((d distribution/uniform))
  (with-slots (a b) d
    (let ((ps '()))
      (when ($parameterp b) (push b ps))
      (when ($parameterp a) (push a ps))
      ps)))

(defmethod $parameter-names ((d distribution/uniform))
  (list :a :b))

(defmethod $ ((d distribution/uniform) name &rest others-and-default)
  (declare (ignore others-and-default))
  (with-slots (a b) d
    (cond ((eq name :a) a)
          ((eq name :b) b))))

(defmethod (setf $) (value (d distribution/uniform) name &rest others)
  (declare (ignore others))
  (with-slots (a b) d
    (cond ((eq name :a) (setf a value))
          ((eq name :b) (setf b value))))
  value)

(defmethod $sample ((d distribution/uniform) &optional (n 1))
  (when (> n 0)
    (with-slots (a b) d
      (cond ((eq n 1) (random/uniform ($scalar a) ($scalar b)))
            (T ($uniform (tensor n) ($scalar a) ($scalar b)))))))

(defmethod $score ((d distribution/uniform) (data number))
  (with-slots (a b) d
    (if (and (<= data ($scalar b)) (>= data ($scalar a)))
        ($- ($log ($- b a)))
        most-negative-single-float)))

(defmethod $score ((d distribution/uniform) (data list))
  (with-slots (a b) d
    (let ((nf ($count (filter (lambda (v) (and (>= v ($scalar a)) (<= v ($scalar b)))) data)))
          (n ($count data)))
      (if (eq n nf)
          ($* n ($- ($log ($- b a))))
          most-negative-single-float))))

(defmethod $score ((d distribution/uniform) (data tensor))
  (with-slots (a b) d
    (let ((n ($count data))
          (nx ($gt data ($scalar b)))
          (nn ($lt data ($scalar a))))
      (if (and (zerop nx) (zerop nn))
          ($* n ($- ($log ($- b a))))
          most-negative-single-float))))

(defclass distribution/gaussian (distribution)
  ((mu :initform 0)
   (sigma :initform 1)))

(defun distribution/gaussian (&optional (mean 0) (stddev 1))
  (let ((dist (make-instance 'distribution/gaussian)))
    (with-slots (mu sigma) dist
      (setf mu mean
            sigma stddev))
    dist))

(defun distribution/normal (&optional (mean 0) (stddev 1))
  (distribution/gaussian mean stddev))

(defmethod $parameters ((d distribution/gaussian))
  (with-slots (mu sigma) d
    (let ((ps '()))
      (when ($parameterp sigma) (push sigma ps))
      (when ($parameterp mu) (push mu ps))
      ps)))

(defmethod $parameter-names ((d distribution/gaussian))
  (list :mu :sigma))

(defmethod $ ((d distribution/gaussian) name &rest others-and-default)
  (declare (ignore others-and-default))
  (with-slots (mu sigma) d
    (cond ((eq name :mu) mu)
          ((eq name :sigma) sigma))))

(defmethod (setf $) (value (d distribution/gaussian) name &rest others)
  (declare (ignore others))
  (with-slots (mu sigma) d
    (cond ((eq name :mu) (setf mu value))
          ((eq name :sigma) (setf sigma value)))
    value))

(defmethod $sample ((d distribution/gaussian) &optional (n 1))
  (when (> n 0)
    (with-slots (mu sigma) d
      (cond ((eq n 1) (random/normal (pv mu) (pv sigma)))
            (T ($normal (tensor n) (pv mu) (pv sigma)))))))

(defmethod $score ((d distribution/gaussian) (data number))
  (with-slots (mu sigma) d
    ($mul -1/2
          ($add ($log (* 2 pi))
                ($add ($mul 2 ($log sigma))
                      ($div ($square ($sub data mu))
                            ($square sigma)))))))

(defmethod $score ((d distribution/gaussian) (data list))
  (with-slots (mu sigma) d
    ($sum ($mul -1/2
                ($add ($log (* 2 pi))
                      ($add ($mul 2 ($log sigma))
                            ($div ($square ($sub (tensor data) mu))
                                  ($square sigma))))))))

(defmethod $score ((d distribution/gaussian) (data tensor))
  (with-slots (mu sigma) d
    ($sum ($mul -1/2
                ($add ($log (* 2 pi))
                      ($add ($mul 2 ($log sigma))
                            ($div ($square ($sub data mu))
                                  ($square sigma))))))))
