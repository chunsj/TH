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

(defmethod $ll ((d distribution/beta) (data number))
  (with-slots (a b) d
    (if (and (> data 0) (< data 1))
        ($+ ($mul ($sub a 1) ($log data))
            ($mul ($sub b 1) ($log ($- 1 data)))
            ($neg ($lbetaf a b)))
        most-negative-single-float)))

(defmethod $ll ((d distribution/beta) (data list))
  ($ll d (tensor data)))

(defmethod $ll ((d distribution/beta) (data tensor))
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

(defmethod $ll ((d distribution/exponential) (data number))
  (if (> data 0)
      (with-slots (l) d
        ($sub ($log l) ($mul l data)))
      most-negative-single-float))

(defmethod $ll ((d distribution/exponential) (data list))
  ($ll d (tensor data)))

(defmethod $ll ((d distribution/exponential) (data tensor))
  (let ((nn ($sum ($lt data 0))))
    (if (zerop nn)
        (with-slots (l) d
          ($sum ($sub ($log l) ($mul l data))))
        most-negative-single-float)))

(defmethod $ll ((d distribution/exponential) (data node))
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

(defmethod $ll ((d distribution/uniform) (data number))
  (with-slots (a b) d
    (if (and (<= data ($scalar b)) (>= data ($scalar a)))
        ($- ($log ($- b a)))
        most-negative-single-float)))

(defmethod $ll ((d distribution/uniform) (data list))
  (with-slots (a b) d
    (let ((nf ($count (filter (lambda (v) (and (>= v ($scalar a)) (<= v ($scalar b)))) data)))
          (n ($count data)))
      (if (eq n nf)
          ($* n ($- ($log ($- b a))))
          most-negative-single-float))))

(defmethod $ll ((d distribution/uniform) (data tensor))
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

(defmethod $ll ((d distribution/gaussian) (data number))
  (with-slots (mu sigma) d
    ($mul -1/2
          ($add ($log (* 2 pi))
                ($add ($mul 2 ($log sigma))
                      ($div ($square ($sub data mu))
                            ($square sigma)))))))

(defmethod $ll ((d distribution/gaussian) (data list))
  (with-slots (mu sigma) d
    ($sum ($mul -1/2
                ($add ($log (* 2 pi))
                      ($add ($mul 2 ($log sigma))
                            ($div ($square ($sub (tensor data) mu))
                                  ($square sigma))))))))

(defmethod $ll ((d distribution/gaussian) (data tensor))
  (with-slots (mu sigma) d
    ($sum ($mul -1/2
                ($add ($log (* 2 pi))
                      ($add ($mul 2 ($log sigma))
                            ($div ($square ($sub data mu))
                                  ($square sigma))))))))

(defclass distribution/gamma (distribution)
  ((k :initform 1D0)
   (s :initform 1D0)))

(defun distribution/gamma (&optional (k 1D0) (s 1D0))
  (let ((d (make-instance 'distribution/gamma))
        (kin k)
        (sin s))
    (with-slots (k s) d
      (setf k kin
            s sin))
    d))

(defmethod $parameters ((d distribution/gamma))
  (let ((ps '()))
    (with-slots (k s) d
      (when ($parameterp s) (push s ps))
      (when ($parameterp k) (push k ps)))
    ps))

(defmethod $parameter-names ((d distribution/gamma))
  (list :k :s))

(defmethod $sample ((d distribution/gamma) &optional (n 1))
  (when (> n 0)
    (with-slots (k s) d
      (cond ((eq n 1) (random/gamma ($scalar k) ($scalar s)))
            (T ($gamma (tensor n) ($scalar k) ($scalar s)))))))

(defmethod $ll ((d distribution/gamma) (data number))
  (with-slots (k s) d
    ($add ($neg ($add ($log ($gammaf k)) ($mul k ($log s))))
          ($sub ($mul ($sub k 1) ($log data))
                ($div data s)))))

(defmethod $ll ((d distribution/gamma) (data list))
  ($ll d (tensor data)))

(defmethod $ll ((d distribution/gamma) (data tensor))
  (with-slots (k s) d
    ($sum ($add ($neg ($add ($log ($gammaf k)) ($mul k ($log s))))
                ($sub ($mul ($sub k 1) ($log data))
                      ($div data s))))))

(defclass distribution/t (distribution)
  ((df :initform 5)
   (l :initform 0D0)
   (s :initform 1D0)))

(defun distribution/t (&optional (df 5) (l 0D0) (s 1D0))
  (let ((d (make-instance 'distribution/t))
        (dfin df)
        (lin l)
        (sin s))
    (with-slots (df l s) d
      (setf df dfin
            l lin
            s sin))
    d))

(defmethod $parameters ((d distribution/t))
  (let ((ps '()))
    (with-slots (df l s) d
      (when ($parameterp s) (push s ps))
      (when ($parameterp l) (push l ps))
      (when ($parameterp df) (push df ps))
      ps)))

(defmethod $parameter-names ((d distribution/t))
  (list :df :l :s))

(defmethod $ ((d distribution/t) name &rest ig)
  (declare (ignore ig))
  (with-slots (df l s) d
    (cond ((eq name :df) df)
          ((eq name :l) l)
          ((eq name :s) s))))

(defmethod (setf $) (value (d distribution/t) name &rest ig)
  (declare (ignore ig))
  (with-slots (df l s) d
    (cond ((eq name :df) (setf df value))
          ((eq name :l) (setf l value))
          ((eq name :s) (setf s value)))
    value))

(defmethod $sample ((d distribution/t) &optional (n 1))
  (when (> n 0)
    (with-slots (df l s) d
      (let ((x ($sample (distribution/gaussian 0 1) n))
            (y ($sample (distribution/gamma (* 0.5 ($scalar df)) 2) n)))
        ($add ($scalar l) ($mul ($scalar s) ($div x ($sqrt ($div y ($scalar df))))))))))

(defmethod $ll ((d distribution/t) (data number))
  (with-slots (df l s) d
    ($sub ($sub ($sub ($sub ($sub ($log ($gammaf ($div ($add df 1D0) 2D0)))
                                  ($log ($gammaf ($div df 2D0))))
                            ($log s))
                      ($mul 0.5D0 ($log pi)))
                ($mul 0.5D0 ($log df)))
          ($mul ($mul 0.5 ($add df 1))
                ($log ($add 1 ($div ($square ($div ($sub data l) s)) df)))))))

(defmethod $ll ((d distribution/t) (data list)) ($ll d (tensor data)))

(defmethod $ll ((d distribution/t) (data tensor))
  (with-slots (df l s) d
    ($sum
     ($sub ($sub ($sub ($sub ($sub ($log ($gammaf ($div ($add df 1D0) 2D0)))
                                   ($log ($gammaf ($div df 2D0))))
                             ($log s))
                       ($mul 0.5D0 ($log pi)))
                 ($mul 0.5D0 ($log df)))
           ($mul ($mul 0.5 ($add df 1))
                 ($log ($add 1 ($div ($square ($div ($sub data l) s)) df))))))))

(defclass distribution/chisq (distribution)
  ((k :initform 1D0)))

(defun distribution/chisq (&optional (k 1D0))
  (let ((d (make-instance 'distribution/chisq))
        (kin k))
    (with-slots (k) d
      (setf k kin))
    d))

(defmethod $parameters ((d distribution/chisq))
  (let ((ps '()))
    (with-slots (k) d
      (when ($parameters k) (push k ps)))
    ps))

(defmethod $parameter-names ((d distribution/chisq))
  (list :k))

(defmethod $ ((d distribution/chisq) name &rest ig)
  (declare (ignore ig))
  (with-slots (k) d
    (cond ((eq name :k) k))))

(defmethod (setf $) (value (d distribution/chisq) name &rest ig)
  (declare (ignore ig))
  (with-slots (k) d
    (cond ((eq name :k) (setf k value))))
  value)

(defmethod $sample ((d distribution/chisq) &optional (n 1))
  (when (> n 0)
    (with-slots (k) d
      (let ((gk (/ ($scalar k) 2D0))
            (s 2D0))
        (cond ((eq n 1) (random/gamma gk s))
              (T ($gamma (tensor n) gk s)))))))

(defmethod $ll ((d distribution/chisq) (data number))
  (with-slots (k) d
    (let ((k2 ($div k 2))
          (x2 (/ data 2)))
      ($sub ($mul ($sub k2 1) ($log data))
            ($add ($add x2 ($mul k2 (log 2)))
                  ($lgammaf k2))))))

(defmethod $ll ((d distribution/chisq) (data list)) ($ll d (tensor data)))

(defmethod $ll ((d distribution/chisq) (data tensor))
  (with-slots (k) d
    (let ((k2 ($div k 2))
          (x2 ($div data 2)))
      ($sum ($sub ($mul ($sub k2 1) ($log data))
                  ($add ($add x2 ($mul k2 (log 2)))
                        ($lgammaf k2)))))))
