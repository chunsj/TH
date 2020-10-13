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

(defmethod $ll ((d distribution/beta) data)
  (with-slots (a b) d
    ($ll/beta data a b)))

(defclass distribution/exponential (distribution)
  ((rate :initform 1.0)))

(defun distribution/exponential (&optional (rate 1D0))
  (let ((dist (make-instance 'distribution/exponential))
        (lin rate))
    (with-slots (rate) dist
      (setf rate lin))
    dist))

(defmethod $parameters ((d distribution/exponential))
  (with-slots (rate) d
    (if ($parameterp rate)
        (list rate)
        nil)))

(defmethod $parameter-names ((d distribution/exponential))
  (list :rate))

(defmethod $ ((d distribution/exponential) name &rest others-and-default)
  (declare (ignore others-and-default))
  (with-slots (rate) d
    (when (eq name :rate)
      rate)))

(defmethod (setf $) (value (d distribution/exponential) name &rest others)
  (declare (ignore others))
  (with-slots (rate) d
    (when (eq name :rate)
      (setf rate value)))
  value)

(defmethod $sample ((d distribution/exponential) &optional (n 1))
  (when (> n 0)
    (with-slots (rate) d
      (cond ((eq n 1) (random/exponential ($scalar rate)))
            (T ($exponential (tensor n) ($scalar rate)))))))

(defmethod $ll ((d distribution/exponential) data)
  (with-slots (rate) d
    ($ll/exponential data rate)))

(defclass distribution/uniform (distribution)
  ((l :initform 0.0)
   (u :initform 1.0)))

(defun distribution/uniform (&optional (l 0D0) (u 1D0))
  (let ((dist (make-instance 'distribution/uniform))
        (lin l)
        (uin u))
    (with-slots (l u) dist
      (setf l lin
            u uin))
    dist))

(defmethod $parameters ((d distribution/uniform))
  (with-slots (l u) d
    (let ((ps '()))
      (when ($parameterp u) (push u ps))
      (when ($parameterp l) (push l ps))
      ps)))

(defmethod $parameter-names ((d distribution/uniform))
  (list :l :u))

(defmethod $ ((d distribution/uniform) name &rest others-and-default)
  (declare (ignore others-and-default))
  (with-slots (l u) d
    (cond ((eq name :l) l)
          ((eq name :u) u))))

(defmethod (setf $) (value (d distribution/uniform) name &rest others)
  (declare (ignore others))
  (with-slots (l u) d
    (cond ((eq name :l) (setf l value))
          ((eq name :u) (setf u value))))
  value)

(defmethod $sample ((d distribution/uniform) &optional (n 1))
  (when (> n 0)
    (with-slots (l u) d
      (cond ((eq n 1) (random/uniform ($scalar l) ($scalar u)))
            (T ($uniform (tensor n) ($scalar l) ($scalar u)))))))

(defmethod $ll ((d distribution/uniform) data)
  (with-slots (l u) d
    ($ll/uniform data l u)))

(defclass distribution/gaussian (distribution)
  ((location :initform 0)
   (scale :initform 1)))

(defun distribution/gaussian (&optional (location 0) (scale 1))
  (let ((dist (make-instance 'distribution/gaussian))
        (m location)
        (s scale))
    (with-slots (location scale) dist
      (setf location m
            scale s))
    dist))

(defun distribution/normal (&optional (location 0) (scale 1))
  (distribution/gaussian location scale))

(defmethod $parameters ((d distribution/gaussian))
  (with-slots (location scale) d
    (let ((ps '()))
      (when ($parameterp scale) (push scale ps))
      (when ($parameterp location) (push location ps))
      ps)))

(defmethod $parameter-names ((d distribution/gaussian))
  (list :location :scale))

(defmethod $ ((d distribution/gaussian) name &rest others-and-default)
  (declare (ignore others-and-default))
  (with-slots (location scale) d
    (cond ((eq name :location) location)
          ((eq name :scale) scale))))

(defmethod (setf $) (value (d distribution/gaussian) name &rest others)
  (declare (ignore others))
  (with-slots (location scale) d
    (cond ((eq name :location) (setf location value))
          ((eq name :scale) (setf scale value)))
    value))

(defmethod $sample ((d distribution/gaussian) &optional (n 1))
  (when (> n 0)
    (with-slots (location scale) d
      (cond ((eq n 1) (random/normal (pv location) (pv scale)))
            (T ($normal (tensor n) (pv location) (pv scale)))))))

(defmethod $ll ((d distribution/gaussian) data)
  (with-slots (location scale) d
    ($ll/gaussian data location scale)))

(defclass distribution/gamma (distribution)
  ((shape :initform 1D0)
   (scale :initform 1D0)))

(defun distribution/gamma (&optional (shape 1D0) (scale 1D0))
  (let ((d (make-instance 'distribution/gamma))
        (kin shape)
        (sin scale))
    (with-slots (shape scale) d
      (setf shape kin
            scale sin))
    d))

(defmethod $parameters ((d distribution/gamma))
  (let ((ps '()))
    (with-slots (shape scale) d
      (when ($parameterp shape) (push shape ps))
      (when ($parameterp scale) (push scale ps)))
    ps))

(defmethod $parameter-names ((d distribution/gamma))
  (list :shape :scale))

(defmethod $sample ((d distribution/gamma) &optional (n 1))
  (when (> n 0)
    (with-slots (shape scale) d
      (cond ((eq n 1) (random/gamma ($scalar shape) ($scalar scale)))
            (T ($gamma (tensor n) ($scalar shape) ($scalar scale)))))))

(defmethod $ll ((d distribution/gamma) data)
  (with-slots (shape scale) d
    ($ll/gamma data shape scale)))

(defclass distribution/t (distribution)
  ((dof :initform 5)
   (location :initform 0D0)
   (scale :initform 1D0)))

(defun distribution/t (&optional (dof 5) (location 0D0) (scale 1D0))
  (let ((d (make-instance 'distribution/t))
        (dfin dof)
        (lin location)
        (sin scale))
    (with-slots (dof location scale) d
      (setf dof dfin
            location lin
            scale sin))
    d))

(defmethod $parameters ((d distribution/t))
  (let ((ps '()))
    (with-slots (dof location scale) d
      (when ($parameterp scale) (push scale ps))
      (when ($parameterp location) (push location ps))
      (when ($parameterp dof) (push dof ps))
      ps)))

(defmethod $parameter-names ((d distribution/t))
  (list :dof :location :scale))

(defmethod $ ((d distribution/t) name &rest ig)
  (declare (ignore ig))
  (with-slots (dof location scale) d
    (cond ((eq name :dof) dof)
          ((eq name :location) location)
          ((eq name :scale) scale))))

(defmethod (setf $) (value (d distribution/t) name &rest ig)
  (declare (ignore ig))
  (with-slots (dof location scale) d
    (cond ((eq name :dof) (setf dof value))
          ((eq name :location) (setf location value))
          ((eq name :scale) (setf scale value)))
    value))

(defmethod $sample ((d distribution/t) &optional (n 1))
  (when (> n 0)
    (with-slots (dof location scale) d
      (let ((x ($sample (distribution/gaussian 0 1) n))
            (y ($sample (distribution/gamma (* 0.5 ($scalar dof)) 2) n)))
        ($add ($scalar location)
              ($mul ($scalar scale)
                    ($div x ($sqrt ($div y ($scalar dof))))))))))

(defmethod $ll ((d distribution/t) data)
  (with-slots (dof location scale) d
    ($ll/t data location scale dof)))

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

(defmethod $ll ((d distribution/chisq) data)
  (with-slots (k) d
    ($ll/chisq data k)))
