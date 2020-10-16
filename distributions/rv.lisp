(in-package th.distributions)

(defgeneric $value (rv))
(defgeneric (setf $value) (value rv))
(defgeneric $observation (rv))
(defgeneric (setf $observation) (observation rv))
(defgeneric $logp (rv))

(defmethod $value ((rv T)) rv)
(defmethod $reset! ((rv T)))
(defmethod $observation ((rv T)) rv)
(defmethod $logp ((rv T)) 0D0) ;; when invalid, return nil as $ll

(defclass random-variable ()
  ((value :initform nil)
   (observedp :initform nil)))

(defmethod print-object ((rv random-variable) stream)
  (with-slots (observedp) rv
    (format stream "~A~A" ($value rv) (if (not observedp) "?" ""))))

(defclass rv/gaussian (random-variable)
  ((location :initform 0D0)
   (scale :initform 1D0)))

(defun rv/gaussian (&key (location 0D0) (scale 1D0) observation)
  (let ((rv (make-instance 'rv/gaussian))
        (l location)
        (s scale))
    (with-slots (location scale value observedp) rv
      (setf location l
            scale s)
      (if observation
          (setf value observation
                observedp T)
          (setf value ($sample/gaussian 1 ($value location) ($value scale))
                observedp nil)))
    rv))
(defun rv/normal (&key (location 0D0) (scale 1D0) observation)
  (rv/gaussian :location location :scale scale :observation observation))

(defmethod $value ((rv rv/gaussian))
  (with-slots (value) rv
    value))

(defmethod (setf $value) (v (rv rv/gaussian))
  (with-slots (value observedp) rv
    (setf value v
          observedp nil)
    v))

(defmethod $observation ((rv rv/gaussian))
  (with-slots (value observedp) rv
    (when observedp value)))

(defmethod (setf $observation) (observation (rv rv/gaussian))
  (with-slots (value observedp) rv
    (setf value observation
          observedp T)
    observation))

(defmethod $ll ((rv rv/gaussian) data)
  (with-slots (location scale) rv
    ($ll/gaussian data ($value location) ($value scale))))

(defmethod $logp ((rv rv/gaussian))
  (with-slots (location scale) rv
    (let ((ll ($ll rv ($value rv)))
          (lp1 ($logp location))
          (lp2 ($logp scale)))
      (when (and ll lp1 lp2)
        ($add ll ($add lp1 lp2))))))

(defclass rv/exponential (random-variable)
  ((rate :initform 1D0)))

(defun rv/exponential (&key (rate 1D0) observation)
  (let ((rv (make-instance 'rv/exponential))
        (l rate))
    (with-slots (rate value) rv
      (setf rate l)
      (when observation (setf value observation)))
    rv))

(defmethod $value ((rv rv/exponential))
  (with-slots (rate value) rv
    (unless value
      (setf value ($sample/exponential 1 ($value rate))))
    value))

(defmethod (setf $value) (observation (rv rv/exponential))
  (with-slots (value) rv
    (setf value observation)
    observation))

(defmethod $ll ((rv rv/exponential) data)
  (with-slots (rate) rv
    ($ll/exponential data ($value rate))))

(defmethod $logp ((rv rv/exponential))
  (with-slots (rate) rv
    ($add ($ll rv ($value rv)) ($logp rate))))

(defclass rv/poisson (random-variable)
  ((rate :initform 1D0)))

(defun rv/poisson (&key (rate 1D0) observation)
  (let ((rv (make-instance 'rv/poisson))
        (l rate))
    (with-slots (rate value) rv
      (setf rate l)
      (when observation (setf value observation)))
    rv))

(defmethod $value ((rv rv/poisson))
  (with-slots (rate value) rv
    (unless value
      (setf value ($sample/poisson 1 ($value rate))))
    value))

(defmethod (setf $value) (observation (rv rv/poisson))
  (with-slots (value) rv
    (setf value observation)
    observation))

(defmethod $ll ((rv rv/poisson) data)
  (with-slots (rate) rv
    ($ll/poisson data ($value rate))))

(defmethod $logp ((rv rv/poisson))
  (with-slots (rate) rv
    ($add ($ll rv ($value rv)) ($logp rate))))

(defclass rv/dice (random-variable)
  ((n :initform 6)))

(defun rv/dice (&key (n 6) observation)
  (let ((rv (make-instance 'rv/dice))
        (nin n))
    (with-slots (n value) rv
      (setf n nin)
      (when observation (setf value observation)))
    rv))

(defmethod $value ((rv rv/dice))
  (with-slots (n value) rv
    (unless value
      (setf value ($sample/dice 1 ($value n))))
    value))

(defmethod (setf $value) (observation (rv rv/dice))
  (with-slots (value) rv
    (setf value observation)
    observation))

(defmethod $ll ((rv rv/dice) data)
  (with-slots (n) rv
    ($ll/dice data ($value n))))

(defmethod $logp ((rv rv/dice))
  (with-slots (n) rv
    ($add ($ll rv ($value rv)) ($logp n))))
