(defpackage :random-variable
  (:use #:common-lisp
        #:mu
        #:th
        #:th.distributions
        #:th.mcmc))

(in-package :random-variable)

(defgeneric $value (rv))
(defgeneric (setf $value) (observation rv))
(defgeneric $logp (rv))

(defmethod $value ((rv T)) rv)
(defmethod $logp ((rv T)) 0D0)

(defclass random-variable ()
  ((value :initform nil)))

(defmethod print-object ((rv random-variable) stream)
  (format stream "~A" ($value rv)))

(defclass rv/gaussian (random-variable)
  ((location :initform 0D0)
   (scale :initform 1D0)))

(defun rv/gaussian (&key (location 0D0) (scale 1D0) observation)
  (let ((rv (make-instance 'rv/gaussian))
        (l location)
        (s scale))
    (with-slots (location scale value) rv
      (setf location l
            scale s)
      (when observation (setf value observation)))
    rv))
(defun rv/normal (&key (location 0D0) (scale 1D0) observation)
  (rv/gaussian :location location :scale scale :observation observation))

(defmethod $value ((rv rv/gaussian))
  (with-slots (location scale value) rv
    (unless value
      (setf value ($sample/gaussian 1 ($value location) ($value scale))))
    value))

(defmethod (setf $value) (observation (rv rv/gaussian))
  (with-slots (value) rv
    (setf value observation)
    observation))

(defmethod $ll ((rv rv/gaussian) data)
  (with-slots (location scale) rv
    ($ll/gaussian data ($value location) ($value scale))))

(defmethod $logp ((rv rv/gaussian))
  (with-slots (location scale) rv
    ($add ($ll rv ($value rv))
          ($add ($logp location) ($logp scale)))))

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

(let ((rv (rv/gaussian))
      (observed (tensor '(0 0 0))))
  (setf ($value rv) observed)
  (prn "VALUE:"($value rv))
  (prn "LOGP:" ($logp rv)))

(prn (rv/exponential))

(let* ((m (rv/exponential))
       (n (rv/normal :location m))
       (obs '(1 1 1)))
  (setf ($value m) 1)
  (prn "VALUE[M]:" ($value m))
  (setf ($value n) obs)
  (prn "VALUE[N]:" ($value n))
  (prn "LOGP:" ($logp n)))

(let ((m (rv/exponential :observation 1))
      (n (rv/normal :location 1))
      (obs '(1 1 1)))
  (setf ($value n) obs)
  (prn "VALUE[N]:" ($value n))
  (prn "LOGP:" ($logp n))
  (prn "VALUE[M]:" ($value m))
  (prn "LOGP-EXP:" ($logp m))
  (prn "SUM-LOGP:" ($add ($logp m) ($logp n))))

(let ((rv (make-instance 'rv/gaussian)))
  (list ($value rv) ($value rv))
  (setf ($value rv) (tensor '(0 0 0.1 -0.1)))
  ($value rv)
  (setf ($value rv) 0)
  ($logp rv))

($ll/normal 0 0 1)

;; XXX OLDER ONE

(defgeneric $val (rv))
(defgeneric $logp (rv))

(defclass random-variable ()
  ((g :initform nil)
   (v :initform nil)))

(defmethod print-object ((rv random-variable) stream)
  (format stream "~A" ($val rv)))

(defun rv (generator)
  (let ((n (make-instance 'random-variable)))
    (with-slots (g v) n
      (setf g generator)
      (setf v ($sample g 1)))
    n))

(defmethod $val ((rv random-variable))
  (with-slots (g v) rv
    (unless v
      (setf v ($sample g 1)))
    v))

(defmethod $reset! ((rv random-variable))
  (with-slots (g v) rv
    (setf v ($sample g 1))
    v))

(defmethod $logp ((rv random-variable))
  (let ((v ($val rv)))
    (with-slots (g) rv
      ($ll g v))))

(rv (distribution/normal))
($val (rv (distribution/normal)))
($reset! (rv (distribution/normal)))
($logp (rv (distribution/normal)))

(defparameter *population* ($sample (distribution/normal 2.5) 5000))
(defparameter *observation* (tensor (loop :repeat 1000
                                          :collect ($ *population* (random 5000)))))

($mean *population*)
($mean *observation*)

(defun potential (position)
  ($- ($ll (distribution/normal position) *observation*)))

(let ((samples (hmc 1000 0 #'potential :step-size 0.05)))
  ;; data mean vs computed parameter which is the mean of the normal distribution
  (list ($mean *population*) ($mean *observation*) ($mean samples)))

;; model
;;
;; parameters
;; distributions - parameters
;; observations - distributions - parameters
;; likelihood - observations - distributions - parameters

;; XXX
;; bare functional log-likelihood and other functions should be in
;; the th.distributions.
;; current class based implementations should depends on them.


(defclass mymodel ()
  ((mu :initform 0)
   (n :initform (distribution/normal 0))))

(defmethod $ll ((m mymodel) data)
  (with-slots (mu n) m
    (setf ($ n :mu) mu)
    ($ll n data)))

(defun $nll (m mu data)
  (let ((muin mu))
    (with-slots (mu) m
      (setf mu muin))
    ($- ($ll m data))))

(defparameter *model* (make-instance 'mymodel))

(defun potential (position) ($nll *model* position *observation*))

(let ((samples (hmc 1000 0 #'potential :step-size 0.05)))
  ;; data mean vs computed parameter which is the mean of the normal distribution
  (list ($mean *population*) ($mean *observation*) ($mean samples)))
