(defpackage :cartpole2
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers
        #:th.env))

(in-package :cartpole2)

(defconstant +gravity+ 9.8D0)
(defconstant +masscart+ 1D0)
(defconstant +masspole+ 0.1D0)
(defconstant +total-mass+ (+ +masscart+ +masspole+))
(defconstant +length+ 0.5D0)
(defconstant +polemass-length+ (* +masspole+ +length+))
(defconstant +force-mag+ 10D0)
(defconstant +tau+ 0.02D0)

(defconstant +x-success-range+ 2.4D0)
(defconstant +theta-success-range+ (/ (* 12 PI) 180D0))

(defconstant +x-threshold+ 2.4D0)
(defconstant +theta-threshold-radians+ (/ PI 2))
(defconstant +c-trans+ 0.01D0)

(defconstant +train-max-steps+ 100)
(defconstant +eval-max-steps+ 3000)

(defclass cartpole-regulator-env ()
  ((mode :initform nil :accessor env/mode)
   (step :initform 0 :accessor env/episode-step)
   (state :initform nil :accessor env/state)))

(defun cartpole-regulator-env (&optional (m :train))
  (let ((n (make-instance 'cartpole-regulator-env)))
    (setf (env/mode n) m)
    (env/reset! n)
    n))

(defmethod env/reset! ((env cartpole-regulator-env))
  (with-slots (mode state step) env
    (setf step 0)
    (setf state (if (eq mode :train)
                    (tensor (list (random/uniform -2.3D0 2.3D0)
                                  0
                                  (random/uniform -0.3 0.3)
                                  0))
                    (tensor (list (random/uniform -1D0 1D0)
                                  0
                                  (random/uniform -0.3 0.3)
                                  0))))
    state))

(defmethod env/step! ((env cartpole-regulator-env) action)
  (let* ((x ($0 (env/state env)))
         (xd ($1 (env/state env)))
         (th ($2 (env/state env)))
         (thd ($3 (env/state env)))
         (force (if (eq action 1) +force-mag+ (- +force-mag+)))
         (costh (cos th))
         (sinth (sin th))
         (tmp (/ (+ force (* +polemass-length+ thd thd sinth))
                 +total-mass+))
         (thacc (/ (- (* +gravity+ sinth) (* costh tmp))
                   (* +length+
                      (- 4/3 (/ (* +masspole+ costh costh) +total-mass+)))))
         (xacc (- tmp (/ (* +polemass-length+ thacc costh) +total-mass+)))
         (cost +c-trans+)
         (done nil)
         (blown nil))
    (incf (env/episode-step env))
    (incf x (* +tau+ xd))
    (incf xd (* +tau+ xacc))
    (incf th (* +tau+ thd))
    (incf thd (* +tau+ thacc))
    (cond ((or (< x (- +x-threshold+)) (> x +x-threshold+)
               (< th (- +theta-threshold-radians+)) (> th +theta-threshold-radians+))
           (setf cost 1D0
                 done T))
          ((and (> x (- +x-success-range+)) (< x +x-success-range+)
                (> th (- +theta-success-range+)) (< th +theta-success-range+))
           (setf cost 0D0
                 done nil))
          (T (setf cost +c-trans+
                   done nil)))
    (when (>= (env/episode-step env)
             (if (eq :train (env/mode env)) +train-max-steps+ +eval-max-steps+))
      (setf blown T))
    (let ((next-state (tensor (list x xd th thd))))
      (setf (env/state env) next-state)
      (list nil next-state cost done blown))))

(defun generate-goal-patterns (&optional (size 100))
  (list (tensor (loop :repeat size
                      :collect (list (random/uniform -0.05 0.05)
                                     (random/normal 0 1)
                                     (random/uniform (- +theta-success-range+)
                                                     +theta-success-range+)
                                     (random/normal 0 1)
                                     (random 2))))
        (zeros size 1)))

(defun collect-experiences (env &optional selector)
  (let ((rollout '())
        (episode-cost 0)
        (state (env/reset! env))
        (done nil)
        (blown nil))
    (loop :while (and (not done) (not blown))
          :for action = (if selector
                            (funcall state)
                            (random 2))
          :for tx = (env/step! env action)
          :do (let ((next-state ($1 tx))
                    (cost ($2 tx)))
                (setf done ($3 tx)
                      blown ($4 tx))
                (push (list state action cost next-state done) rollout)
                (incf episode-cost cost)
                (setf state next-state)))
    (list (reverse rollout) episode-cost)))
