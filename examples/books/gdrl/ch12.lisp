(defpackage :gdrl-ch12
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers
        #:th.env
        #:th.env.cartpole
        #:th.env.pendulum))

(in-package :gdrl-ch12)

(defparameter *max-action* 2)

(defun actor-model (&optional (ni 3) (no 1))
  (let ((h 16)
        (max-action *max-action*))
    (sequential-layer
     (affine-layer ni h
                   :weight-initializer :random-uniform
                   :activation :relu)
     (affine-layer h h
                   :weight-initializer :random-uniform
                   :activation :relu)
     (affine-layer h no
                   :weight-initializer :random-uniform
                   :activation :tanh)
     (functional-layer
      (lambda (x &key (trainp t))
       (declare (ignore trainp))
       ($* x max-action))))))

(defun criticl-model (&optional (ns 3) (na 1) (no 1))
  (let ((h 16))
    (sequential-layer
     (affine-layer (+ ns na) h
                   :weight-initializer :random-uniform
                   :activation :relu)
     (affine-layer h h
                   :weight-initializer :random-uniform
                   :activation :relu)
     (affine-layer h no
                   :weight-initializer :random-uniform
                   :activation :nil))))

(defun actor (m state &optional (randomizedp T) (trainp T))
  (let ((s (if (eq ($ndim state) 1)
               ($unsqueeze state 0)
               state))
        (e (random/normal 0 1)))
    (if randomizedp
        ($clamp ($+ ($execute m s :trainp trainp) e) (- *max-action*) *max-action*)
        ($execute m s :trainp trainp))))

(defun critic (m state action &optional (trainp T))
  (let* ((s (if (eq ($ndim state) 1)
                ($unsqueeze state 0)
                state))
         (at (if ($tensorp action) action (tensor (list action))))
         (a (if (eq ($ndim at) 1)
                ($unsqueeze at 0)
                at))
         (x ($cat s a 1)))
    ($execute m x :trainp trainp)))

(defun critics (m1 m2 state action &optional (trainp T))
  (list (critic m1 state action trainp)
        (critic m2 state action trainp)))

(defun sync-model (ms mt &optional (tau -1))
  (loop :for ps :in ($parameters ms)
        :for pt :in ($parameters mt)
        :do (if (< tau 0)
                ($set! ($data pt) ($clone ($data ps)))
                ($set! ($data pt) ($+ ($* tau ($data ps))
                                      ($* (- 1 tau) ($data pt)))))))

(defun collect-experience (env am &optional (n 100))
  (let ((s (env/reset! env)))
    (loop :repeat n
          :for a = ($scalar (actor am s T nil))
          :for tx = (env/step! env a)
          :for ns = (transition/next-state tx)
          :for r = (transition/reward tx)
          :collect (let ((s0 s))
                     (setf s ns)
                     (list s0 a r ns)))))

(defparameter *am* (actor-model))
(defparameter *am-target* (actor-model))
(defparameter *cm1* (criticl-model))
(defparameter *cm2* (criticl-model))
(defparameter *cm1-target* (criticl-model))
(defparameter *cm2-target* (criticl-model))

(sync-model *am* *am-target*)
(sync-model *cm1* *cm1-target*)
(sync-model *cm2* *cm2-target*)

;; TD3
(let ((env (pendulum-env))
      (epochs 2000)
      (gamma 0.99)
      (tau 0.005)
      (nsample 1000)
      (npolicy 2)
      (nprn 50)
      (trainp T)
      (lra 0.001)
      (lrc 0.001)
      (ql nil)
      (al nil))
  ($cg! *am*)
  ($cg! *cm1*)
  ($cg! *cm2*)
  (loop :repeat epochs
        :for ne :from 1
        :for exps = (collect-experience env *am* nsample)
        :for states = ($catn (mapcar (lambda (e) ($unsqueeze ($0 e) 0)) exps) 0)
        :for actions = ($catn (mapcar (lambda (e) (tensor (list (list ($1 e))))) exps) 0)
        :for rewards = ($catn (mapcar (lambda (e) (tensor (list (list ($2 e))))) exps) 0)
        :for next-states = ($catn (mapcar (lambda (e) ($unsqueeze ($3 e) 0)) exps) 0)
        :for next-actions = (actor *am-target* next-states T nil)
        :for q1s-target = (critic *cm1-target* next-states next-actions nil)
        :for q2s-target = (critic *cm2-target* next-states next-actions nil)
        :for qs-target = ($+ rewards ($* gamma ($min ($cat q1s-target q2s-target 1) 1)))
        :for q1s = (critic *cm1* states actions trainp)
        :for q2s = (critic *cm2* states actions trainp)
        :for qloss = ($+ ($mse q1s qs-target) ($mse q2s qs-target))
        :do (let ((qlv (if ($parameterp qloss) ($data qloss) qloss))
                  (alv nil))
              ($amgd! *cm1* lrc)
              ($amgd! *cm2* lrc)
              (when (zerop (rem ne npolicy))
                (let* ((as (actor *am* states nil trainp))
                       (qs (critic *cm1* states as trainp))
                       (aloss ($- ($mean qs))))
                  (setf alv (if ($parameterp aloss) ($data aloss) aloss))
                  ($amgd! *am* lra)
                  ($cg! *cm1*)
                  (sync-model *am* *am-target* tau)
                  (sync-model *cm1* *cm1-target* tau)
                  (sync-model *cm2* *cm2-target* tau)))
              (when qlv
                (if ql
                    (setf ql (+ (* 0.9 ql) (* 0.1 qlv)))
                    (setf ql qlv)))
              (when alv
                (if al
                    (setf al (+ (* 0.9 al) (* 0.1 alv)))
                    (setf al alv)))
              (when (zerop (rem ne nprn))
                (prn (format nil "[~5D] ~10,2F ~10,4F | ~10,2F" ne ql al
                             ($scalar ($mean rewards))))))))

(let* ((env (pendulum-env))
       (s (env/reset! env)))
  (loop :repeat 100
        :for a = ($scalar (actor *am* s nil nil))
        :for tx = (env/step! env a)
        :for ns = (transition/next-state tx)
        :for r = (transition/reward tx)
        :collect (progn
                   (setf s ns)
                   (list ns r))))
