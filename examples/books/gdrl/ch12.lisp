(defpackage :gdrl-ch12
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers
        #:th.env
        #:th.env.cartpole
        #:th.env.pendulum))

(in-package :gdrl-ch12)

(defun actor-model (&optional (ni 3) (no 1))
  (let ((h 16)
        (max-action 2D0))
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

(defun actor (m state &optional (trainp T))
  (let ((s (if (eq ($ndim state) 1)
               ($unsqueeze state 0)
               state)))
    ($execute m s :trainp trainp)))

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
