(defpackage :gdrl-ch08
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers
        #:th.env
        #:th.env.cartpole))

(in-package :gdrl-ch08)

(defun fcq (input-dim output-dim)
  (sequential-layer
   (affine-layer input-dim 512 :activation :selu)
   (affine-layer 512 128 :activation :selu)
   (affine-layer 128 output-dim :activation :nil)))

(defun epsilon-greedy (model state &key (epsilon 0.1D0))
  (let ((qvs ($evaluate model state)))
    (if (> (random 1D0) epsilon)
        ($argmax qvs 1)
        (tensor.long ($bernoulli (tensor ($size state 0) 1) 0.5D0)))))

;; xxx i don't know why $mse does not work
(defun loss (v r) ($mse v r))

(let* ((w ($parameter (zeros 4 2)))
       (a '((0) (1) (0) (1)))
       (r (ones 4 1)))
  (loop :repeat 3
        :for iter :from 0
        :for l = (loss ($gather w 1 a) r)
        :do (progn
              ($rmgd! w 0.1)
              (prn iter ":" ($data l)))))

(let* ((m (fcq 4 2))
       (x ($uniform (tensor 1 4) -0.05 0.05)))
  (epsilon-greedy m x))

(let* ((net (fcq 4 2))
       (x ($uniform (tensor 10 4) -0.05 0.05))
       (q ($evaluate net x))
       (dones (zeros 10 1))
       (rewards (ones 10 1))
       (gamma 0.99)
       (maxqs ($* (car ($max q 1)) ($- 1 dones)))
       (target-qs ($+ rewards ($* gamma maxqs)))
       (actions '((0) (0) (0) (1) (1) (1) (0) (0) (0) (1)))
       (qsa ($gather ($execute net x) 1 actions)))
  (list target-qs qsa))

(let ((env (cartpole-env))
      (n 10))
  (env/reset! env)
  (let ((states (->> (loop :for i :from 0 :below n
                           :for action = (random 2)
                           :collect (env/step! env action))
                     (mapcar #'cadr))))
    (-> (apply #'$concat states)
        ($reshape! n 4))))

(defparameter *model* (fcq 4 2))

(let ((env (cartpole-env))
      (model *model*)
      (nbatch 1024)
      (ntrain 40)
      (epsilon 0.5D0)
      (gamma 1D0)
      (lr 0.0001))
  (loop :repeat 1
        :for episode :from 0
        :for state = (env/reset! env)
        :do (let ((experiences '()))
              (loop :for i :from 0 :below nbatch
                    :for action = (-> (epsilon-greedy model ($reshape state 1 4) :epsilon epsilon)
                                      ($ 0 0))
                    :do (let* ((tx (env/step! env action))
                               (next-state (transition/next-state tx))
                               (reward (transition/reward tx))
                               (terminalp (transition/terminalp tx)))
                          (push (list state action reward next-state terminalp)
                                experiences)
                          (setf state next-state)
                          (when terminalp
                            (setf state (env/reset! env)))))
              (let ((ne ($count experiences)))
                (let* ((states (-> (apply #'$concat (mapcar #'$0 experiences))
                                   ($reshape ne 4)))
                       (actions (-> (mapcar #'$1 experiences)
                                    (tensor.long)
                                    ($reshape ne 1)))
                       (rewards (-> (mapcar #'$2 experiences)
                                    (tensor)
                                    ($reshape ne 1)))
                       (next-states (-> (apply #'$concat (mapcar #'$3 experiences))
                                        ($reshape ne 4)))
                       (donesf (-> (mapcar (lambda (p) (if ($4 p) 0 1)) experiences)
                                   (tensor)
                                   ($reshape ne 1))))
                  (with-max-heap ()
                    (loop :repeat ntrain
                          :for k :from 1
                          :do (let* ((maxaqsp (-> ($evaluate model next-states)
                                                  ($max 1)
                                                  (car)))
                                     (target-qs ($+ rewards ($* gamma maxaqsp donesf)))
                                     (qsa ($gather ($execute model states) 1 actions))
                                     (loss (loss qsa target-qs)))
                                (when (zerop (rem k ntrain)) (prn episode loss))
                                ($gd! model lr)))))))))
