(defpackage :gdrl-ch11
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers
        #:th.env
        #:th.env.cartpole))

(in-package :gdrl-ch11)

(defun model (&optional (ni 4) (no 2))
  (let ((h1 5)
        (h2 5))
    (sequential-layer
     (affine-layer ni h1 :weight-initializer :random-uniform)
     (affine-layer h1 h2 :weight-initializer :random-uniform)
     (affine-layer h2 no :weight-initializer :random-uniform
                         :activation :softmax))))

(defun policy (m state) ($execute m ($unsqueeze state 0)))

(defun select-action (m state)
  (let* ((probs (policy m state))
         (action ($multinomial ($data probs) 1)))
    (list ($scalar action) ($gather probs 1 action))))

(defun reinforce-bp (m &optional (max-episodes 2000))
  (let ((gamma 0.99)
        (lr 0.01)
        (env (cartpole-v0-env 500))
        (episode-rewards '()))
    (loop :repeat max-episodes
          :for e :from 1
          :for state = (env/reset! env)
          :for rewards = '()
          :for logits = '()
          :for score = 0
          :for done = nil
          :do (let ((loss 0))
                (loop :while (not done)
                      :for (action prob) = (select-action m state)
                      :for (_ next-state reward terminalp) = (env/step! env action)
                      :do (let* ((logit ($log prob)))
                            (push logit logits)
                            (push reward rewards)
                            (incf score reward)
                            p                            (setf state next-state
                                                               done terminalp)))
                (loop :for logit :in (reverse logits)
                      :for rwds :on (reverse rewards)
                      :do (let ((returns (reduce #'+ (loop :for r :in rwds
                                                           :for i :from 0
                                                           :collect (* r (expt gamma i))))))
                            (setf loss ($- loss ($* logit returns)))))
                ($amgd! m lr)
                (push score episode-rewards)
                (when (zerop (rem e 100))
                  (prn (format nil "~5D: ~8,2F" e score)))))
    (reverse episode-rewards)))

(defparameter *m* (model))
(reinforce-bp *m*)
