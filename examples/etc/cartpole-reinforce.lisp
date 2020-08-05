(defpackage :cartpole-reinforce
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers
        #:th.env
        #:th.env.cartpole))

(in-package :cartpole-reinforce)

;; most simpliest implementation using CartPole-v0
;; this does not uses auto differentiation of TH.

(defun policy (state w)
  (let* ((z ($@ ($unsqueeze state 0) w))
         (exp ($exp z)))
    ($/ exp ($sum exp))))

(defun softmax-grad (sm)
  (let ((s ($reshape sm ($count sm) 1)))
    ($- ($diag ($reshape s ($count s))) ($@ s ($transpose s)))))

(defun reinforce-simple (w &optional (max-episodes 2000))
  (let ((gamma 0.99)
        (lr 0.0001)
        (env (cartpole-env :easy :reward 300))
        (avg-score nil))
    (loop :repeat max-episodes
          :for e :from 1
          :for state = (env/reset! env)
          :for grads = '()
          :for rewards = '()
          :for score = 0
          :for done = nil
          :do (progn
                (loop :while (not done)
                      :for probs = (policy state w)
                      :for action = ($scalar ($multinomial probs 1))
                      :for (_ next-state reward terminalp successp) = (env/step! env action)
                      :do (let* ((dsoftmax ($ (softmax-grad probs) action))
                                 (dlog ($/ dsoftmax ($ probs 0 action)))
                                 (grad ($@ ($transpose state)
                                           ($unsqueeze dlog 0))))
                            (push grad grads)
                            (push reward rewards)
                            (incf score reward)
                            (setf state next-state
                                  done terminalp)))
                (loop :for grad :in (reverse grads)
                      :for rwds :on (reverse rewards)
                      :do (let ((returns (reduce #'+ (loop :for r :in rwds
                                                           :for i :from 0
                                                           :collect (* r (expt gamma i))))))
                            ($set! w ($+ w ($* lr grad returns)))))
                (if (null avg-score)
                    (setf avg-score score)
                    (setf avg-score (+ (* 0.9 avg-score) (* 0.1 score))))
                (when (zerop (rem e 100))
                  (prn (format nil "~5D: ~8,2F / ~8,2F" e score avg-score)))))
    avg-score))

(defparameter *w* (rnd 4 2))
(reinforce-simple *w*)

(evaluate (cartpole-env :eval) (lambda (state) ($scalar ($argmax (policy state *w*) 1))))

;; using auto differentiation of TH.

(defun policy (state w) ($softmax ($@ ($unsqueeze state 0) w)))

(defun select-action (state w)
  (let* ((probs (policy state w))
         (action ($multinomial ($data probs) 1)))
    (list ($scalar action) ($gather probs 1 action))))

(defun reinforce-bp (w &optional (max-episodes 2000))
  (let ((gamma 0.99)
        (lr 0.01)
        (env (cartpole-env :easy :reward 300))
        (avg-score nil))
    (loop :repeat max-episodes
          :for e :from 1
          :for state = (env/reset! env)
          :for rewards = '()
          :for logits = '()
          :for score = 0
          :for done = nil
          :do (let ((loss 0))
                (loop :while (not done)
                      :for (action prob) = (select-action state w)
                      :for (_ next-state reward terminalp successp) = (env/step! env action)
                      :do (let* ((logit ($log prob)))
                            (push logit logits)
                            (push reward rewards)
                            (incf score reward)
                            (setf state next-state
                                  done terminalp)))
                (loop :for logit :in (reverse logits)
                      :for rwds :on (reverse rewards)
                      :do (let ((returns (reduce #'+ (loop :for r :in rwds
                                                           :for i :from 0
                                                           :collect (* r (expt gamma i))))))
                            (setf loss ($- loss ($* logit returns)))))
                ($amgd! w lr)
                (if (null avg-score)
                    (setf avg-score score)
                    (setf avg-score (+ (* 0.9 avg-score) (* 0.1 score))))
                (when (zerop (rem e 100))
                  (prn (format nil "~5D: ~8,2F / ~8,2F" e score avg-score)))))
    avg-score))

(defparameter *w* ($parameter (rnd 4 2)))
(reinforce-bp *w*)

(evaluate (cartpole-env :eval) (lambda (state) ($scalar ($argmax (policy state ($data *w*)) 1))))
