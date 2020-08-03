(defpackage :gdrl-ch11
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers
        #:th.env
        #:th.env.cartpole))

(in-package :gdrl-ch11)

;;
;; REINFORCE
;;

(defun model (&optional (ni 4) (no 2))
  (let ((h1 5)
        (h2 5))
    (sequential-layer
     (affine-layer ni h1 :weight-initializer :random-uniform)
     (affine-layer h1 h2 :weight-initializer :random-uniform)
     (affine-layer h2 no :weight-initializer :random-uniform
                         :activation :softmax))))

(defun policy (m state &key (trainp T)) ($execute m ($unsqueeze state 0) :trainp trainp))

(defun select-action (m state)
  (let* ((probs (policy m state))
         (entropy ($- ($dot probs ($log probs))))
         (action ($multinomial ($data probs) 1))
         (pa ($gather probs 1 action)))
    (list ($scalar action) pa entropy)))

(defun reinforce (m &optional (max-episodes 2000))
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
                      :for (action prob entropy) = (select-action m state)
                      :for (_ next-state reward terminalp) = (env/step! env action)
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
                ($amgd! m lr)
                (push score episode-rewards)
                (when (zerop (rem e 100))
                  (prn (format nil "~5D: ~8,2F" e score)))))
    (reverse episode-rewards)))

(defparameter *m* (model))
(reinforce *m*)

(defun evaluate-model (env m)
  (let ((done nil)
        (state (env/reset! env))
        (score 0))
    (loop :while (not done)
          :for probs = (policy m state :trainp nil)
          :for action = ($scalar ($argmax probs 1))
          :for (_ next-state reward terminalp) = (env/step! env action)
          :do (progn
                (incf score reward)
                (setf state next-state
                      done terminalp)))
    score))

(evaluate-model (cartpole-v0-env 1000) *m*)

;; interestingly, you can run this policy model with regulated cartpole as well
(evaluate (cartpole-env :eval)
          (lambda (state)
            ($scalar ($argmax (policy *m* state :trainp nil) 1))))


;;
;; VANILLA POLICY GRADIENT, VPG
;;

(defun vmodel (&optional (ni 4) (no 1))
  (let ((h1 5)
        (h2 5))
    (sequential-layer
     (affine-layer ni h1 :weight-initializer :random-uniform)
     (affine-layer h1 h2 :weight-initializer :random-uniform)
     (affine-layer h2 no :weight-initializer :random-uniform
                         :activation :tanh))))

(defun vpg (m vm &optional (max-episodes 2000))
  (let ((gamma 0.99)
        (beta 0.001)
        (lr 0.02)
        (env (cartpole-v0-env 200))
        (episode-rewards '()))
    (loop :repeat max-episodes
          :for e :from 1
          :for state = (env/reset! env)
          :for rewards = '()
          :for logits = '()
          :for entropies = '()
          :for vals = '()
          :for score = 0
          :for done = nil
          :do (let ((policy-loss 0)
                    (value-loss 0))
                (loop :while (not done)
                      :for (action prob entropy) = (select-action m state)
                      :for (_ next-state reward terminalp) = (env/step! env action)
                      :for v = ($execute vm ($unsqueeze state 0))
                      :do (let* ((logit ($log prob)))
                            (push logit logits)
                            (push reward rewards)
                            (push ($* beta entropy) entropies)
                            (push v vals)
                            (incf score reward)
                            (setf state next-state
                                  done terminalp)))
                (loop :for logit :in (reverse logits)
                      :for rwds :on (reverse rewards)
                      :for et :in (reverse entropies)
                      :for v :in (reverse vals)
                      :do (let* ((returns (reduce #'+ (loop :for r :in rwds
                                                            :for i :from 0
                                                            :collect (* r (expt gamma i)))))
                                 (adv ($- returns v)))
                            (setf policy-loss ($- policy-loss ($+ ($* logit ($data adv)) et)))
                            (setf value-loss ($+ value-loss ($square adv)))))
                ;;(setf policy-loss ($/ policy-loss ($count logits)))
                ;;(setf value-loss ($/ value-loss ($count logits)))
                ($amgd! m lr)
                ($amgd! vm lr)
                (push score episode-rewards)
                (when (zerop (rem e 100))
                  (prn (format nil "~5D: ~8,2F" e score)))))
    (reverse episode-rewards)))

(defparameter *m* (model))
(defparameter *vm* (vmodel))
(vpg *m* *vm* 2000)

(evaluate-model (cartpole-v0-env 1000) *m*)

;; application on the regulated cartpole
(evaluate (cartpole-env :eval)
          (lambda (state)
            ($scalar ($argmax (policy *m* state :trainp nil) 1))))
