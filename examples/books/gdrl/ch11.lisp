(defpackage :gdrl-ch11
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers
        #:th.env
        #:th.env.cartpole))

(in-package :gdrl-ch11)

(defun returns (rewards gamma)
  (loop :for r :in rewards
        :for i :from 0
        :summing (* r (expt gamma i))))

(defun train-env (&optional (max-steps 300)) (cartpole-env :easy :reward max-steps))
(defun eval-env () (cartpole-env :eval))

;;
;; REINFORCE
;;

(defun model (&optional (ni 4) (no 2))
  (let ((h 8))
    (sequential-layer
     (affine-layer ni h :weight-initializer :random-uniform
                        :activation :relu)
     (affine-layer h no :weight-initializer :random-uniform
                        :activation :softmax))))

(defun policy (m state &optional (trainp T)) ($execute m ($unsqueeze state 0) :trainp trainp))

(defun select-action (m state)
  (let* ((probs (policy m state))
         (logits ($log ($clamp probs
                               single-float-epsilon
                               (- 1 single-float-epsilon))))
         (entropy ($- ($dot probs logits)))
         (action ($multinomial ($data probs) 1))
         (logit ($gather logits 1 action)))
    (list ($scalar action) logit entropy)))

(defun action-selector (m)
  (lambda (state)
    (let ((probs (policy m state nil)))
      ($scalar ($argmax probs 1)))))

(defun reinforce (m &optional (max-episodes 4000))
  (let* ((gamma 0.99)
         (lr 0.001)
         (env (train-env))
         (avg-score nil)
         (success nil))
    (loop :while (not success)
          :repeat max-episodes
          :for e :from 1
          :for state = (env/reset! env)
          :for rewards = '()
          :for logits = '()
          :for score = 0
          :for done = nil
          :do (let ((loss 0)
                    (ne 0))
                (loop :while (not done)
                      :for (action logit entropy) = (select-action m state)
                      :for (next-state reward terminalp) = (cdr (env/step! env action))
                      :do (progn
                            (push logit logits)
                            (push reward rewards)
                            (incf ne)
                            (incf score reward)
                            (setf state next-state
                                  done terminalp)))
                (when (> ne 0)
                  (loop :for logit :in (reverse logits)
                        :for rwds :on (reverse rewards)
                        :do (let ((returns (returns rwds gamma)))
                              ($incf loss ($- ($* logit returns)))))
                  ;;(setf loss ($/ loss ne))
                  ($amgd! m lr)
                  (if (null avg-score)
                      (setf avg-score score)
                      (setf avg-score (+ (* 0.9 avg-score) (* 0.1 score)))))
                ($cg! m)
                (when (zerop (rem e 100))
                  (let ((escore (cadr (evaluate (eval-env) (action-selector m)))))
                    (if (and (>= avg-score (* 0.9 300)) (>= escore 3000)) (setf success T))
                    (prn (format nil "~5D: ~8,4F / ~5,0F | ~8,2F" e avg-score escore
                                 ($scalar loss)))))))
    avg-score))

(defparameter *m* (model))
(reinforce *m* 4000)

(evaluate (eval-env) (action-selector *m*))

;;
;; VANILLA POLICY GRADIENT, VPG
;;

(defun pmodel (&optional (ni 4) (no 2))
  (let ((h 8))
    (sequential-layer
     (affine-layer ni h :weight-initializer :random-uniform
                        :activation :relu)
     (affine-layer h no :weight-initializer :random-uniform
                        :activation :softmax))))

(defun vmodel (&optional (ni 4) (no 1))
  (let ((h 16))
    (sequential-layer
     (affine-layer ni h :weight-initializer :random-uniform
                        :activation :relu)
     (affine-layer h no :weight-initializer :random-uniform
                        :activation :nil))))

(defun vpg (pm vm &optional (max-episodes 4000))
  (let* ((gamma 0.99)
         (beta 0.001)
         (plr 0.001)
         (vlr 0.001)
         (env (train-env))
         (avg-score nil)
         (success nil))
    (loop :while (not success)
          :repeat max-episodes
          :for e :from 1
          :for state = (env/reset! env)
          :for rewards = '()
          :for logits = '()
          :for entropies = '()
          :for vals = '()
          :for score = 0
          :for done = nil
          :do (let ((ploss 0)
                    (vloss 0)
                    (ne 0))
                (loop :while (not done)
                      :for (action logit entropy) = (select-action pm state)
                      :for (_ next-state reward terminalp) = (env/step! env action)
                      :for v = ($execute vm ($unsqueeze state 0))
                      :do (progn
                            (push logit logits)
                            (push reward rewards)
                            (push ($* beta entropy) entropies)
                            (push v vals)
                            (incf ne)
                            (incf score reward)
                            (setf state next-state
                                  done terminalp)))
                (loop :for logit :in (reverse logits)
                      :for rwds :on (reverse rewards)
                      :for et :in (reverse entropies)
                      :for v :in (reverse vals)
                      :do (let* ((returns (returns rwds gamma))
                                 (adv ($- returns v)))
                            ($incf ploss ($- ($+ ($* logit ($data adv)) et)))
                            ($incf vloss ($square adv))))
                ;;($/ policy-loss ($/ ploss ne))
                ;;($/ value-loss ($/ vloss ne))
                ($amgd! pm plr)
                ($amgd! vm vlr)
                (if (null avg-score)
                    (setf avg-score score)
                    (setf avg-score (+ (* 0.9 avg-score) (* 0.1 score))))
                (when (zerop (rem e 100))
                  (let ((escore (cadr (evaluate (eval-env) (action-selector pm)))))
                    (if (and (>= avg-score (* 0.9 300)) (>= escore 3000)) (setf success T))
                    (prn (format nil "~5D: ~8,4F / ~5,0F | ~8,2F ~10,2F" e avg-score escore
                                 ($scalar ploss) ($scalar vloss)))))))
    avg-score))

(defparameter *pm* (model))
(defparameter *vm* (vmodel))
(vpg *pm* *vm* 4000)

(evaluate (eval-env) (action-selector *pm*))

;; TODO/PLAN XXX
;;
;; before proceed into actor-critic, first tacke with the n-step bootstrapping problem.
;; after that actor-critic with monte-carlo or vge could be approcheable.
;;

;;
;; ACTOR-CRITIC - SHARED NETWORK MODEL
;;

(defun ac-model (&optional (ns 4) (na 2) (nv 1))
  (let ((h 64))
    (let ((common-net (sequential-layer
                       (affine-layer ns h :weight-initializer :random-uniform
                                          :activation :tanh)
                       (affine-layer h h :weight-initializer :random-uniform
                                         :activation :tanh)))
          (policy-net (affine-layer h na :weight-initializer :random-uniform
                                         :activation :softmax))
          (value-net (affine-layer h nv :weight-initializer :random-uniform
                                        :activation :nil)))
      (sequential-layer common-net
                        (parallel-layer policy-net value-net)))))

(defun policy-and-value (m state &optional (trainp T))
  ($execute m ($unsqueeze state 0) :trainp trainp))

(defun select-action (m state)
  (let* ((out (policy-and-value m state))
         (probs ($0 out))
         (val ($1 out))
         (entropy ($- ($dot probs ($log probs))))
         (action ($multinomial ($data probs) 1))
         (pa ($gather probs 1 action)))
    (list ($scalar action) pa entropy val)))

(defun action-selector (m)
  (lambda (state)
    (let* ((policy-and-value (policy-and-value m state nil))
           (policy ($0 policy-and-value)))
      ($scalar ($argmax policy 1)))))

(defun ac (acm &optional (max-episodes 2000))
  (let* ((gamma 0.99)
         (beta 0.001)
         (lr 0.001)
         (nex 100)
         (env (cartpole-v0-env nex))
         (avg-score nil))
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
                    (value-loss 0)
                    (ne 0)
                    (loss nil))
                (loop :while (not done)
                      :for (action prob entropy v) = (select-action acm state)
                      :for (_ next-state reward terminalp) = (env/step! env action)
                      :do (let* ((logit ($log ($clamp prob
                                                      single-float-epsilon
                                                      (- 1 single-float-epsilon)))))
                            (push logit logits)
                            (push reward rewards)
                            (push ($* beta entropy) entropies)
                            (push v vals)
                            (incf ne)
                            (incf score reward)
                            (setf state next-state
                                  done terminalp)))
                (loop :for logit :in (reverse logits)
                      :for rwds :on (reverse rewards)
                      :for et :in (reverse entropies)
                      :for v :in (reverse vals)
                      :do (let* ((returns (returns rwds gamma))
                                 (adv ($- returns v)))
                            (setf policy-loss ($- policy-loss ($+ ($* logit ($data adv)) et)))
                            (setf value-loss ($+ value-loss ($square adv)))))
                (setf loss ($+ policy-loss ($/ value-loss ne)))
                ($amgd! acm lr)
                (if (null avg-score)
                    (setf avg-score score)
                    (setf avg-score (+ (* 0.9 avg-score) (* 0.1 score))))
                (when (zerop (rem e 100))
                  (let ((escore (cadr (evaluate (cartpole-v0-env *eval-steps*)
                                                (action-selector acm)))))
                    (prn (format nil "~5D: ~8,4F ~5,0F | ~8,2F ~10,2F" e avg-score escore
                                 ($scalar policy-loss) ($scalar value-loss)))))))
    avg-score))

(defparameter *m* (ac-model))
(ac *m* 2000)

(evaluate (cartpole-v0-env 1000) (action-selector *m*))
(evaluate (cartpole-env :eval) (action-selector *m*))

;; N-STEP

(defun n-step-ac (acm &optional (max-steps 10) (max-episodes 4000))
  (let* ((gamma 0.99)
         (beta 0.001)
         (lr 0.01)
         (bstep 500)
         (env (cartpole-v0-env bstep))
         (avg-score nil))
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
                    (value-loss 0)
                    (loss nil))
                (loop :while (not done)
                      :for steps :from 1
                      :for (action prob entropy v) = (select-action-ac acm state)
                      :for (_ next-state reward terminalp) = (env/step! env action)
                      :do (let* ((logit ($log prob)))
                            (push logit logits)
                            (push reward rewards)
                            (push ($* beta entropy) entropies)
                            (push v vals)
                            (incf score reward)
                            (when (or terminalp (zerop (rem steps max-steps)))
                              (if (and terminalp (< steps bstep))
                                  (push 0 rewards)
                                  (push ($scalar v) rewards))
                              (loop :for logit :in (reverse logits)
                                    :for rwds :on (reverse rewards)
                                    :for et :in (reverse entropies)
                                    :for v :in (reverse vals)
                                    :do (let* ((returns (returns rwds gamma))
                                               (adv ($- returns v)))
                                          (setf policy-loss ($- policy-loss
                                                                ($+ ($* logit ($data adv)) et)))
                                          (setf value-loss ($+ value-loss ($square adv)))))
                              (setf loss ($+ policy-loss ($sqrt value-loss)))
                              ($amgd! acm lr)
                              (setf rewards '()
                                    logits '()
                                    entropies '()
                                    vals '()))
                            (setf state next-state
                                  done terminalp)))
                (if (null avg-score)
                    (setf avg-score score)
                    (setf avg-score (+ (* 0.9 avg-score) (* 0.1 score))))
                (when (zerop (rem e 100))
                  (prn (format nil "~5D: ~8,2F / ~8,2F" e score avg-score)))))
    avg-score))

(defparameter *m* (ac-model))
(n-step-ac *m* 600 2000)

(evaluate-model-ac (cartpole-v0-env 1000) *m*)
