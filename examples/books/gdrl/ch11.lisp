(defpackage :gdrl-ch11
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers
        #:th.env
        #:th.env.cartpole))

(in-package :gdrl-ch11)

(defun train-env (&optional (max-steps 300)) (cartpole-env :easy :reward max-steps))
(defun eval-env () (cartpole-env :eval))

(defun clamp-probs (probs)
  ($clamp probs
          single-float-epsilon
          (- 1 single-float-epsilon)))

(defun $logPs (probs) ($log (clamp-probs probs)))

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

(defun policy (m state &optional (trainp T))
  (let ((s (if (eq ($ndim state) 1)
               ($unsqueeze state 0)
               state)))
    ($execute m s :trainp trainp)))

(defun select-action (m state &optional (trainp T))
  (let* ((probs (policy m state trainp))
         (logPs ($logPs probs))
         (ps (if ($parameterp probs) ($data probs) probs))
         (entropy ($- ($dot ps logPs)))
         (action ($multinomial ps 1))
         (logP ($gather logPs 1 action)))
    (list ($scalar action) logP entropy)))

(defun action-selector (m)
  (lambda (state)
    (let ((probs (policy m state nil)))
      ($scalar ($argmax probs 1)))))

(defun reinforce (m &optional (max-episodes 4000))
  "REINFORCE updating per every episode."
  (let* ((gamma 0.99)
         (lr 0.01)
         (env (train-env))
         (avg-score nil)
         (success nil))
    (loop :while (not success)
          :repeat max-episodes
          :for e :from 1
          :for state = (env/reset! env)
          :for rewards = '()
          :for logPs = '()
          :for score = 0
          :for done = nil
          :do (let ((losses nil))
                (loop :while (not done)
                      :for (action logP entropy) = (select-action m state)
                      :for (next-state reward terminalp) = (cdr (env/step! env action))
                      :do (progn
                            (push logP logPs)
                            (push reward rewards)
                            (incf score reward)
                            (setf state next-state
                                  done terminalp)))
                (setf logPs (reverse logPs))
                (setf rewards (rewards (reverse rewards) gamma T))
                (loop :for logP :in logPs
                      :for vt :in rewards
                      :for i :from 0
                      :for gm = (expt gamma i)
                      :for l = ($- ($* gm logP vt))
                      ;; in practice, we don't have to collect losses.
                      ;; each loss has independent computational graph.
                      :do (push l losses))
                ($amgd! m lr)
                (if (null avg-score)
                    (setf avg-score score)
                    (setf avg-score (+ (* 0.9 avg-score) (* 0.1 score))))
                (when (zerop (rem e 100))
                  (let ((escore (cadr (evaluate (eval-env) (action-selector m)))))
                    (if (and (>= avg-score (* 0.9 300)) (>= escore 3000)) (setf success T))
                    (prn (format nil "~5D: ~8,2F / ~5,0F" e avg-score escore))))))
    avg-score))

;; train with REINFORCE
(defparameter *m* (model))
(reinforce *m* 4000)

;; evaluation
(evaluate (eval-env) (action-selector *m*))

;;
;; REINFORCE - batch updating
;;

(defun select-action (m state) ($scalar ($multinomial (policy m state nil) 1)))

(defun trace-episode (env m gamma &optional (nb 1))
  "collect episode trajectories with given policy model"
  (let ((states nil)
        (actions nil)
        (rewards nil)
        (gammas nil)
        (done nil)
        (score 0)
        (state nil))
    (loop :repeat nb
          :do (progn
                (setf state (env/reset! env))
                (loop :while (not done)
                      :for action = (select-action m state)
                      :for (_ next-state reward terminalp) = (env/step! env action)
                      :for i :from 0
                      :do (progn
                            (push ($list state) states)
                            (push action actions)
                            (push reward rewards)
                            (push (expt gamma i) gammas)
                            (incf score reward)
                            (setf state next-state
                                  done terminalp)))
                (setf done nil)))
    (let ((n ($count states)))
      (list (tensor (reverse states))
            (-> (tensor.long (reverse actions))
                ($reshape! n 1))
            (-> (rewards (reverse rewards) gamma T)
                (tensor)
                ($reshape! n 1))
            (-> (tensor (reverse gammas))
                ($reshape! n 1))
            (/ score nb)))))

(defun compute-loss (m states actions rewards gammas)
  (let ((logPs ($gather ($logPs (policy m states)) 1 actions)))
    ($mean ($* -1 gammas rewards logPs))))

(defun reinforce (m &optional (nbatch 5) (max-episodes 4000))
  "REINFORCE with batch updating"
  (let* ((gamma 0.99)
         (lr 0.04)
         (env (train-env))
         (avg-score nil)
         (success nil))
    (loop :while (not success)
          :repeat (round (/ max-episodes nbatch))
          :for e :from 1
          ;;:for state = (env/reset! env)
          :do (let* ((res (trace-episode env m gamma nbatch))
                     (states ($0 res))
                     (actions ($1 res))
                     (rewards ($2 res))
                     (gammas ($3 res))
                     (score ($4 res))
                     (loss nil))
                (setf loss (compute-loss m states actions rewards gammas))
                ($amgd! m lr)
                (if (null avg-score)
                    (setf avg-score score)
                    (setf avg-score (+ (* 0.9 avg-score) (* 0.1 score))))
                (when (zerop (rem e 100))
                  (let ((escore (cadr (evaluate (eval-env) (action-selector m)))))
                    (if (and (>= avg-score (* 0.9 300)) (>= escore 3000)) (setf success T))
                    (prn (format nil "~5D: ~8,2F / ~5,0F ~12,4F" e avg-score escore
                                 ($scalar ($data loss))))))))
    avg-score))

(defparameter *m* (model))
(reinforce *m* 10 4000) ;; more batch means large learning rate

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

(defun select-action (m state &optional (trainp T))
  (let* ((probs (policy m state trainp))
         (logPs ($logPs probs))
         (ps (if ($parameterp probs) ($data probs) probs))
         (entropy ($- ($dot ps logPs)))
         (action ($multinomial ps 1))
         (logP ($gather logPs 1 action)))
    (list ($scalar action) logP entropy)))

(defun val (m state &optional (trainp T))
  (let ((s (if (eq ($ndim state) 1)
               ($unsqueeze state 0)
               state)))
    ($execute m s :trainp trainp)))

(defun vpg (pm vm &optional (max-episodes 4000))
  (let* ((gamma 0.99)
         (beta 0.001)
         (plr 0.01)
         (vlr 0.01)
         (env (train-env))
         (avg-score nil)
         (success nil))
    (loop :while (not success)
          :repeat max-episodes
          :for e :from 1
          :for state = (env/reset! env)
          :for rewards = '()
          :for logPs = '()
          :for entropies = '()
          :for vals = '()
          :for score = 0
          :for done = nil
          :do (let ((plosses nil)
                    (vlosses nil))
                (loop :while (not done)
                      :for (action logP entropy) = (select-action pm state)
                      :for (_ next-state reward terminalp) = (env/step! env action)
                      :for v = (val vm state)
                      :do (progn
                            (push logP logPs)
                            (push reward rewards)
                            (push ($* beta entropy) entropies)
                            (push v vals)
                            (incf score reward)
                            (setf state next-state
                                  done terminalp)))
                (setf logPs (reverse logPs)
                      entropies (reverse entropies)
                      vals (reverse vals))
                (setf rewards (rewards (reverse rewards) gamma T))
                (loop :for logP :in logPs
                      :for vt :in rewards
                      :for et :in entropies
                      :for v :in vals
                      :for i :from 0
                      :for gm = (expt gamma i)
                      ;; in practice, we don't have to collect losses.
                      ;; each loss has independent computational graph.
                      :do (let ((adv ($- vt v)))
                            (push ($- ($+ ($* gm logP ($data adv)) et)) plosses)
                            (push ($square adv) vlosses)))
                ($amgd! pm plr)
                ($amgd! vm vlr)
                (if (null avg-score)
                    (setf avg-score score)
                    (setf avg-score (+ (* 0.9 avg-score) (* 0.1 score))))
                (when (zerop (rem e 100))
                  (let ((escore (cadr (evaluate (eval-env) (action-selector pm)))))
                    (if (and (>= avg-score (* 0.9 300)) (>= escore 3000)) (setf success T))
                    (prn (format nil "~5D: ~8,2F / ~5,0F" e avg-score escore))))))
    avg-score))

(defparameter *pm* (model))
(defparameter *vm* (vmodel))
(vpg *pm* *vm* 4000)

(evaluate (eval-env) (action-selector *pm*))

;;
;; ACTOR-CRITIC - ADVANTAGE ACTOR CRITIC (A2C) BATCH
;;

(defun ac (pm vm &optional (max-episodes 4000))
  (let* ((gamma 0.99)
         (beta 0.001)
         (plr 0.01)
         (vlr 0.01)
         (env (train-env))
         (avg-score nil)
         (success nil))
    (loop :while (not success)
          :repeat max-episodes
          :for e :from 1
          :for state = (env/reset! env)
          :for rewards = '()
          :for logPs = '()
          :for entropies = '()
          :for vals = '()
          :for score = 0
          :for done = nil
          :do (let ((plosses nil)
                    (vlosses nil))
                (loop :while (not done)
                      :for (action logP entropy) = (select-action pm state)
                      :for (_ next-state reward terminalp) = (env/step! env action)
                      :for v = (val vm state)
                      :do (progn
                            (push logP logPs)
                            (push reward rewards)
                            (push ($* beta entropy) entropies)
                            (push v vals)
                            (incf score reward)
                            (setf state next-state
                                  done terminalp)))
                (setf logPs (reverse logPs)
                      entropies (reverse entropies)
                      vals (reverse vals))
                (setf rewards (rewards (reverse rewards) gamma T))
                (loop :for logP :in logPs
                      :for vt :in rewards
                      :for et :in entropies
                      :for v :in vals
                      :for i :from 0
                      :for gm = (expt gamma i)
                      ;; in practice, we don't have to collect losses.
                      ;; each loss has independent computational graph.
                      :do (let ((adv ($- vt v)))
                            (push ($- ($+ ($* gm logP ($data adv)) et)) plosses)
                            (push ($square adv) vlosses)))
                ($amgd! pm plr)
                ($amgd! vm vlr)
                (if (null avg-score)
                    (setf avg-score score)
                    (setf avg-score (+ (* 0.9 avg-score) (* 0.1 score))))
                (when (zerop (rem e 100))
                  (let ((escore (cadr (evaluate (eval-env) (action-selector pm)))))
                    (if (and (>= avg-score (* 0.9 300)) (>= escore 3000)) (setf success T))
                    (prn (format nil "~5D: ~8,2F / ~5,0F" e avg-score escore))))))
    avg-score))

(defparameter *pm* (model))
(defparameter *vm* (vmodel))
(ac *pm* *vm* 4000)

(evaluate (eval-env) (action-selector *pm*))

;;
;; ACTOR-CRITIC - ONLINE VERSION
;;

(defun ac2 (pm vm &optional (max-episodes 4000))
  (let* ((gamma 0.99)
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
          :for logPs = '()
          :for entropies = '()
          :for vals = '()
          :for score = 0
          :for done = nil
          :do (let ((plosses nil)
                    (vlosses nil))
                (loop :while (not done)
                      :for (action logP entropy) = (select-action pm state)
                      :for (_ next-state reward terminalp) = (env/step! env action)
                      :for v = (val vm state)
                      :for vn = (val vm next-state)
                      :for adv = ($+ reward
                                     ($- ($* (- 1 (if terminalp 1 0))
                                             gamma
                                             vn)
                                         v))
                      :do (let ((vloss ($square adv))
                                (ploss ($- ($* ($data adv) logP))))
                            (incf score reward)
                            (push ($data vloss) vlosses)
                            (push ($data ploss) plosses)
                            ($amgd! pm plr)
                            ($amgd! vm vlr)
                            (setf state next-state
                                  done terminalp)))
                (if (null avg-score)
                    (setf avg-score score)
                    (setf avg-score (+ (* 0.9 avg-score) (* 0.1 score))))
                (when (zerop (rem e 100))
                  (let ((escore (cadr (evaluate (eval-env) (action-selector pm)))))
                    (if (and (>= avg-score (* 0.9 300)) (>= escore 3000)) (setf success T))
                    (prn (format nil "~5D: ~8,2F / ~5,0F" e avg-score escore))))))
    avg-score))

(defparameter *pm* (model))
(defparameter *vm* (vmodel))
(ac2 *pm* *vm* 4000)

(evaluate (eval-env) (action-selector *pm*))

;;
;; ACTOR-CRITIC - N-STEP
;;

(defun acn (pm vm &optional (max-episodes 4000))
  (let* ((gamma 0.99)
         (beta 0.001)
         (plr 0.01)
         (vlr 0.01)
         (max-steps 200)
         (env (train-env))
         (avg-score nil)
         (success nil))
    (loop :while (not success)
          :repeat max-episodes
          :for e :from 1
          :for state = (env/reset! env)
          :for rewards = '()
          :for logPs = '()
          :for entropies = '()
          :for vals = '()
          :for score = 0
          :for steps = 0
          :for done = nil
          :do (let ((plosses nil)
                    (vlosses nil))
                (loop :while (not done)
                      :for (action logP entropy) = (select-action pm state)
                      :for (_ next-state reward terminalp) = (env/step! env action)
                      :for v = (val vm state)
                      :do (progn
                            (push logP logPs)
                            (push reward rewards)
                            (push ($* beta entropy) entropies)
                            (push v vals)
                            (incf score reward)
                            (incf steps)
                            ;; every at most max-steps, do train
                            (when (or terminalp (zerop (rem steps max-steps)))
                              (setf logPs (reverse logPs)
                                    entropies (reverse entropies)
                                    vals (reverse vals))
                              (if (> ($count rewards) 1)
                                  (setf rewards (rewards (reverse rewards) gamma T))
                                  (setf rewards (reverse rewards)))
                              (loop :for logP :in logPs
                                    :for vt :in rewards
                                    :for et :in entropies
                                    :for v :in vals
                                    :for i :from 0
                                    :for gm = (expt gamma i)
                                    ;; in practice, we don't have to collect losses.
                                    ;; each loss has independent computational graph.
                                    :do (let ((adv ($- vt v)))
                                          (push ($- ($+ ($* gm logP ($data adv)) et)) plosses)
                                          (push ($square adv) vlosses)))
                              ($amgd! pm plr)
                              ($amgd! vm vlr)
                              (setf logPs nil
                                    entropies nil
                                    vals nil
                                    rewards nil
                                    plosses nil
                                    vlosses nil))
                            (setf state next-state
                                  done terminalp)))
                (if (null avg-score)
                    (setf avg-score score)
                    (setf avg-score (+ (* 0.9 avg-score) (* 0.1 score))))
                (when (zerop (rem e 100))
                  (let ((escore (cadr (evaluate (eval-env) (action-selector pm)))))
                    (if (and (>= avg-score (* 0.9 300)) (>= escore 3000)) (setf success T))
                    (prn (format nil "~5D: ~8,2F / ~5,0F" e avg-score escore))))))
    avg-score))

(defparameter *pm* (model))
(defparameter *vm* (vmodel))
(acn *pm* *vm* 4000)

(evaluate (eval-env) (action-selector *pm*))

;;
;; ACTOR-CRITIC - SHARED NETWORK MODEL
;;

(defun smodel (&optional (ns 4) (na 2) (nv 1))
  (let ((h 16))
    (let ((common-net (sequential-layer
                       (affine-layer ns h :weight-initializer :random-uniform
                                          :activation :relu)
                       (affine-layer h h :weight-initializer :random-uniform
                                         :activation :relu)))
          (policy-net (affine-layer h na :weight-initializer :random-uniform
                                         :activation :softmax))
          (value-net (affine-layer h nv :weight-initializer :random-uniform
                                        :activation :nil)))
      (sequential-layer common-net
                        (parallel-layer policy-net value-net)))))

(defun policy-and-value (m state &optional (trainp T))
  ($execute m ($unsqueeze state 0) :trainp trainp))

(defun select-action (m state &optional (trainp T))
  (let* ((out (policy-and-value m state))
         (probs ($0 out))
         (logPs ($logPs probs))
         (val ($1 out))
         (ps (if ($parameterp probs) ($data probs) probs))
         (entropy ($- ($dot ps logPs)))
         (action ($multinomial ps 1))
         (logP ($gather logPs 1 action)))
    (list ($scalar action) logP entropy val)))

(defun action-selector (m)
  (lambda (state)
    (let* ((policy-and-value (policy-and-value m state nil))
           (policy ($0 policy-and-value)))
      ($scalar ($argmax policy 1)))))

(defun acs (m &optional (max-episodes 4000))
  (let* ((gamma 0.99)
         (beta 0.001)
         (lr 0.01)
         (pw 1)
         (vw 0.6)
         (env (train-env))
         (avg-score nil)
         (success nil))
    (loop :while (not success)
          :repeat max-episodes
          :for e :from 1
          :for state = (env/reset! env)
          :for rewards = '()
          :for logPs = '()
          :for entropies = '()
          :for vals = '()
          :for score = 0
          :for done = nil
          :do (let ((ploss 0)
                    (vloss 0)
                    (loss 0))
                (loop :while (not done)
                      :for (action logP entropy v) = (select-action m state)
                      :for (_ next-state reward terminalp) = (env/step! env action)
                      :do (progn
                            (push logP logPs)
                            (push reward rewards)
                            (push ($* beta entropy) entropies)
                            (push v vals)
                            (incf score reward)
                            (setf state next-state
                                  done terminalp)))
                (setf logPs (reverse logPs)
                      entropies (reverse entropies)
                      vals (reverse vals))
                (setf rewards (rewards (reverse rewards) gamma T))
                (loop :for logP :in logPs
                      :for vt :in rewards
                      :for et :in entropies
                      :for v :in vals
                      :for i :from 0
                      :for gm = (expt gamma i)
                      ;; in practice, we don't have to collect losses.
                      ;; each loss has independent computational graph.
                      :do (let ((adv ($- vt v)))
                            (setf ploss ($- ploss ($+ ($* gm logP ($data adv)) et)))
                            (setf vloss ($+ vloss ($square adv)))))
                (setf loss ($+ ($* pw ploss) ($* vw vloss)))
                ($amgd! m lr)
                (if (null avg-score)
                    (setf avg-score score)
                    (setf avg-score (+ (* 0.9 avg-score) (* 0.1 score))))
                (when (zerop (rem e 100))
                  (let ((escore (cadr (evaluate (eval-env) (action-selector m)))))
                    (if (and (>= avg-score (* 0.9 300)) (>= escore 3000)) (setf success T))
                    (prn (format nil "~5D: ~8,2F / ~5,0F" e avg-score escore))))))
    avg-score))

(defparameter *sm* (smodel))
(acs *sm* 4000)

(evaluate (eval-env) (action-selector *sm*))

;;
;; ACTOR-CRITIC - SHARED N-STEP
;;

(defun acsn (m &optional (max-episodes 4000))
  (let* ((gamma 0.99)
         (beta 0.001)
         (lr 0.01)
         (pw 1)
         (vw 0.6)
         (max-steps 200)
         (env (train-env))
         (avg-score nil)
         (success nil))
    (loop :while (not success)
          :repeat max-episodes
          :for e :from 1
          :for state = (env/reset! env)
          :for rewards = '()
          :for logPs = '()
          :for entropies = '()
          :for vals = '()
          :for score = 0
          :for steps = 0
          :for done = nil
          :do (let ((ploss 0)
                    (vloss 0)
                    (loss 0))
                (loop :while (not done)
                      :for (action logP entropy v) = (select-action m state)
                      :for (_ next-state reward terminalp) = (env/step! env action)
                      :do (progn
                            (push logP logPs)
                            (push reward rewards)
                            (push ($* beta entropy) entropies)
                            (push v vals)
                            (incf score reward)
                            (incf steps)
                            (when (or terminalp (zerop (rem steps max-steps)))
                              (setf logPs (reverse logPs)
                                    entropies (reverse entropies)
                                    vals (reverse vals))
                              (if (> ($count rewards) 1)
                                  (setf rewards (rewards (reverse rewards) gamma T))
                                  (setf rewards (reverse rewards)))
                              (loop :for logP :in logPs
                                    :for vt :in rewards
                                    :for et :in entropies
                                    :for v :in vals
                                    :for i :from 0
                                    :for gm = (expt gamma i)
                                    ;; in practice, we don't have to collect losses.
                                    ;; each loss has independent computational graph.
                                    :do (let ((adv ($- vt v)))
                                          (setf ploss ($- ploss ($+ ($* gm logP ($data adv)) et)))
                                          (setf vloss ($+ vloss ($square adv)))))
                              (setf loss ($+ ($* pw ploss) ($* vw vloss)))
                              ($amgd! m lr)
                              (setf logPs nil
                                    entropies nil
                                    vals nil
                                    rewards nil
                                    ploss 0
                                    vloss 0
                                    loss 0))
                            (setf state next-state
                                  done terminalp)))
                (if (null avg-score)
                    (setf avg-score score)
                    (setf avg-score (+ (* 0.9 avg-score) (* 0.1 score))))
                (when (zerop (rem e 100))
                  (let ((escore (cadr (evaluate (eval-env) (action-selector m)))))
                    (if (and (>= avg-score (* 0.9 300)) (>= escore 3000)) (setf success T))
                    (prn (format nil "~5D: ~8,2F / ~5,0F" e avg-score escore))))))
    avg-score))

(defparameter *sm* (smodel))
(acsn *sm* 4000)

(evaluate (eval-env) (action-selector *sm*))
