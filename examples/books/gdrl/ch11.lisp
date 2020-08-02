(defpackage :gdrl-ch11
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers
        #:th.env
        #:th.env.cartpole))

(in-package :gdrl-ch11)

;; using auto differentiation of TH.

(defun policy (state w) ($softmax ($@ ($unsqueeze state 0) w)))

(defun select-action (state w)
  (let* ((probs (policy state w))
         (action ($multinomial ($data probs) 1)))
    (list ($scalar action) ($gather probs 1 action))))

(defun reinforce-bp (w &optional (max-episodes 2000))
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
                      :for (action prob) = (select-action state w)
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
                ($amgd! w lr)
                (push score episode-rewards)
                (when (zerop (rem e 100))
                  (prn (format nil "~5D: ~8,2F" e score)))))
    (reverse episode-rewards)))

(defparameter *w* ($parameter (rnd 4 2)))
(reinforce-bp *w*)

;; XXX following codes are previous exploration to learn REINFORCE

(defun model (&optional (ni 4) (no 2))
  (let ((h1 5)
        (h2 5))
    (sequential-layer
     (affine-layer ni h1 :weight-initializer :random-uniform)
     (affine-layer h1 h2 :weight-initializer :random-uniform)
     (affine-layer h2 no :weight-initializer :random-uniform))))

(defun $logsumexp (x &optional (dim -1))
  (unless (< dim 0)
    (let* ((maxe ($max x dim))
           (ses ($sum ($exp ($- x ($expand maxe ($size x)))) dim)))
      ($+ ($log ses) maxe))))

(defun fullpass (model x)
  (let* ((logits ($execute model x))
         (logits ($- logits ($expand ($logsumexp logits 1) ($size logits))))
         (argmax ($argmax ($data logits) 1))
         (probs ($softmax logits))
         (sample ($multinomial ($data probs) 1))
         (log-prob-sample ($gather logits 1 sample))
         (entropy ($sum ($- ($* logits probs)) 1)))
    (list sample ($eq argmax sample) log-prob-sample entropy)))

(defun select-action (model x)
  (let* ((logits ($evaluate model x))
         (logits ($- logits ($expand ($logsumexp logits 1) ($size logits))))
         (probs ($softmax logits))
         (sample ($multinomial probs 1)))
    ($scalar sample)))

(defun select-greedy-action (model x)
  (let ((logits ($evaluate model x)))
    ($scalar ($argmax logits 1))))

(defun best-action-selector (model)
  (lambda (state) (select-action model ($unsqueeze state 0))))

(defun optimize-model (model env)
  (let* ((gamma 0.95)
         (lr 0.1)
         (logpas '())
         (costs '())
         (cost 0)
         (loss nil)
         (done nil)
         (state (env/reset! env)))
    (loop :while (not done)
          :for n :from 0
          :for pass = (fullpass model ($unsqueeze state 0))
          :for action = ($scalar ($0 pass))
          :for exploratoryp = (eq 1 ($scalar ($1 pass)))
          :for logpa = ($2 pass)
          :for tx = (env/step! env action)
          :for next-state = (transition/next-state tx)
          :for c = (transition/reward tx)
          :for terminalp = (transition/terminalp tx)
          :do (let ((dcost (* c (expt gamma n))))
                (push logpa logpas)
                (push dcost costs)
                (incf cost dcost)
                (setf state next-state
                      done terminalp)))
    (setf logpas (apply #'$concat (reverse logpas)))
    (setf costs ($unsqueeze (tensor (reverse costs)) 0))
    (setf loss ($mean ($* logpas costs)))
    ($rmgd! model lr)
    ($data loss)))

(defparameter *m* (model))
($cg! *m*)

(let ((train-env (cartpole-env)))
  (apply #'max (loop :repeat 10000
                     :collect (optimize-model *m* train-env))))

(let ((eval-env (cartpole-regulator-env :eval)))
  (evaluate eval-env (best-action-selector *m*)))

;; XXX another

(defun policy (&optional (ni 4) (no 2))
  (let ((nh 128))
    (sequential-layer
     (affine-layer ni nh
                   :weight-initializer :random-uniform
                   :activation :relu)
     (affine-layer nh no
                   :weight-initializer :random-uniform
                   :activation :softmax))))

(defun select-action (policy state &optional (trainp T))
  (let* ((probs ($execute policy ($unsqueeze state 0) :trainp trainp))
         (action ($multinomial (if trainp ($data probs) probs) 1))
         (lp ($log ($clamp ($gather probs 1 action)
                           single-float-epsilon
                           (- 1 single-float-epsilon)))))
    (list ($scalar action) lp)))

(defun train-episode (policy env)
  (let* ((gamma 0.99D0)
         (costs nil)
         (lprobs nil)
         (done nil)
         (steps 0)
         (state (env/reset! env)))
    (loop :while (not done)
          :for action-selection = (select-action policy state)
          :for action = ($0 action-selection)
          :for logprob = ($1 action-selection)
          :for tx = (env/step! env action)
          :for next-state = (transition/next-state tx)
          :for cost = (transition/reward tx)
          :for terminalp = (transition/terminalp tx)
          :do (let ((discounted-cost (* (expt gamma steps) cost)))
                (push discounted-cost costs)
                (push logprob lprobs)
                (incf steps)
                (setf state next-state
                      done terminalp)))
    (let ((acost 0)
          (loss 0))
      (loop :for cost :in costs
            :for lp :in lprobs
            :collect (progn
                       (incf acost cost)
                       (setf loss ($+ loss ($* acost lp)))))
      ($amgd! policy 1E-2)
      (list ($scalar ($data loss)) acost steps))))

(defparameter *policy* (policy))
(defparameter *env* (cartpole-env))

(loop :repeat 1000
      :for ep :from 1
      :do (let ((res (train-episode *policy* *env*)))
            (when (zerop (rem ep 20))
              (format T "EP ~5D | LOSS ~8,4F / ~4D~%" ep ($0 res) ($2 res)))))

;; xxx another yet from numpy

(defun policy (state w)
  (let* ((z ($@ ($unsqueeze state 0) w))
         (exp ($exp z)))
    ($/ exp ($sum exp))))

(defun softmax-grad (sm)
  (let ((s ($reshape sm ($count sm) 1)))
    ($- ($diag ($reshape s ($count s))) ($@ s ($transpose s)))))

(defun reinforce-simple (w &optional (max-episodes 2000))
  (let ((gamma 0.99)
        (lr 0.001)
        (env (cartpole-v0-env 500))
        (episode-rewards '()))
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
                      :for tx = (env/step! env action)
                      :for next-state = (transition/next-state tx)
                      :for reward = (transition/reward tx)
                      :for terminalp = (transition/terminalp tx)
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
                            (setf w ($+ w ($* lr grad returns)))))
                (push score episode-rewards)
                (when (zerop (rem e 100))
                  (prn (format nil "~5D: ~8,2F" e score)))))))

(defparameter *w* (rnd 4 2))

(reinforce-simple *w*)
