(defpackage :gdrl-ch04
  (:use #:common-lisp
        #:mu
        #:mplot
        #:th
        #:th.env)
  (:import-from #:th.env.bandits))

(in-package :gdrl-ch04)

(defun run-episodes (env &key (name "STRATEGY NAME") strategy initq initn (nepisodes 1000))
  (let ((q (zeros (env/action-count env)))
        (n (tensor.int (zeros (env/action-count env))))
        (qe (tensor nepisodes (env/action-count env)))
        (returns (tensor nepisodes))
        (actions (tensor.int nepisodes)))
    (when initq (funcall initq q))
    (when initn (funcall initn n))
    (loop :for e :from 0 :below nepisodes
          :for action = (funcall strategy e q)
          :for tx = (env/step! env action)
          :for reward = (transition/reward tx)
          :do (progn
                (incf ($ n action))
                (incf ($ q action) (/ (- reward ($ q action)) ($ n action)))
                (setf ($ qe e) q)
                (setf ($ returns e) reward)
                (setf ($ actions e) action)))
    (list name returns qe actions)))

(defun pure-exploitation (env &key (nepisodes 1000))
  (run-episodes env :name "PURE EXPLOITATION"
                    :strategy (lambda (e q)
                                (declare (ignore e))
                                ($argmax q 0))
                    :nepisodes nepisodes))

(defun pure-exploration (env &key (nepisodes 1000))
  (run-episodes env :name "PURE EXPLORATION"
                    :strategy (lambda (e q)
                                (declare (ignore e))
                                (random ($count q)))
                    :nepisodes nepisodes))

(defun epsilon-greedy (env &key (epsilon 0.01) (nepisodes 1000))
  (run-episodes env :name (format nil "E-GREEDY ~A" epsilon)
                    :strategy (lambda (e q)
                                (declare (ignore e))
                                (if (> (random 1D0) epsilon)
                                    ($argmax q 0)
                                    (random ($count q))))
                    :nepisodes nepisodes))

(defun linear-decreasing-epsilon-greedy (env &key (epsilon0 1D0)
                                               (min-epsilon 0.01)
                                               (decay-ratio 0.05)
                                               (nepisodes 1000))
  (let ((decay-episodes (* nepisodes decay-ratio)))
    (run-episodes env :name (format nil "LINEAR E-GREEDY ~A ~A ~A"
                                    epsilon0 min-epsilon decay-ratio)
                      :strategy (lambda (e q)
                                  (let ((epsilon (+ (* (- 1D0 (/ e decay-episodes))
                                                       (- epsilon0 min-epsilon))
                                                    min-epsilon)))
                                    (if (< epsilon min-epsilon)
                                        (setf epsilon min-epsilon))
                                    (if (> epsilon epsilon0)
                                        (setf epsilon epsilon0))
                                    (if (> (random 1D0) epsilon)
                                        ($argmax q 0)
                                        (random ($count q)))))
                      :nepisodes nepisodes)))

(defun exponential-decreasing-epsilon-greedy (env &key (epsilon0 1D0)
                                                    (min-epsilon 0.01)
                                                    (decay-ratio 0.1)
                                                    (nepisodes 1000))
  (let ((epsilons (let ((es ($+ ($* ($/ 0.01 (logspace -2 0 (round (* nepisodes decay-ratio))))
                                    (- epsilon0 min-epsilon))
                                min-epsilon))
                        (eps (tensor nepisodes)))
                    ($fill! eps ($last es))
                    (setf ($subview eps 0 ($count es)) es)
                    eps)))
    (run-episodes env :name (format nil "EXPONENTIAL E-GREEDY ~A ~A ~A"
                                    epsilon0 min-epsilon decay-ratio)
                      :strategy (lambda (e q)
                                  (let ((epsilon ($ epsilons e)))
                                    (if (> (random 1D0) epsilon)
                                        ($argmax q 0)
                                        (random ($count q))))))))

(defun optimistic-initialization (env &key (optimistic-estimate 1D0)
                                        (initial-count 100)
                                        (nepisodes 1000))
  (let ((q (-> (tensor ($count (env-action-space env))) ($fill! optimistic-estimate)))
        (n (-> (tensor.int ($count (env-action-space env))) ($fill! initial-count)))
        (qe (tensor nepisodes ($count (env-action-space env))))
        (returns (tensor nepisodes))
        (actions (tensor.int nepisodes))
        (name (format nil "Optimistic ~A ~A" optimistic-estimate initial-count)))
    (loop :for e :from 0 :below nepisodes
          :for maxres = ($max q 0)
          :for action = ($ (cadr maxres) 0)
          :for tx = (env-step! env action)
          :for reward = ($2 tx)
          :do (progn
                (incf ($ n action))
                (incf ($ q action) (/ (- reward ($ q action)) ($ n action)))
                (setf ($ qe e) q)
                (setf ($ returns e) reward)
                (setf ($ actions e) action)))
    (list name returns qe actions)))

(defun softmax-strategy (env &key (temperature0 99999) (min-temperature 0.00001D0)
                               (decay-ratio 0.04)
                               (nepisodes 1000))
  (let* ((q (zeros ($count (env-action-space env))))
         (n (tensor.int (zeros ($count (env-action-space env)))))
         (qe (tensor nepisodes ($count (env-action-space env))))
         (returns (tensor nepisodes))
         (actions (tensor.int nepisodes))
         (name (format nil "Softmax ~A ~A ~A" temperature0 min-temperature decay-ratio)))
    (loop :for e :from 0 :below nepisodes
          :for decay-episodes = (* nepisodes decay-ratio)
          :for temperature = (let ((temp (- 1 (/ e decay-episodes))))
                               (setf temp (* temp (- temperature0 min-temperature)))
                               (incf temp min-temperature)
                               (if (< temp min-temperature) (setf temp min-temperature))
                               (if (> temp temperature0) (setf temp temperature0))
                               temp)
          :for scaled-q = ($/ q temperature)
          :for norm-q = ($- scaled-q ($max scaled-q))
          :for exp-q = ($exp norm-q)
          :for probs = ($/ exp-q ($sum exp-q))
          :for action = ($choice (env-action-space env) probs)
          :for tx = (env-step! env action)
          :for reward = ($2 tx)
          :do (progn
                (incf ($ n action))
                (incf ($ q action) (/ (- reward ($ q action)) ($ n action)))
                (setf ($ qe e) q)
                (setf ($ returns e) reward)
                (setf ($ actions e) action)))
    (list name returns qe actions)))

(defun upper-confidence-bound-strategy (env &key (c 2) (nepisodes 1000))
  (let* ((q (zeros ($count (env-action-space env))))
         (n (tensor.int (zeros ($count (env-action-space env)))))
         (qe (tensor nepisodes ($count (env-action-space env))))
         (returns (tensor nepisodes))
         (actions (tensor.int nepisodes))
         (name (format nil "UCB ~A" c)))
    (loop :for e :from 0 :below nepisodes
          :for action = (let ((a e))
                          (when (>= e ($count q))
                            (let* ((u ($sqrt ($* c ($/ (log e) (tensor n))))))
                              (setf a ($argmax ($+ u q) 0))))
                          a)
          :for tx = (env-step! env action)
          :for reward = ($2 tx)
          :do (progn
                (incf ($ n action))
                (incf ($ q action) (/ (- reward ($ q action)) ($ n action)))
                (setf ($ qe e) q)
                (setf ($ returns e) reward)
                (setf ($ actions e) action)))
    (list name returns qe actions)))

(defun sample-normal (means stds)
  (let ((samples (tensor ($count means))))
    (loop :for i :from 0 :below ($count means)
          :do (setf ($ samples i) ($normal th::*generator*
                                           ($ means i)
                                           ($ stds i))))
    samples))

(defun thompson-sampling-strategy (env &key (alpha 1) (beta 1) (nepisodes 1000))
  (let* ((q (zeros ($count (env-action-space env))))
         (n (tensor.int (zeros ($count (env-action-space env)))))
         (qe (tensor nepisodes ($count (env-action-space env))))
         (returns (tensor nepisodes))
         (actions (tensor.int nepisodes))
         (name (format nil "Thompson Sampling ~A ~A" alpha beta)))
    (loop :for e :from 0 :below nepisodes
          :for samples = (sample-normal q ($/ alpha ($+ ($sqrt (tensor n)) beta)))
          :for action = ($argmax samples)
          :for tx = (env-step! env action)
          :for reward = ($2 tx)
          :do (progn
                (incf ($ n action))
                (incf ($ q action) (/ (- reward ($ q action)) ($ n action)))
                (setf ($ qe e) q)
                (setf ($ returns e) reward)
                (setf ($ actions e) action)))
    (list name returns qe actions)))

(defun basic-experiments ()
  (list (lambda (env)
          (pure-exploitation env))
        (lambda (env)
          (pure-exploration env))
        (lambda (env)
          (epsilon-greedy env :epsilon 0.07))
        (lambda (env)
          (epsilon-greedy env :epsilon 0.1))
        (lambda (env)
          (linear-decreasing-epsilon-greedy env :epsilon0 1.0
                                                :min-epsilon 0.0
                                                :decay-ratio 0.1))
        (lambda (env)
          (linear-decreasing-epsilon-greedy env :epsilon0 0.3
                                                :min-epsilon 0.001
                                                :decay-ratio 0.1))
        (lambda (env)
          (exponential-decreasing-epsilon-greedy env :epsilon0 1.0
                                                     :min-epsilon 0.0
                                                     :decay-ratio 0.1))
        (lambda (env)
          (exponential-decreasing-epsilon-greedy env :epsilon0 0.3
                                                     :min-epsilon 0.0
                                                     :decay-ratio 0.3))
        (lambda (env)
          (optimistic-initialization env :optimistic-estimate 1.0 :initial-count 10))
        (lambda (env)
          (optimistic-initialization env :optimistic-estimate 1.0 :initial-count 50))))

(defun advanced-experiments ()
  (list (lambda (env) (pure-exploitation env))
        (lambda (env) (pure-exploration env))
        (lambda (env)
          (exponential-decreasing-epsilon-greedy env :epsilon0 0.3
                                                     :min-epsilon 0.0
                                                     :decay-ratio 0.3))
        (lambda (env)
          (optimistic-initialization env :optimistic-estimate 1.0 :initial-count 10))
        (lambda (env) (softmax-strategy env :decay-ratio 0.005))
        (lambda (env) (softmax-strategy env :temperature0 100
                                       :min-temperature 0.01
                                       :decay-ratio 0.005))
        (lambda (env) (upper-confidence-bound-strategy env :c 0.2))
        (lambda (env) (upper-confidence-bound-strategy env :c 0.5))
        (lambda (env) (thompson-sampling-strategy env))
        (lambda (env) (thompson-sampling-strategy env :alpha 0.5 :beta 0.5))))

(defun run-experiments (experiments env)
  (let* ((true-q (true-q env))
         (opt-v (opt-v true-q))
         (res #{}))
    (loop :for experiment :in experiments
          :for strategy-result = (progn
                                   (env-reset! env)
                                   (funcall experiment env))
          :for name = ($0 strategy-result)
          :for returns = ($1 strategy-result)
          :for q-episodes = ($2 strategy-result)
          :for action-episodes = ($3 strategy-result)
          :for cum-returns = ($cumsum returns)
          :for mean-rewards = ($/ cum-returns ($+ 1 (arange 0 ($count returns))))
          :for q-selected = (tensor (loop :for i :from 0 :below ($count action-episodes)
                                          :for a = ($ action-episodes i)
                                          :collect ($ true-q a)))
          :for regret = ($- opt-v q-selected)
          :for cum-regret = ($cumsum regret)
          :do (setf ($ res name)
                    (let ((sres #{}))
                      (setf ($ sres :returns) returns
                            ($ sres :cum-returns) cum-returns
                            ($ sres :qe) q-episodes
                            ($ sres :ae) action-episodes
                            ($ sres :cum-regret) cum-regret
                            ($ sres :mean-rewards) mean-rewards)
                      sres)))
    res))

(let ((b2-vs '()))
  (loop :repeat 5
        :do (let* ((env (th.env.bandits:two-armed-random-fixed-bandit-env))
                   (true-q (true-q env))
                   (opt-v (opt-v true-q)))
              (prn "")
              (prn "***")
              (prn (env-p-dist env))
              (prn "Q:" true-q)
              (prn "V*: " opt-v)
              (push opt-v b2-vs)))
  (prn "")
  (prn "***")
  (prn "Mean V*:" (/ (reduce #'+ b2-vs) ($count b2-vs))))

(let* ((env (th.env.bandits:two-armed-random-fixed-bandit-env))
       (true-q (true-q env))
       (expres (epsilon-greedy env)))
  (cons true-q expres))

(let* ((env (th.env.bandits:two-armed-random-fixed-bandit-env))
       (true-q (true-q env))
       (expres (optimistic-initialization env :optimistic-estimate 1D0
                                              :initial-count 50)))
  (cons true-q expres))

(let* ((env (th.env.bandits:two-armed-random-fixed-bandit-env))
       (true-q (true-q env))
       (expres (thompson-sampling-strategy env)))
  (cons true-q expres))

(env (th.env.bandits:two-armed-random-fixed-bandit-env))


(defparameter *basic-results*
  (run-experiments (basic-experiments) (th.env.bandits:two-armed-random-fixed-bandit-env)))
(let* ((name "Pure exploration")
       (vs ($list ($ ($ *basic-results* name) :mean-rewards))))
  (plot-lines (nthcdr 200 vs) :yrange (cons 0 1)))

(defparameter *advanced-results*
  (run-experiments (advanced-experiments) (th.env.bandits:two-armed-random-fixed-bandit-env)))
(let* ((name "Thompson Sampling 0.5 0.5")
       (vs ($list ($ ($ *advanced-results* name) :mean-rewards))))
  (plot-lines (nthcdr 200 vs) :yrange (cons 0 1)))

(let* ((env (th.env.bandits:ten-armed-gaussian-bandit-env))
       (true-q (true-q env))
       (opt-v (opt-v true-q)))
  (list (env-p-dist env) (env-r-dist env) true-q opt-v))

(let* ((env (th.env.bandits:ten-armed-gaussian-bandit-env))
       (true-q (true-q env))
       (expres (optimistic-initialization env :optimistic-estimate 1D0
                                              :initial-count 50)))
  (cons true-q expres))

(defparameter *basic-results*
  (run-experiments (basic-experiments) (th.env.bandits:ten-armed-gaussian-bandit-env)))
(let* ((name "Pure exploration")
       (vs ($list ($ ($ *basic-results* name) :cum-regret))))
  (plot-lines (nthcdr 200 vs) :yrange (cons 0 100)))

(defparameter *advanced-results*
  (run-experiments (advanced-experiments) (th.env.bandits:ten-armed-gaussian-bandit-env)))
(let* ((name "Thompson Sampling 0.5 0.5")
       (vs ($list ($ ($ *advanced-results* name) :cum-regret))))
  (plot-lines (nthcdr 200 vs) :yrange (cons 0 100)))
