(defpackage :gdrl-ch04
  (:use #:common-lisp
        #:mu
        #:th
        #:th.env)
  (:import-from #:th.env.bandits))

(in-package :gdrl-ch04)

(defun true-q (env) ($* (env-p-dist env) (env-r-dist env)))
(defun opt-v (true-q) ($max true-q))

(defun pure-exploitation (env &key (nepisodes 1000))
  (let ((q (zeros ($count (env-action-space env))))
        (n (tensor.int (zeros ($count (env-action-space env)))))
        (qe (tensor nepisodes ($count (env-action-space env))))
        (returns (tensor nepisodes))
        (actions (tensor.int nepisodes))
        (name "Pure exploitation"))
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

(defun pure-exploration (env &key (nepisodes 1000))
  (let ((q (zeros ($count (env-action-space env))))
        (n (tensor.int (zeros ($count (env-action-space env)))))
        (qe (tensor nepisodes ($count (env-action-space env))))
        (returns (tensor nepisodes))
        (actions (tensor.int nepisodes))
        (name "Pure exploration"))
    (loop :for e :from 0 :below nepisodes
          :for action = (random ($count q))
          :for tx = (env-step! env action)
          :for reward = ($2 tx)
          :do (progn
                (incf ($ n action))
                (incf ($ q action) (/ (- reward ($ q action)) ($ n action)))
                (setf ($ qe e) q)
                (setf ($ returns e) reward)
                (setf ($ actions e) action)))
    (list name returns qe actions)))

(defun epsilon-greedy (env &key (epsilon 0.01) (nepisodes 1000))
  (let ((q (zeros ($count (env-action-space env))))
        (n (tensor.int (zeros ($count (env-action-space env)))))
        (qe (tensor nepisodes ($count (env-action-space env))))
        (returns (tensor nepisodes))
        (actions (tensor.int nepisodes))
        (name (format nil "E-greedy ~A" epsilon)))
    (loop :for e :from 0 :below nepisodes
          :for maxres = ($max q 0)
          :for action = (if (> (random 1D0) epsilon)
                            ($ (cadr maxres) 0)
                            (random ($count q)))
          :for tx = (env-step! env action)
          :for reward = ($2 tx)
          :do (progn
                (incf ($ n action))
                (incf ($ q action) (/ (- reward ($ q action)) ($ n action)))
                (setf ($ qe e) q)
                (setf ($ returns e) reward)
                (setf ($ actions e) action)))
    (list name returns qe actions)))

(defun linear-decreasing-epsilon-greedy (env &key (epsilon0 1D0)
                                               (min-epsilon 0.01)
                                               (decay-ratio 0.05)
                                               (nepisodes 1000))
  (let ((q (zeros ($count (env-action-space env))))
        (n (tensor.int (zeros ($count (env-action-space env)))))
        (qe (tensor nepisodes ($count (env-action-space env))))
        (returns (tensor nepisodes))
        (actions (tensor.int nepisodes))
        (name (format nil "Linear e-greedy ~A ~A ~A" epsilon0 min-epsilon decay-ratio)))
    (loop :for e :from 0 :below nepisodes
          :for decay-episodes = (* nepisodes decay-ratio)
          :for epsilon = (let ((epsilon (- 1D0 (/ e decay-episodes))))
                           (setf epsilon (* epsilon (- epsilon0 min-epsilon)))
                           (incf epsilon min-epsilon)
                           (if (< epsilon min-epsilon)
                               (setf epsilon min-epsilon))
                           (if (> epsilon epsilon0)
                               (setf epsilon epsilon0))
                           epsilon)
          :for maxres = ($max q 0)
          :for action = (if (> (random 1D0) epsilon)
                            ($ (cadr maxres) 0)
                            (random ($count q)))
          :for tx = (env-step! env action)
          :for reward = ($2 tx)
          :do (progn
                (incf ($ n action))
                (incf ($ q action) (/ (- reward ($ q action)) ($ n action)))
                (setf ($ qe e) q)
                (setf ($ returns e) reward)
                (setf ($ actions e) action)))
    (list name returns qe actions)))

(defun exponential-decreasing-epsilon-greedy (env &key (epsilon0 1D0)
                                                    (min-epsilon 0.01)
                                                    (decay-ratio 0.1)
                                                    (nepisodes 1000))
  (let* ((q (zeros ($count (env-action-space env))))
         (n (tensor.int (zeros ($count (env-action-space env)))))
         (qe (tensor nepisodes ($count (env-action-space env))))
         (returns (tensor nepisodes))
         (actions (tensor.int nepisodes))
         (decay-episodes (round (* nepisodes decay-ratio)))
         (epsilons (let ((es ($/ 0.01 (logspace -2 0 decay-episodes))))
                     (setf es ($* es (- epsilon0 min-epsilon)))
                     (setf es ($+ es min-epsilon))
                     (let ((eps (tensor nepisodes)))
                       ($fill! eps ($last es))
                       (setf ($subview eps 0 ($count es)) es)
                       eps)))
         (name (format nil "Exponential e-greedy ~A ~A ~A" epsilon0 min-epsilon decay-ratio)))
    (loop :for e :from 0 :below nepisodes
          :for decay-episodes = (* nepisodes decay-ratio)
          :for epsilon = ($ epsilons e)
          :for maxres = ($max q 0)
          :for action = (if (> (random 1D0) epsilon)
                            ($ (cadr maxres) 0)
                            (random ($count q)))
          :for tx = (env-step! env action)
          :for reward = ($2 tx)
          :do (progn
                (incf ($ n action))
                (incf ($ q action) (/ (- reward ($ q action)) ($ n action)))
                (setf ($ qe e) q)
                (setf ($ returns e) reward)
                (setf ($ actions e) action)))
    (list name returns qe actions)))

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
                               (if (> temp temperature0) (setf temp temperature0)))
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
                          (when (> e ($count q))
                            (let* ((u ($sqrt ($* c ($/ (log e) N))))
                                   (maxres ($max ($+ u q) 0 )))
                              ($ (cadr maxres) 0)))
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

(defun thompson-sampling-strategy (env &key (alpha 1) (beta 0) (nepisodes 1000))
  (let* ((q (zeros ($count (env-action-space env))))
         (n (tensor.int (zeros ($count (env-action-space env)))))
         (qe (tensor nepisodes ($count (env-action-space env))))
         (returns (tensor nepisodes))
         (actions (tensor.int nepisodes))
         (name (format nil "Thompson Sampling ~A ~A" alpha beta)))
    (loop :for e :from 0 :below nepisodes
          :for samples = (sample-normal q ($/ alpha ($+ ($sqrt n) beta)))
          :for action = (let ((maxres ($max samples 0)))
                          ($ (cadr maxres) 0))
          :for tx = (env-step! env action)
          :for reward = ($2 tx)
          :do (progn
                (incf ($ n action))
                (incf ($ q action) (/ (- reward ($ q action)) ($ n action)))
                (setf ($ qe e) q)
                (setf ($ returns e) reward)
                (setf ($ actions e) action)))
    (list name returns qe actions)))

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
