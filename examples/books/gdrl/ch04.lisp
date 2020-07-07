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
          :for action = (funcall strategy e q n)
          :for tx = (env/step! env action)
          :for reward = (transition/reward tx)
          :do (progn
                (incf ($ n action))
                (incf ($ q action) (/ (- reward ($ q action)) ($ n action)))
                (setf ($ qe e) q)
                (setf ($ returns e) reward)
                (setf ($ actions e) action)))
    (list name returns qe actions)))

(defun expr/name (res) ($ res 0))
(defun expr/returns (res) ($ res 1))
(defun expr/qe (res) ($ res 2))
(defun expr/actions (res) ($ res 3))

(defun pure-exploitation (env &key (nepisodes 1000))
  (run-episodes env :name "PURE EXPLOITATION"
                    :strategy (lambda (e q n)
                                (declare (ignore e n))
                                ($argmax q))
                    :nepisodes nepisodes))

(defun pure-exploration (env &key (nepisodes 1000))
  (run-episodes env :name "PURE EXPLORATION"
                    :strategy (lambda (e q n)
                                (declare (ignore e n))
                                (random ($count q)))
                    :nepisodes nepisodes))

(defun epsilon-greedy (env &key (epsilon 0.01) (nepisodes 1000))
  (run-episodes env :name (format nil "E-GREEDY ~A" epsilon)
                    :strategy (lambda (e q n)
                                (declare (ignore e n))
                                (if (> (random 1D0) epsilon)
                                    ($argmax q)
                                    (random ($count q))))
                    :nepisodes nepisodes))

(defun linear-decreasing-epsilon-greedy (env &key (epsilon0 1D0)
                                               (min-epsilon 0.01)
                                               (decay-ratio 0.05)
                                               (nepisodes 1000))
  (let ((decay-episodes (* nepisodes decay-ratio)))
    (run-episodes env :name (format nil "LINEAR E-GREEDY ~A ~A ~A"
                                    epsilon0 min-epsilon decay-ratio)
                      :strategy (lambda (e q n)
                                  (declare (ignore n))
                                  (let ((epsilon (+ (* (- 1D0 (/ e decay-episodes))
                                                       (- epsilon0 min-epsilon))
                                                    min-epsilon)))
                                    (if (< epsilon min-epsilon)
                                        (setf epsilon min-epsilon))
                                    (if (> epsilon epsilon0)
                                        (setf epsilon epsilon0))
                                    (if (> (random 1D0) epsilon)
                                        ($argmax q)
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
                      :strategy (lambda (e q n)
                                  (declare (ignore n))
                                  (let ((epsilon ($ epsilons e)))
                                    (if (> (random 1D0) epsilon)
                                        ($argmax q)
                                        (random ($count q))))))))

(defun optimistic-initialization (env &key (optimistic-estimate 1D0)
                                        (initial-count 100)
                                        (nepisodes 1000))
  (run-episodes env :name (format nil "OPTIMISTIC INITIALIZATION ~A ~A"
                                  optimistic-estimate initial-count)
                    :initq (lambda (q) ($fill! q optimistic-estimate))
                    :initn (lambda (n) ($fill! n initial-count))
                    :strategy (lambda (e q n)
                                (declare (ignore e n))
                                ($argmax q))
                    :nepisodes nepisodes))

(defun softmax-strategy (env &key (temperature0 99999) (min-temperature 0.00001D0)
                               (decay-ratio 0.04)
                               (nepisodes 1000))
  (let ((decay-episodes (* nepisodes decay-ratio)))
    (run-episodes env :name (format nil "SOFTMAX ~A ~A ~A"
                                    temperature0 min-temperature decay-ratio)
                      :strategy (lambda (e q n)
                                  (declare (ignore n))
                                  (let* ((tmp (let ((temp (- 1 (/ e decay-episodes))))
                                                (setf temp (* temp (- temperature0 min-temperature)))
                                                (incf temp min-temperature)
                                                (if (< temp min-temperature)
                                                    (setf temp min-temperature))
                                                (if (> temp temperature0)
                                                    (setf temp temperature0))
                                                temp))
                                         (scaled-q ($/ q tmp))
                                         (norm-q ($- scaled-q ($max scaled-q)))
                                         (exp-q ($exp norm-q))
                                         (probs ($/ exp-q ($sum exp-q))))
                                    ($choice (env/action-space env) probs)))
                      :nepisodes nepisodes)))

(defun upper-confidence-bound-strategy (env &key (c 2) (nepisodes 1000))
  (run-episodes env :name (format nil "UCB ~A" c)
                    :strategy (lambda (e q n)
                                (let ((a e))
                                  (when (>= e ($count q))
                                    (let ((u ($sqrt ($* c ($/ (log e) (tensor n))))))
                                      (setf a ($argmax ($+ u q)))))
                                  a))
                    :nepisodes nepisodes))

(defun thompson-sampling-strategy (env &key (alpha 1) (beta 1) (nepisodes 1000))
  (run-episodes env :name (format nil "THOMPSON SAMPLING ~A ~A" alpha beta)
                    :strategy (lambda (e q n)
                                (declare (ignore e))
                                ($argmax (random-normals q ($/ alpha ($+ ($sqrt (tensor n)) beta)))))
                    :nepisodes nepisodes))

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

(defun report-last-qs (env strategies)
  (let* ((true-q (env/true-q env))
         (opt-v ($max true-q))
         (expreses (loop :for strategy :in strategies
                         :collect (funcall strategy env))))
    (prn "***********")
    (prn "* RESULTS *")
    (prn "***********")
    (prn "")
    (prn "* TRUE Q:" true-q)
    (prn "* OPTM V:" opt-v "* OPTM A:" ($argmax true-q))
    (loop :for er :in expreses
          :for sname = (expr/name er)
          :for qe = (expr/qe er)
          :for lq = ($ qe (1- ($size qe 0)))
          :do (prn sname "=>" lq))
    (prn "")))

;; testing strategy - basic
(report-last-qs (th.env.bandits:two-armed-bernoulli-bandit-env 0.8)
                (basic-experiments))

;; testing strategy - advanced
(report-last-qs (th.env.bandits:two-armed-bernoulli-bandit-env 0.8)
                (advanced-experiments))

(defun run-experiments (experiments env)
  (let* ((true-q (env/true-q env))
         (opt-v ($max true-q))
         (res #{}))
    (loop :for experiment :in experiments
          :for strategy-result = (progn
                                   (env/reset! env)
                                   (funcall experiment env))
          :for name = (expr/name strategy-result)
          :for returns = (expr/returns strategy-result)
          :for q-episodes = (expr/qe strategy-result)
          :for action-episodes = (expr/actions strategy-result)
          :for accum-returns = ($cumsum returns)
          :for mean-rewards = ($/ accum-returns ($+ 1 (arange 0 ($count returns))))
          :for q-selected = (tensor (loop :for i :from 0 :below ($count action-episodes)
                                          :for a = ($ action-episodes i)
                                          :collect ($ true-q a)))
          :for regret = ($- opt-v q-selected)
          :for accum-regret = ($cumsum regret)
          :do (setf ($ res name)
                    (let ((sres #{}))
                      (setf ($ sres :returns) returns
                            ($ sres :accum-returns) accum-returns
                            ($ sres :qe) q-episodes
                            ($ sres :ae) action-episodes
                            ($ sres :accum-regret) accum-regret
                            ($ sres :mean-rewards) mean-rewards)
                      sres)))
    res))

(defparameter *basic-results* (run-experiments
                               (basic-experiments)
                               (th.env.bandits:two-armed-bernoulli-bandit-env 0.8)))
(let* ((name "EXPONENTIAL E-GREEDY 1.0 0.0 0.1")
       (vs ($list ($ ($ *basic-results* name) :accum-regret))))
  (plot-lines (nthcdr 1 vs) :yrange (cons 0 30)))

(defparameter *advanced-results* (run-experiments
                                  (advanced-experiments)
                                  (th.env.bandits:two-armed-bernoulli-bandit-env 0.8)))
(let* ((name "SOFTMAX 100 0.01 0.005")
       (vs ($list ($ ($ *advanced-results* name) :accum-regret))))
  (plot-lines (nthcdr 1 vs) :yrange (cons 0 5)))

;; 10 armed bandit
(let* ((env (th.env.bandits:ten-armed-gaussian-bandit-env))
       (true-q (env/true-q env))
       (opt-v ($max true-q)))
  (list (env/p-dist env) (env/r-dist env) true-q opt-v))

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
