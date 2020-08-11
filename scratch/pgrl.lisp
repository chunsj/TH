(defpackage :policy-gradient-related-scratchpad
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers
        #:th.env
        #:th.env.cartpole)
  (:import-from #:th.env.examples #:short-corridor-env))

(in-package :policy-gradient-related-scratchpad)

;; to compare manual and autodiff
(defparameter *w0* ($* 0.01 (rndn 4 2)))

;; utility methods

(defun softmax-grad (sm)
  (let ((s ($transpose sm)))
    ($- ($diagflat s) ($@ s ($transpose s)))))

(defun policy-grad (ps action state)
  "computing d/dw of log P(action)"
  (let* ((ds ($ (softmax-grad ps) action))
         (dl ($/ ds ($ ps 0 action))) ;; note that we're differentiating log(P(action))
         (dw ($@ ($transpose state) ($unsqueeze dl 0))))
    dw))

(defun mean (xs) (/ (reduce #'+ xs) (length xs)))
(defun variance (xs)
  (let* ((n (length xs))
         (m (/ (reduce #'+ xs) n)))
    (/ (reduce #'+ (mapcar (lambda (x)
                             (expt (abs (- x m)) 2))
                           xs))
       n)))
(defun sd (xs) (sqrt (variance xs)))

(defun z-scored (vs fl)
  (if fl
      (let ((m (mean vs))
            (s (sd vs)))
        (mapcar (lambda (v) (/ (- v m) (if (> s 0) s 1))) vs))
      vs))

(defun returns (rewards gamma &optional standardizep)
  (let ((running 0))
    (-> (loop :for r :in (reverse rewards)
              :collect (progn
                         (setf running ($+ r (* gamma running)))
                         running))
        (reverse)
        (z-scored standardizep))))

;; our simple policy model
(defun policy (state w) ($softmax ($@ ($unsqueeze state 0) w)))

;; action selection
(defun select-action (state w &optional greedy)
  (let* ((probs (policy state w))
         (action (if greedy
                     ($argmax (if ($parameterp probs) ($data probs) probs) 1)
                     ($multinomial (if ($parameterp probs) ($data probs) probs) 1))))
    (list ($scalar action) ($gather probs 1 action) probs)))

;; selector for evaluation
(defun selector (w)
  (lambda (state)
    ($scalar ($argmax (policy state (if ($parameterp w) ($data w) w)) 1))))

;; most simpliest implementation using CartPole-v0
;; this does not uses auto differentiation of TH.

(defun reinforce-simple (env w &optional (max-episodes 2000))
  (let ((gamma 0.99)
        (lr 0.001)
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
                      :for (action prob probs) = (select-action state w T)
                      :for (_ next-state reward terminalp successp) = (env/step! env action)
                      :do (let ((grad (policy-grad probs action state)))
                            (push grad grads)
                            (push reward rewards)
                            (incf score reward)
                            (setf state next-state
                                  done terminalp)))
                (loop :for grad :in (reverse grads)
                      :for gt :in (returns (reverse rewards) gamma T)
                      :for i :from 0
                      :for gm = (expt gamma i)
                      :do  ($set! w ($+ w ($* lr gm gt grad))))
                (if (null avg-score)
                    (setf avg-score score)
                    (setf avg-score (+ (* 0.9 avg-score) (* 0.1 score))))
                (when (zerop (rem e 100))
                  (prn (format nil "~5D: ~8,2F / ~8,2F" e score avg-score)))))
    avg-score))

(defparameter *w* ($clone *w0*))
(reinforce-simple (cartpole-fixed-env) *w* 1)

(evaluate (cartpole-fixed-env) (selector *w*))

;; using auto differentiation of TH.

(defun reinforce-bp (env w &optional (max-episodes 2000))
  (let ((gamma 0.99)
        (lr 0.01)
        (avg-score nil))
    (loop :repeat max-episodes
          :for e :from 1
          :for state = (env/reset! env)
          :for rewards = '()
          :for logPs = '()
          :for score = 0
          :for done = nil
          :do (let ((losses nil))
                (loop :while (not done)
                      :for (action prob) = (select-action state w T)
                      :for (_ next-state reward terminalp successp) = (env/step! env action)
                      :do (let* ((logP ($log prob)))
                            (push logP logPs)
                            (push reward rewards)
                            (incf score reward)
                            (setf state next-state
                                  done terminalp)))
                (loop :for logP :in (reverse logPs)
                      :for gt :in (returns (reverse rewards) gamma T)
                      :for i :from 0
                      :for gm = (expt gamma i)
                      :do (push ($- ($* gm logP gt)) losses)) ;; not sure on this
                (reduce #'$+ losses)
                ($amgd! w lr)
                (if (null avg-score)
                    (setf avg-score score)
                    (setf avg-score (+ (* 0.9 avg-score) (* 0.1 score))))
                (when (zerop (rem e 100))
                  (prn (format nil "~5D: ~8,2F / ~8,2F" e score avg-score)))))
    avg-score))

(defparameter *w* ($parameter ($clone *w0*)))
(reinforce-bp (cartpole-fixed-env) *w* 1)

(evaluate (cartpole-fixed-env) (selector *w*))

;;
;; SHORT CORRIDOR
;;

(defun reinforce-corridor (w &optional (max-episodes 2000))
  (let ((gamma 1)
        (lr 0.001)
        (env (short-corridor-env))
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
                      :repeat 100
                      :for probs = (policy (tensor (list state)) w)
                      :for action = ($scalar ($multinomial probs 1))
                      :for (reward next-state terminalp) = (env/step! env action)
                      :do (let ((grad (policy-grad probs action state)))
                            (push grad grads)
                            (push reward rewards)
                            (incf score reward)
                            (setf state next-state
                                  done terminalp)))
                (loop :for grad :in (reverse grads)
                      :for gt :in (returns (reverse rewards) gamma T)
                      :for i :from 0
                      :for gm = (expt gamma i)
                      :do  ($set! w ($+ w ($* lr gm gt grad))))
                (if (null avg-score)
                    (setf avg-score score)
                    (setf avg-score (+ (* 0.9 avg-score) (* 0.1 score))))
                (when (zerop (rem e 100))
                  (prn (format nil "~5D: ~8,2F / ~8,2F" e score avg-score)))))
    avg-score))

(defparameter *w* (tensor '((-1.47 1.47))))
(reinforce-corridor *w*)

;; XXX gradient checking
(let* ((w0 (tensor '((0.01 -0.01) (-0.01 0.01) (0.01 -0.01) (-0.01 0.01))))
       (w1 ($clone w0))
       (w2 ($parameter ($clone w0))))
  (let* ((s (tensor '(0.1 -0.1 0.2 -0.2)))
         (a 0)
         (p1 (policy s w1))
         (p2 (policy s w2))
         (lp2 ($log ($ p2 0 a))))
    (list p1 p2 lp2
          (policy-grad p1 0 s)
          ($gradient w2))))
