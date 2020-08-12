(defpackage :policy-gradient-comparison
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers
        #:th.env
        #:th.env.cartpole)
  (:import-from #:th.env.examples #:short-corridor-env))

(in-package :policy-gradient-comparison)

;; for comparison, we need same starting point
(defparameter *w0* (tensor '((0.002 0.007)
                             (-0.002 0.005)
                             (-0.005 0.003)
                             (0.004 -0.008))))

;; utility functions for manual gradient computation
(defun softmax-grad (sm)
  (let ((s ($transpose sm)))
    ($- ($diagflat s) ($@ s ($transpose s)))))

(defun policy-grad (ps action state)
  "computing d/dw of log P(action)"
  (let* ((ds ($ (softmax-grad ps) action))
         (dl ($/ ds ($ ps 0 action))) ;; note that we're differentiating log(P(action))
         (dw ($@ ($transpose state) ($unsqueeze dl 0))))
    dw))

;; for normalization with z-score
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

;; our the simple policy model for testing
(defun policy (state w) ($softmax ($@ ($unsqueeze state 0) w)))
(defun policy (state w)
  (let* ((hs ($@ ($unsqueeze state 0) w))
         (ss ($exp ($- hs ($max hs)))))
    ($/ ss ($sum ss))))

;; action selection
(defun select-action (state w &optional greedy)
  (let* ((probs (policy state w))
         (action (if greedy
                     ($argmax (if ($parameterp probs) ($data probs) probs) 1)
                     ($multinomial (if ($parameterp probs) ($data probs) probs) 1))))
    (list ($scalar action) ($gather probs 1 action) probs)))

;; action selector for evaluation
(defun selector (w)
  (lambda (state)
    ($scalar ($argmax (policy state (if ($parameterp w) ($data w) w)) 1))))

;; to store gradient values for comparison
(defparameter *manual-grads* nil)
(defparameter *backprop-grads* nil)

;; REINFORCE implementations - one with manual backprop, the other auto backprop
(defun reinforce-simple (env w &optional (max-episodes 2000))
  (let ((gamma 0.99)
        (lr 0.01)
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
                (setf *manual-grads* nil)
                (loop :for grad :in (reverse grads)
                      :for gt :in (returns (reverse rewards) gamma T)
                      :for i :from 0
                      :for gm = (expt gamma i)
                      :for gv = ($* gm gt grad)
                      :do  (progn
                             ;; for comparison, store value without learning rate
                             (push gv *manual-grads*)
                             ($set! w ($+ w ($* lr gv)))))
                (if (null avg-score)
                    (setf avg-score score)
                    (setf avg-score (+ (* 0.9 avg-score) (* 0.1 score))))
                (when (zerop (rem e 100))
                  (prn (format nil "~5D: ~8,2F / ~8,2F" e score avg-score)))))
    avg-score))

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
                      :for (action prob probs) = (select-action state w T)
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
                      :do (push ($- ($* gm logP gt)) losses))
                (reduce #'$+ losses)
                (setf *backprop-grads* nil)
                (loop :for f :in (th::$fns w)
                      :do (push (funcall f) *backprop-grads*))
                (setf *backprop-grads* (reverse *backprop-grads*))
                ($gd! w lr)
                (if (null avg-score)
                    (setf avg-score score)
                    (setf avg-score (+ (* 0.9 avg-score) (* 0.1 score))))
                (when (zerop (rem e 100))
                  (prn (format nil "~5D: ~8,2F / ~8,2F" e score avg-score)))))
    avg-score))

;; to check gradient values, first, compute a single episode
(defparameter *wm* ($clone *w0*))
(defparameter *wb* ($parameter ($clone *w0*)))

(reinforce-simple (cartpole-fixed-env 300) *wm* 10)
(reinforce-bp (cartpole-fixed-env 300) *wb* 10)

;; check gradient values - the difference should be almost zero
(eq ($count *backprop-grads*) ($count *manual-grads*))
(loop :for bg :in *backprop-grads*
      :for mg :in *manual-grads*
      :for d = ($+ bg mg) ;; bg and mg has counter sign
      :summing ($scalar ($sum ($square d))))

;; compare trained results
(defparameter *wm* ($clone *w0*))
(defparameter *wb* ($parameter ($clone *w0*)))

;; with 100 iterations
(reinforce-simple (cartpole-fixed-env 300) *wm* 100)
(reinforce-bp (cartpole-fixed-env 300) *wb* 100)

;; compare weight values - this as well, should be almost zero
($scalar ($sum ($square ($- *wm* ($data *wb*)))))

;; with 2000 more iterations
(reinforce-simple (cartpole-fixed-env 300) *wm* 2000)
(reinforce-bp (cartpole-fixed-env 300) *wb* 2000)

;; compare weight values - this as well, should be almost zero
($scalar ($sum ($square ($- *wm* ($data *wb*)))))

;; compare without collecting gradient values but with greedy selection
(defun reinforce-simple (env w &optional (max-episodes 2000))
  (let ((gamma 0.99)
        (lr 0.01)
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
                      :for gv = ($* gm gt grad)
                      :do  ($set! w ($+ w ($* lr gv))))
                (if (null avg-score)
                    (setf avg-score score)
                    (setf avg-score (+ (* 0.9 avg-score) (* 0.1 score))))
                (when (zerop (rem e 100))
                  (prn (format nil "MG ~5D: ~8,2F / ~8,2F" e score avg-score)))))
    avg-score))

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
                      :for (action prob probs) = (select-action state w T)
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
                      :do (push ($- ($* gm logP gt)) losses))
                ($gd! w lr)
                (if (null avg-score)
                    (setf avg-score score)
                    (setf avg-score (+ (* 0.9 avg-score) (* 0.1 score))))
                (when (zerop (rem e 100))
                  (prn (format nil "AG ~5D: ~8,2F / ~8,2F" e score avg-score)))))
    avg-score))

;; compare trained results
(defparameter *wm* ($clone *w0*))
(defparameter *wb* ($parameter ($clone *w0*)))

;; with 2000 iterations
(reinforce-simple (cartpole-fixed-env 100) *wm* 2000)
(reinforce-bp (cartpole-fixed-env 100) *wb* 2000)

;; compare weight values - this as well, should be almost zero
($scalar ($sum ($square ($- *wm* ($data *wb*)))))

(evaluate (cartpole-fixed-env 200) (selector *wm*))
(evaluate (cartpole-fixed-env 200) (selector *wb*))

;; now train with probabilistic selection
(defun reinforce-simple (env w &optional (max-episodes 2000))
  (let ((gamma 0.99)
        (lr 0.01)
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
                      :for (action prob probs) = (select-action state w)
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
                      :for gv = ($* gm gt grad)
                      :do  ($set! w ($+ w ($* lr gv))))
                (if (null avg-score)
                    (setf avg-score score)
                    (setf avg-score (+ (* 0.9 avg-score) (* 0.1 score))))
                (when (zerop (rem e 100))
                  (prn (format nil "MG ~5D: ~8,2F / ~8,2F" e score avg-score)))))
    avg-score))

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
                      :for (action prob) = (select-action state w)
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
                      :do (push ($- ($* gm logP gt)) losses))
                ($gd! w lr)
                (if (null avg-score)
                    (setf avg-score score)
                    (setf avg-score (+ (* 0.9 avg-score) (* 0.1 score))))
                (when (zerop (rem e 100))
                  (prn (format nil "AG ~5D: ~8,2F / ~8,2F" e score avg-score)))))
    avg-score))

;; compare trained results
(defparameter *wm* ($clone *w0*))
(defparameter *wb* ($parameter ($clone *w0*)))

;; with 2000 iterations
(reinforce-simple (cartpole-fixed-env 200) *wm* 2000)
(reinforce-bp (cartpole-fixed-env 200) *wb* 2000)

;; compare weight values - this would be different but not much
($scalar ($sum ($square ($- *wm* ($data *wb*)))))

(evaluate (cartpole-fixed-env 1000) (selector *wm*))
(evaluate (cartpole-fixed-env 1000) (selector *wb*))
