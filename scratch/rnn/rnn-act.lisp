;; from
;; https://medium.com/@SeoJaeDuk/only-numpy-vanilla-recurrent-neural-network-with-activation-deriving-back-propagation-through-time-4110964a9316

(defpackage :rnn-act
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :rnn-act)

(defparameter *x* (tensor '((0 0 0)
                            (0 0 1)
                            (0 1 1))))
(defparameter *y* (tensor '((3) (2) (1))))

(defparameter *wx* (rndn 1 1))
(defparameter *wr* (rndn 1 1))

(defparameter *number-of-epoch* 15000)

(defparameter *learning-rate-wx* 0.001)
(defparameter *learning-rate-wr* 0.001)

;; original code uses this global state but i think it's wrong, state should be reset
(defparameter *state* (zeros ($size *x* 0) (1+ ($size *x* 1))))
(defparameter *grad-over-time* (apply #'zeros ($size *x*)))

(defun dsigmoid (y) ($* y ($- 1 y)))

(defun fstep (idx &optional (state *state*))
  (let* ((z ($+ ($@ ($index state 1 idx) *wr*) ($@ ($index *x* 1 idx) *wx*)))
         (a ($sigmoid z)))
    (setf ($index state 1 (1+ idx)) a)
    a))

(defun ostep (idx &optional (state *state*))
  (let* ((a ($+ ($@ ($index state 1 idx) *wr*) ($@ ($index *x* 1 idx) *wx*))))
    (setf ($index state 1 (1+ idx)) a)
    a))

(loop :for iter :from 0 :below *number-of-epoch*
      :for state = (zeros ($size *x* 0) (1+ ($size *x* 1)))
      :for s1o = (fstep 0 state)
      :for s2o = (fstep 1 state)
      :for s3i = (ostep 2 state)
      :for cost = ($/ ($sum ($expt ($- ($index state 1 3) *y*) 2)) ($size *x* 1))
      :for g2 = (let ((g ($* ($- ($index state 1 3) *y*) (/ 2 ($size *x* 1)))))
                  (setf ($index *grad-over-time* 1 2) g)
                  g)
      :for g1 = (let ((g ($* ($@ ($index *grad-over-time* 1 2) *wr*) (dsigmoid s2o))))
                  (setf ($index *grad-over-time* 1 1) g)
                  g)
      :for g0 = (let ((g ($* ($@ ($index *grad-over-time* 1 1) *wr*) (dsigmoid s1o))))
                  (setf ($index *grad-over-time* 1 0) g)
                  g)
      :for grad-wx = ($sum ($+ ($* ($index *grad-over-time* 1 2) ($index *x* 1 2))
                               ($* ($index *grad-over-time* 1 1) ($index *x* 1 1))
                               ($* ($index *grad-over-time* 1 0) ($index *x* 1 0))))
      :for grad-wr = ($sum ($+ ($* ($index *grad-over-time* 1 2) ($index state 1 2))
                               ($* ($index *grad-over-time* 1 1) ($index state 1 1))
                               ($* ($index *grad-over-time* 1 0) ($index state 1 0))))
      :do (let ((dwx ($* ($- *learning-rate-wx*) grad-wx))
                (dwr ($* ($- *learning-rate-wr*) grad-wr)))
            ($add! *wx* dwx)
            ($add! *wr* dwr)
            (when (zerop (mod iter 1000))
              (prn iter cost))))

(let ((state (zeros ($size *x* 0) (1+ ($size *x* 1)))))
  (fstep 0 state)
  (fstep 1 state)
  (prn (ostep 2 state))
  (prn ($round (ostep 2 state))))
