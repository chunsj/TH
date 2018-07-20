;; from
;; https://blog.aidangomez.ca/2016/04/17/Backpropogating-an-LSTM-A-Numerical-Example/

(defpackage :rnn-lstm
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :rnn-lstm)

;;
;; original, deep graph lstm implementation
;;

(defparameter *wa* ($variable '((0.45 0.25))))
(defparameter *ua* ($variable '((0.15))))
(defparameter *ba* ($variable '((0.2))))

(defparameter *wi* ($variable '((0.95 0.8))))
(defparameter *ui* ($variable '((0.8))))
(defparameter *bi* ($variable '((0.65))))

(defparameter *wf* ($variable '((0.7 0.45))))
(defparameter *uf* ($variable '((0.1))))
(defparameter *bf* ($variable '((0.15))))

(defparameter *wo* ($variable '((0.6 0.4))))
(defparameter *uo* ($variable '((0.25))))
(defparameter *bo* ($variable '((0.1))))

(defparameter *alpha* 0.1)

(defparameter *x* ($constant '((1 0.5) (2 3))))
(defparameter *y* ($constant '((0.5 1.25))))

(defparameter *out0* ($constant (zeros 1 1)))
(defparameter *state0* ($constant (zeros 1 1)))

(loop :for iter :from 0 :below 1
      :do (let ((pout *out0*)
                (pstate *state0*)
                (losses nil)
                (preds nil))
            (loop :for step :from 0 :below ($size *x* 1)
                  :for xt = ($index *x* 1 step)
                  :for at = ($tanh ($+ ($@ *wa* xt) ($@ *ua* pout) *ba*))
                  :for it = ($sigmoid ($+ ($@ *wi* xt) ($@ *ui* pout) *bi*))
                  :for ft = ($sigmoid ($+ ($@ *wf* xt) ($@ *uf* pout) *bf*))
                  :for ot = ($sigmoid ($+ ($@ *wo* xt) ($@ *uo* pout) *bo*))
                  :for state = ($+ ($* at it) ($* ft pstate))
                  :for out = ($* ($tanh state) ot)
                  :for y = ($index *y* 1 step)
                  :for d = ($- y out)
                  :for l = ($* ($expt d 2) ($constant 0.5))
                  :do (progn
                        (setf pout out)
                        (setf pstate state)
                        (push out preds)
                        (push l losses)))
            ($bptt! losses)
            ;; check whether the result is correct with values in the blog
            (let ((ws (list *wa* *wi* *wf* *wo*)))
              (prn "GRADS")
              (loop :for w :in ws :do (prn ($gradient w))))
            ;; after update, gradients are gone
            ($gd! ($0 losses) *alpha*)))

;;
;; broken down to reduce graph depth, this is supposed to be faster when time step is longer
;;

(defparameter *wa* ($variable '((0.45 0.25))))
(defparameter *ua* ($variable '((0.15))))
(defparameter *ba* ($variable '((0.2))))

(defparameter *wi* ($variable '((0.95 0.8))))
(defparameter *ui* ($variable '((0.8))))
(defparameter *bi* ($variable '((0.65))))

(defparameter *wf* ($variable '((0.7 0.45))))
(defparameter *uf* ($variable '((0.1))))
(defparameter *bf* ($variable '((0.15))))

(defparameter *wo* ($variable '((0.6 0.4))))
(defparameter *uo* ($variable '((0.25))))
(defparameter *bo* ($variable '((0.1))))

(defparameter *alpha* 0.1)

(defparameter *x* ($constant '((1 0.5) (2 3))))
(defparameter *y* ($constant '((0.5 1.25))))

(defparameter *out0* ($constant (zeros 1 1)))
(defparameter *state0* ($constant (zeros 1 1)))

(loop :for iter :from 0 :below 1
      :do (let ((pout *out0*)
                (pstate *state0*)
                (outs nil)
                (states nil)
                (losses nil)
                (preds nil))
            (loop :for step :from 0 :below ($size *x* 1)
                  :for xt = ($index *x* 1 step)
                  :for at = ($tanh ($+ ($@ *wa* xt) ($@ *ua* pout) *ba*))
                  :for it = ($sigmoid ($+ ($@ *wi* xt) ($@ *ui* pout) *bi*))
                  :for ft = ($sigmoid ($+ ($@ *wf* xt) ($@ *uf* pout) *bf*))
                  :for ot = ($sigmoid ($+ ($@ *wo* xt) ($@ *uo* pout) *bo*))
                  :for state = ($+ ($* at it) ($* ft pstate))
                  :for out = ($* ($tanh state) ot)
                  :for y = ($index *y* 1 step)
                  :for d = ($- y out)
                  :for l = ($* ($expt d 2) ($constant 0.5))
                  :do (progn
                        (setf pout ($variable ($clone ($data out))))
                        (setf pstate ($variable ($clone ($data state))))
                        (push (list out pout) outs)
                        (push (list state pstate) states)
                        (push out preds)
                        (push l losses)))
            ($bptt! losses)
            (setf outs (cdr outs))
            (setf states (cdr states))
            (loop :for (out pout) :in outs
                  :for cg = ($gradient pout)
                  :do ($bp! out cg))
            (loop :for (out pout) :in states
                  :for cg = ($gradient pout)
                  :do ($bp! out cg))
            ;; check whether the result is correct with values in the blog
            (let ((ws (list *wa* *wi* *wf* *wo*)))
              (prn "GRADS")
              (loop :for w :in ws :do (prn ($gradient w))))))
