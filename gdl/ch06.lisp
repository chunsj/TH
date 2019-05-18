(defpackage :gdl-ch06
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :gdl-ch06)

(defparameter *streetlights* (tensor '((1 0 1)
                                       (0 1 1)
                                       (0 0 1)
                                       (1 1 1)
                                       (0 1 1)
                                       (1 0 1))))
(defparameter *walk-vs-stop* (tensor '(0 1 0 1 1 0)))

(defparameter *weights* (tensor '(0.5 0.48 -0.7)))
(defparameter *alpha* 0.1)

;; learning for single data
(let ((input ($ *streetlights* 0))
      (goal-prediction ($ *walk-vs-stop* 0)))
  (loop :for i :from 1 :to 20
        :for prediction = ($dot input *weights*)
        :for err = (expt (- prediction goal-prediction) 2)
        :for delta = (- prediction goal-prediction)
        :do (let ((nw ($- *weights* ($* *alpha* ($* input delta)))))
              (setf *weights* nw)
              (prn (list err prediction goal-prediction)))))

;; learning for every data - reset *weights*
(defparameter *weights* (tensor '(0.5 0.48 -0.7)))
(defparameter *alpha* 0.1)

(loop :for n :from 1 :to 50
      :for all-err = 0
      :do (progn
            (loop :for i :from 0 :below ($size *streetlights* 0)
                  :for input = ($ *streetlights* i)
                  :for goal-prediction = ($ *walk-vs-stop* i)
                  :for prediction = ($dot input *weights*)
                  :for err = (expt (- prediction goal-prediction) 2)
                  :for delta = (- prediction goal-prediction)
                  :do (progn
                        (setf all-err (+ all-err err))
                        (setf *weights* ($- *weights* ($* *alpha* ($* input delta))))))
            (prn (list n all-err))))

;; test learned weights
(prn ($mv *streetlights* *weights*))

;; first deep neural network
(setf ($seed th::*generator*) 101)

(defparameter *alpha* 0.2)
(defparameter *hidden-size* 4)

(defparameter *streetlights* (tensor '((1 0 1)
                                       (0 1 1)
                                       (0 0 1)
                                       (1 1 1))))
(defparameter *walk-vs-stop* ($transpose (tensor '((1 1 0 0)))))

(defparameter *weights-0-1* ($- ($* 2 (rnd 3 *hidden-size*)) 1))
(defparameter *weights-1-2* ($- ($* 2 (rnd *hidden-size* 1)) 1))

(defun relu (x) ($* (tensor ($gt x 0)) x))
(defun drelu (output) (tensor ($gt output 0)))

(loop :for n :from 1 :to 60
      :for layer-2-error = 0
      :do (progn
            (loop :for i :from 0 :below ($size *streetlights* 0)
                  :for layer-0 = ($index *streetlights* 0 (list i))
                  :for layer-1 = (relu ($mm layer-0 *weights-0-1*))
                  :for layer-2 = ($mm layer-1 *weights-1-2*)
                  :for y = ($index *walk-vs-stop* 0 (list i))
                  :for err = ($sum ($expt ($sub layer-2 y) 2))
                  :for layer-2-delta = ($sub layer-2 y)
                  :for layer-1-delta = ($* ($mm layer-2-delta ($transpose *weights-1-2*))
                                           (drelu layer-1))
                  :do (let ((dweights-1-2 ($* *alpha* ($mm ($transpose layer-1)
                                                           layer-2-delta)))
                            (dweights-0-1 ($* *alpha* ($mm ($transpose layer-0)
                                                           layer-1-delta))))
                        (setf layer-2-error (+ layer-2-error err))
                        (setf *weights-1-2* ($- *weights-1-2* dweights-1-2))
                        (setf *weights-0-1* ($- *weights-0-1* dweights-0-1))))
            (prn layer-2-error)))

;; result
(prn (-> ($@ *streetlights* *weights-0-1*)
         (relu)
         ($@ *weights-1-2*)))
