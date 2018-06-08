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
              (print (list err prediction goal-prediction)))))

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
            (print (list n all-err))))

;; test learned weights
(print ($mv *streetlights* *weights*))
