(defpackage :gdl-ch03
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :gdl-ch03)

;; simple neural network
(defun neural-network (input weight)
  (* input weight))

(defparameter *number-of-toes* '(8.5 9.5 10 9))

(let ((weight 0.1))
  (prn (neural-network ($ *number-of-toes* 0) weight)))

;; multiple inputs
(defun wsum (input weights)
  (->> (mapcar #'* input weights)
       (reduce #'+)))

(defun neural-network (input weights)
  (wsum input weights))

(defparameter *ntoes* '(8.5 9.5 9.9 9.0))
(defparameter *wlrec* '(0.65 0.8 0.8 0.9))
(defparameter *nfans* '(1.2 1.3 0.5 1.0))

(let ((input (list ($ *ntoes* 0)
                   ($ *wlrec* 0)
                   ($ *nfans* 0)))
      (weight '(0.1 0.2 0)))
  (prn (neural-network input weight)))

;; multiple output
(defun neural-network (input weights)
  (mapcar (lambda (w) (* input w)) weights))

(let ((weight '(0.3 0.2 0.9)))
  (prn (neural-network ($ *wlrec* 0) weight)))

;; multiple input & multiple output
(defun neural-network (input weight)
  ($mv weight input))

(let ((weight (tensor '((0.1 0.1 -0.3)
                        (0.1 0.2 0.0)
                        (0.0 1.3 0.1))))
      (input (tensor (list ($ *ntoes* 0)
                           ($ *wlrec* 0)
                           ($ *nfans* 0)))))
  (prn (neural-network input weight)))

;; stacked!
(defun neural-network (input ih hp)
  (->> input
       ($mv ih)
       ($mv hp)))

(let ((input-to-hidden (tensor '((0.1 0.2 -0.1)
                                 (-0.1 0.1 0.9)
                                 (0.1 0.4 0.1))))
      (hidden-to-prediction (tensor '((0.3 1.1 -0.3)
                                      (0.1 0.2 0.0)
                                      (0.0 1.3 1.1))))
      (input (tensor (list ($ *ntoes* 0)
                           ($ *wlrec* 0)
                           ($ *nfans* 0)))))
  (prn (neural-network input input-to-hidden hidden-to-prediction)))
