(defpackage :gdl-ch11
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :gdl-ch11)

;; onehots
(defparameter *onehots* #{})

(setf ($ *onehots* "cat") (tensor '(1 0 0 0)))
(setf ($ *onehots* "the") (tensor '(0 1 0 0)))
(setf ($ *onehots* "dog") (tensor '(0 0 1 0)))
(setf ($ *onehots* "sat") (tensor '(0 0 0 1)))

(defun word2hot (w) ($ *onehots* w))

(let ((sentence '("the" "cat" "sat")))
  (print (reduce #'$+ (mapcar #'word2hot sentence))))
