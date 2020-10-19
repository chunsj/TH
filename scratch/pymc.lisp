(defpackage :pymc-like
  (:use #:common-lisp
        #:mu
        #:th
        #:th.distributions))

(in-package :pymc-like)

(defvar *disasters* '(4 5 4 0 1 4 3 4 0 6 3 3 4 0 2 6
                      3 3 5 4 5 3 1 4 4 1 5 5 3 4 2 5
                      2 2 3 4 2 1 3 2 2 1 1 1 1 3 0 0
                      1 0 1 1 0 0 3 1 0 3 2 2 0 1 1 1
                      0 1 0 1 0 0 0 2 1 0 0 0 1 1 0 2
                      3 3 1 1 2 1 1 1 1 2 4 2 0 0 1 4
                      0 0 0 1 0 0 0 0 0 1 0 0 1 0 1))

(let ((switch-point (r/discrete-uniform :lower 0 :upper (1- ($count *disasters*))))
      (early-mean (r/exponential :rate 1))
      (late-mean (r/exponential :rate 1)))
  (let ((disasters-early (subseq *disasters* 0 ($value switch-point)))
        (disasters-late (subseq *disasters* (1- ($value switch-point)))))
    (let ((d1 (r/poisson :rate early-mean :observation disasters-early))
          (d2 (r/poisson :rate late-mean :observation disasters-late))))))

(let ((switch-point (r/discrete-uniform :lower 0 :upper (1- ($count *disasters*))))
      (early-mean (r/exponential :rate 1))
      (late-mean (r/exponential :rate 1))
      (rate (r/lambda (lambda (s e l)
                        (let ((rates (tensor ($count *disasters*))))
                          (loop :for i :from 0 :below s
                                :do (setf ($ rates i) e))
                          (loop :for i :from s :below ($count rates)
                                :do (setf ($ rate i) l))
                          rates))))
      (disasters (r/poisson :rate rate :observation *disasters*))))
