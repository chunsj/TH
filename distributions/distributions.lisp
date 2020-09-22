(defpackage :th.distributions
  (:use #:common-lisp
        #:mu
        #:th)
  (:export #:$sample
           #:$score
           #:$parameter-names
           #:distribution/bernoulli
           #:distribution/binomial
           #:distribution/discrete
           #:distribution/poisson
           #:distribution/beta
           #:distribution/exponential
           #:distribution/gaussian
           #:distribution/normal
           #:distribution/uniform))

(in-package :th.distributions)

(defgeneric $sample (distribution &optional n))
(defgeneric $score (distribution data))
(defgeneric $parameter-names (distribution))

(defclass distribution () ())

(defmethod $sample ((d distribution) &optional (n 1)) (declare (ignore n)) nil)
(defmethod $score ((d distribution) data)
  (declare (ignore))
  most-negative-single-float)
(defmethod $parameter-names ((d distribution)) '())
(defmethod $parameters ((d distribution)) '())

(defun pv (pv)
  (if ($parameterp pv)
      ($data pv)
      pv))
