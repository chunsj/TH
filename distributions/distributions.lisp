(defpackage :th.distributions
  (:use #:common-lisp
        #:mu
        #:th)
  (:export #:$sample
           #:$ll
           #:$parameter-names
           #:distribution/bernoulli
           #:distribution/binomial
           #:distribution/discrete
           #:distribution/poisson
           #:distribution/beta
           #:distribution/exponential
           #:distribution/gaussian
           #:distribution/normal
           #:distribution/gamma
           #:distribution/t
           #:distribution/chisq
           #:distribution/uniform))

(in-package :th.distributions)

(defgeneric $sample (distribution &optional n) (:documentation "returns random sample."))
(defgeneric $ll (distribution data) (:documentation "returns log likelihood."))
(defgeneric $parameter-names (distribution) (:documentation "returns a list of parameter names."))

(defclass distribution () ())

(defmethod $sample ((d distribution) &optional (n 1)) (declare (ignore n)) nil)
(defmethod $ll ((d distribution) data)
  (declare (ignore data))
  most-negative-single-float)
(defmethod $parameter-names ((d distribution)) '())
(defmethod $parameters ((d distribution)) '())

(defun pv (pv)
  (if ($parameterp pv)
      ($data pv)
      pv))
