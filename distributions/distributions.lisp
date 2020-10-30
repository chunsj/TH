(defpackage :th.distributions
  (:use #:common-lisp
        #:mu
        #:th)
  (:export #:$ll/bernoulli
           #:$ll/beta
           #:$ll/binomial
           #:$ll/chisq
           #:$ll/dice
           #:$ll/exponential
           #:$ll/gamma
           #:$ll/gaussian
           #:$ll/normal
           #:$ll/poisson
           #:$ll/t
           #:$ll/uniform
           #:$sample/bernoulli
           #:$sample/beta
           #:$sample/binomial
           #:$sample/chisq
           #:$sample/dice
           #:$sample/exponential
           #:$sample/gamma
           #:$sample/gaussian
           #:$sample/normal
           #:$sample/poisson
           #:$sample/t
           #:$sample/uniform
           #:$ll
           #:$sample
           #:$parameter-names
           #:distribution/bernoulli
           #:distribution/beta
           #:distribution/binomial
           #:distribution/chisq
           #:distribution/dice
           #:distribution/exponential
           #:distribution/gamma
           #:distribution/gaussian
           #:distribution/normal
           #:distribution/poisson
           #:distribution/t
           #:distribution/uniform
           #:$sample!
           #:$logp
           #:$observation
           #:$continuousp
           #:$propose
           #:rv/variable
           #:rv/discrete-uniform
           #:rv/exponential
           #:rv/poisson
           #:$mcmc/trace
           #:$mcmc/mle
           #:$mcmc/mean
           #:$mcmc/sd
           #:$mcmc/error
           #:$mcmc/count
           #:$mcmc/autocorrelation
           #:$mcmc/quantiles
           #:$mcmc/hpd
           #:$mcmc/geweke
           #:$mcmc/summary
           #:$mcmc/aic
           #:$mcmc/dic
           #:$mcmc/gof
           #:mh))

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
