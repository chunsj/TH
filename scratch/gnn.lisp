;; https://medium.com/@BorisAKnyazev/tutorial-on-graph-neural-networks-for-computer-vision-and-beyond-part-1-3d9fada3b80d

(defpackage :gnn-example
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers))

(in-package :gnn-example)

(defparameter *c* 2)
(defparameter *f* 8)

(defparameter *w* (affine-layer *c* *f* :activation :nil))

(defparameter *x* (rndn 1 *c*))

(prn ($execute *w* *x* :trainp nil))

(defparameter *n* 6)

(defparameter *x* (rndn *n* *c*))
(defparameter *a* (rnd *n* *n*))

(prn ($execute *w* ($@ *a* *x*) :trainp nil))
