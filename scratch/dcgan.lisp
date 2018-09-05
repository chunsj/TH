;; from
;; https://github.com/soumith/dcgan.torch

(defpackage :dcgan
  (:use #:common-lisp
        #:mu
        #:th
        #:th.db.mnist))

(in-package :dcgan)

(defparameter *nz* 100)
(defparameter *ngf* 64)
(defparameter *nc* 1) ;; color plane

(defparameter *stdv* (/ 1 (sqrt (* *nz* (* *ngf* 8) 4 4))))
(defun initw (w) ($* *stdv* ($- ($* w 2) 1)))

(defparameter *generator* (parameters))
(defparameter *gk1* ($parameter *generator* (initw (rnd *nz* (* *ngf* 8) 4 4))))
(defparameter *gb1* ($parameter *generator* (initw (rnd (* *ngf* 8)))))
(defparameter *gk2* ($parameter *generator* (initw (rnd (* *ngf* 8) (* *ngf* 4) 4 4))))
(defparameter *gb2* ($parameter *generator* (initw (rnd (* *ngf* 4)))))
(defparameter *gk3* ($parameter *generator* (initw (rnd (* *ngf* 4) (* *ngf* 2) 4 4))))
(defparameter *gb3* ($parameter *generator* (initw (rnd (* *ngf* 2)))))
(defparameter *gk4* ($parameter *generator* (initw (rnd (* *ngf* 2) *ngf* 4 4))))
(defparameter *gb4* ($parameter *generator* (initw (rnd *ngf*))))
(defparameter *gk5* ($parameter *generator* (initw (rnd *ngf* *nc* 4 4))))
(defparameter *gb5* ($parameter *generator* (initw (rnd *nc*))))

(let ((noise ($constant (rndn 10 *nz* 1 1))))
  ($cg! *generator*)
  (prn noise)
  (prn (-> noise
           ($dconv2d *gk1* *gb1*)
           ($selu)
           ($dconv2d *gk2* *gb2* 2 2 1 1)
           ($selu)
           ($dconv2d *gk3* *gb3* 2 2 1 1)
           ($selu)
           ($dconv2d *gk4* *gb4* 2 2 1 1)
           ($selu)
           ($dconv2d *gk5* *gb5* 2 2 1 1))))
