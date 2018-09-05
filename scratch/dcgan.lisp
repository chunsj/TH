;; from
;; https://github.com/soumith/dcgan.torch

(ql:quickload :opticl)

(defpackage :dcgan
  (:use #:common-lisp
        #:mu
        #:th
        #:th.db.mnist))

(in-package :dcgan)

(defparameter *nz* 100)
(defparameter *ngf* 64)
(defparameter *ndf* 64)
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

(defun generate (z)
  (-> z
      ($dconv2d *gk1* *gb1*)
      ($selu)
      ($dconv2d *gk2* *gb2* 2 2 1 1)
      ($selu)
      ($dconv2d *gk3* *gb3* 2 2 1 1)
      ($selu)
      ($dconv2d *gk4* *gb4* 2 2 1 1)
      ($selu)
      ($dconv2d *gk5* *gb5* 2 2 1 1)))

(let ((noise ($constant (rndn 10 *nz* 1 1))))
  ($cg! *generator*)
  (prn noise)
  (prn (generate noise)))

(defun initd (w) ($* 0.02 w))

(defparameter *discriminator* (parameters))
(defparameter *dk1* ($parameter *discriminator* (initd (rndn *ndf* *nc* 4 4))))
(defparameter *db1* ($parameter *discriminator* (initd (rndn *ndf*))))
(defparameter *dk2* ($parameter *discriminator* (initd (rndn (* 2 *ndf*) *ndf* 4 4))))
(defparameter *db2* ($parameter *discriminator* (initd (rndn (* 2 *ndf*)))))
(defparameter *dk3* ($parameter *discriminator* (initd (rndn (* 4 *ndf*) (* 2 *ndf*) 4 4))))
(defparameter *db3* ($parameter *discriminator* (initd (rndn (* 4 *ndf*)))))
(defparameter *dk4* ($parameter *discriminator* (initd (rndn (* 8 *ndf*) (* 4 *ndf*) 4 4))))
(defparameter *db4* ($parameter *discriminator* (initd (rndn (* 8 *ndf*)))))
(defparameter *dk5* ($parameter *discriminator* (initd (rndn 1 (* 8 *ndf*) 4 4))))
(defparameter *db5* ($parameter *discriminator* (initd (rndn 1))))

(defun discriminate (x)
  (-> x
      ($conv2d *dk1* *db1* 2 2 1 1)
      ($selu)
      ($conv2d *dk2* *db2* 2 2 1 1)
      ($selu)
      ($conv2d *dk3* *db3* 2 2 1 1)
      ($selu)
      ($conv2d *dk4* *db4* 2 2 1 1)
      ($selu)
      ($conv2d *dk5* *db5*)
      ($sigmoid)
      ($reshape 10 1)))

(let ((input ($constant (rndn 10 *nc* 64 64))))
  ($cg! *discriminator*)
  (prn input)
  (prn (discriminate input)))

;; XXX use this to resize mnist data to 64x64
;;(opticl:resize-image ...)
