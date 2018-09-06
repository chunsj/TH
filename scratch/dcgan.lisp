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
(defparameter *imgw* 28)
(defparameter *imgh* 28)
(defparameter *nimg* (* *imgw* *imgh*))

(defparameter *generator* (parameters))
(defparameter *gw1* ($parameter *generator* (vxavier (list *nz* *nimg*))))
(defparameter *gb1* ($parameter *generator* (zeros 1 *nimg*)))
(defparameter *gk2* ($parameter *generator* ($* 0.01 (rndn 16 32 4 4))))
(defparameter *gb2* ($parameter *generator* ($* 0.01 (rndn 32))))
(defparameter *gk3* ($parameter *generator* ($* 0.04 (rndn 32 1 4 4))))
(defparameter *gb3* ($parameter *generator* ($* 0.04 (rndn 1))))

(defun generate (z)
  (let ((nbatch ($size z 0)))
    (-> z
        ($affine *gw1* *gb1*)
        ($reshape nbatch 16 7 7) ;; 16 plane, 7x7
        ($selu)
        ($dconv2d *gk2* *gb2* 2 2 1 1) ;; 32 plane, 14x14
        ($selu)
        ($dconv2d *gk3* *gb3* 2 2 1 1) ;; 1 plane, 28x28
        ($tanh))))

;; generator shape checking
(let* ((nbatch 10)
       (noise ($constant (rndn nbatch *nz*))))
  ($cg! *generator*)
  (prn noise)
  (prn (generate noise))
  ($cg! *generator*))

(defparameter *discriminator* (parameters))
(defparameter *dk1* ($parameter *discriminator* ($* 0.04 (rndn 32 1 4 4))))
(defparameter *db1* ($parameter *discriminator* ($* 0.04 (rndn 32))))
(defparameter *dk2* ($parameter *discriminator* ($* 0.01 (rndn 16 32 4 4))))
(defparameter *db2* ($parameter *discriminator* ($* 0.01 (rndn 16))))
(defparameter *dw3* ($parameter *discriminator* ($* 0.03 (rndn *nimg* *nimg*))))
(defparameter *db3* ($parameter *discriminator* (zeros 1 *nimg*)))
(defparameter *dw4* ($parameter *discriminator* ($* 0.04 (rndn *nimg* 1))))
(defparameter *db4* ($parameter *discriminator* (zeros 1 1)))

(defun discriminate (x)
  (let ((nbatch ($size x 0)))
    (-> x
        ($conv2d *dk1* *db1* 2 2 1 1) ;; 32 plane, 14x14
        ($lrelu)
        ($conv2d *dk2* *db2* 2 2 1 1) ;; 16 plane, 7x7
        ($selu)
        ($reshape nbatch *nimg*) ;; 1x784, flatten
        ($affine *dw3* *db3*)
        ($selu)
        ($affine *dw4* *db4*) ;; 1x1
        ($sigmoid))))

;; discriminator shape checking
(let* ((nbatch 10)
       (x ($constant (rnd nbatch 1 *imgh* *imgw*))))
  ($cg! *discriminator*)
  (prn x)
  (prn (discriminate x))
  ($cg! *discriminator*))
