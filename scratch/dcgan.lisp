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

(defparameter *generator* (parameters))
(defparameter *gw1* ($parameter *generator* (vxavier (list *nz* 784))))
(defparameter *gb1* ($parameter *generator* (zeros 1 784)))
(defparameter *gk2* ($parameter *generator* (rnd 16 32 4 4)))
(defparameter *gb2* ($parameter *generator* (rnd 32)))
(defparameter *gk3* ($parameter *generator* (rnd 32 1 4 4)))
(defparameter *gb3* ($parameter *generator* (rnd 1)))

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

(let* ((nbatch 10)
       (noise ($constant (rndn nbatch *nz*))))
  (prn noise)
  (prn (generate noise)))

(defparameter *discriminator* (parameters))
(defparameter *dk1* ($parameter *discriminator* (rndn 32 1 4 4)))
(defparameter *db1* ($parameter *discriminator* (rndn 32)))
(defparameter *dk2* ($parameter *discriminator* (rndn 16 32 4 4)))
(defparameter *db2* ($parameter *discriminator* (rndn 16)))
(defparameter *dw3* ($parameter *discriminator* (rndn 784 784)))
(defparameter *db3* ($parameter *discriminator* (zeros 1 784)))
(defparameter *dw4* ($parameter *discriminator* (rndn 784 1)))
(defparameter *db4* ($parameter *discriminator* (zeros 1 1)))

(defun discriminate (x)
  (let ((nbatch ($size x 0)))
    (-> x
        ($conv2d *dk1* *db1* 2 2 1 1) ;; 32 plane, 14x14
        ($selu)
        ($conv2d *dk2* *db2* 2 2 1 1) ;; 16 plane, 7x7
        ($selu)
        ($reshape nbatch 784) ;; 1x784, flatten
        ($affine *dw3* *db3*)
        ($selu)
        ($affine *dw4* *db4*)
        ($sigmoid))))

(let* ((nbatch 10)
       (x ($constant (rndn nbatch 1 28 28))))
  (prn x)
  (prn (discriminate x)))
