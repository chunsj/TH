(defpackage :affine-work
  (:use #:common-lisp
        #:th
        #:mu))

(in-package :affine-work)

;; to find out how one can efficiently compute affine transformation
;; or the so called linear layer

(defparameter *weight* (rndn 3 4))
(defparameter *input* (tensor '((1 1 1)
                                (2 2 2))))
(defparameter *bias* (tensor '(1 1 1 1)))
(defparameter *bias2* (tensor '((1 1 1 1))))

(defparameter *1d-input* (tensor '(1 2 3)))
($mm ($reshape *1d-input* 1 3) *weight*)
($mv ($transpose *weight*) *1d-input*)

;; preliminnary test
(prn ($@ *input* *weight*))

(defun allocate-addbuf (nframe)
  (let ((tensor (make-instance th::*default-tensor-class*)))
    (th::allocate-tensor-handle tensor (list nframe))
    ($one! tensor)))

(defun deallocate-addbuf (addbuf)
  (th::deallocate-tensor-handle addbuf))

(let* ((addbuf (allocate-addbuf *input*)))
  (prn addbuf)
  (th::deallocate-tensor-handle addbuf)
  (prn addbuf))

(defun affine (x w b &optional ones)
  (let ((dim ($ndim x)))
    (cond ((eq dim 1) (let ((output ($copy! ($resize! ($empty x) (list ($size w 1))) b)))
                        ($addmv! output x w 1 1)
                        output))
          ((eq dim 2) (let* ((nframe ($size x 0))
                             (output ($zero! ($resize! ($empty x) (list nframe ($size w 1)))))
                             (addbuf (or ones (allocate-addbuf nframe))))
                        ($addmm! output x w 1 0)
                        ($addr! output addbuf b 1 1)
                        (when (null ones) (deallocate-addbuf addbuf))
                        output)))))

($affine *input* *weight* *bias2*)
($xwpb *input* *weight* *bias*)

(defparameter *os* (ones ($size *input* 0)))

;; 0.9xx
(progn
  (gcf)
  (time (loop :for i :from 0 :below 100000
              :do ($xwpb *input* *weight* *bias*)))
  (gcf))

;; 0.6xx - 30% better
(progn
  (gcf)
  (time (loop :for i :from 0 :below 100000
              :do (affine *input* *weight* *bias*)))
  (gcf))

;; 0.6xx
(progn
  (gcf)
  (time (loop :for i :from 0 :below 100000
              :do ($xwpb *input* *weight* *bias* *os*)))
  (gcf))

;; 0.3xx - 40% better
(progn
  (gcf)
  (time (loop :for i :from 0 :below 100000
              :do (affine *input* *weight* *bias* *os*)))
  (gcf))

;; from torch linear layer
(let* ((dim ($ndim *input*))
       (nframe ($size *input* 0))
       (output ($zero! ($resize! ($empty *input*) (list nframe ($size *weight* 1)))))
       (addbuf ($one! ($resize! ($empty *input*) (list nframe)))))
  ($addmm! output *input* *weight* 1 0)
  (prn output)
  ($addr! output addbuf *bias* 1 1)
  (prn output)
  (prn ($- output (tensor '((1 1 1 1) (1 1 1 1))))))
