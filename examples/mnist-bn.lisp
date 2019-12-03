(defpackage :mnist-bn
  (:use #:common-lisp
        #:mu
        #:th
        #:th.db.mnist
        #:th.db.fashion))

(in-package :mnist-bn)

;; use one of following
(defparameter *mnist* (read-mnist-data))
(defparameter *mnist* (read-fashion-data))

(defparameter *data-size* 500)

(defparameter *x-train* ($index ($ *mnist* :train-images) 0 (xrange 0 *data-size*)))
(defparameter *y-train* ($index ($ *mnist* :train-labels) 0 (xrange 0 *data-size*)))

(defparameter *batch-size* 100)
(defparameter *batch-count* (/ *data-size* *batch-size*))

(defparameter *x-batches*
  (loop :for i :from 0 :below *batch-count*
        :for rng = (loop :for k :from (* i *batch-size*) :below (* (1+ i) *batch-size*)
                         :collect k)
        :collect ($contiguous! ($index *x-train* 0 rng))))
(defparameter *y-batches*
  (loop :for i :from 0 :below *batch-count*
        :for rng = (loop :for k :from (* i *batch-size*) :below (* (1+ i) *batch-size*)
                         :collect k)
        :collect ($contiguous! ($index *y-train* 0 rng))))

(defparameter *input-size* 784)
(defparameter *weight-size* 100)
(defparameter *output-size* 10)

(defparameter *w01* (vhe '(*input-size* *weight-size*)))
(defparameter *b01* ($parameter (zeros *weight-size*)))
(defparameter *w02* (vhe '(*weight-size* *weight-size*)))
(defparameter *b02* ($parameter (zeros *weight-size*)))
(defparameter *w03* (vhe '(*weight-size* *weight-size*)))
(defparameter *b03* ($parameter (zeros *weight-size*)))
(defparameter *w04* (vhe '(*weight-size* *weight-size*)))
(defparameter *b04* ($parameter (zeros *weight-size*)))
(defparameter *w05* (vhe '(*weight-size* *weight-size*)))
(defparameter *b05* ($parameter (zeros *weight-size*)))
(defparameter *w06* (vhe '(*weight-size* *output-size*)))
(defparameter *b06* ($parameter (zeros *output-size*)))

(defparameter *p01* (list *w01* *b01* *w02* *b02* *w03* *b03*
                          *w04* *b04* *w05* *b05* *w06* *b06*))

(defparameter *w11* (vhe '(*input-size* *weight-size*)))
(defparameter *b11* ($parameter (zeros *weight-size*)))
(defparameter *g11* ($parameter (ones *input-size*)))
(defparameter *e11* ($parameter (zeros *input-size*)))
(defparameter *rm11* (zeros *input-size*))
(defparameter *rv11* (ones *input-size*))
(defparameter *sm11* (zeros *input-size*))
(defparameter *sd11* (ones *input-size*))
(defparameter *w12* (vhe '(*weight-size* *weight-size*)))
(defparameter *b12* ($parameter (zeros *weight-size*)))
(defparameter *g12* ($parameter (ones *weight-size*)))
(defparameter *e12* ($parameter (zeros *weight-size*)))
(defparameter *rm12* (zeros *weight-size*))
(defparameter *rv12* (ones *weight-size*))
(defparameter *sm12* (zeros *weight-size*))
(defparameter *sd12* (ones *weight-size*))
(defparameter *w13* (vhe '(*weight-size* *weight-size*)))
(defparameter *b13* ($parameter (zeros *weight-size*)))
(defparameter *g13* ($parameter (ones *weight-size*)))
(defparameter *e13* ($parameter (zeros *weight-size*)))
(defparameter *rm13* (zeros *weight-size*))
(defparameter *rv13* (ones *weight-size*))
(defparameter *sm13* (zeros *weight-size*))
(defparameter *sd13* (ones *weight-size*))
(defparameter *w14* (vhe '(*weight-size* *weight-size*)))
(defparameter *b14* ($parameter (zeros *weight-size*)))
(defparameter *g14* ($parameter (ones *weight-size*)))
(defparameter *e14* ($parameter (zeros *weight-size*)))
(defparameter *rm14* (zeros *weight-size*))
(defparameter *rv14* (ones *weight-size*))
(defparameter *sm14* (zeros *weight-size*))
(defparameter *sd14* (ones *weight-size*))
(defparameter *w15* (vhe '(*weight-size* *weight-size*)))
(defparameter *b15* ($parameter (zeros *weight-size*)))
(defparameter *g15* ($parameter (ones *weight-size*)))
(defparameter *e15* ($parameter (zeros *weight-size*)))
(defparameter *rm15* (zeros *weight-size*))
(defparameter *rv15* (ones *weight-size*))
(defparameter *sm15* (zeros *weight-size*))
(defparameter *sd15* (ones *weight-size*))
(defparameter *w16* (vhe '(*weight-size* *output-size*)))
(defparameter *b16* ($parameter (zeros *output-size*)))

(defparameter *p02* (list *w11* *b11* *w12* *b12* *w13* *b13*
                          *w14* *b14* *w15* *b15* *w16* *b16*
                          *g11* *e11* *g12* *e12* *g13* *e13*
                          *g14* *e14* *g15* *e15*))
(defparameter *a02* (list *rm11* *rv11* *sm11* *sd11*
                          *rm12* *rv12* *sm12* *sd12*
                          *rm13* *rv13* *sm13* *sd13*
                          *rm14* *rv14* *sm14* *sd14*
                          *rm15* *rv15* *sm15* *sd15*))

(defparameter *w21* (vhe '(*input-size* *weight-size*)))
(defparameter *b21* ($parameter (zeros *weight-size*)))
(defparameter *w22* (vhe '(*weight-size* *weight-size*)))
(defparameter *b22* ($parameter (zeros *weight-size*)))
(defparameter *w23* (vhe '(*weight-size* *weight-size*)))
(defparameter *b23* ($parameter (zeros *weight-size*)))
(defparameter *w24* (vhe '(*weight-size* *weight-size*)))
(defparameter *b24* ($parameter (zeros *weight-size*)))
(defparameter *w25* (vhe '(*weight-size* *weight-size*)))
(defparameter *b25* ($parameter (zeros *weight-size*)))
(defparameter *w26* (vhe '(*weight-size* *output-size*)))
(defparameter *b26* ($parameter (zeros *output-size*)))

(defparameter *p03* (list *w21* *b21* *w22* *b22* *w23* *b23*
                          *w24* *b24* *w25* *b25* *w26* *b26*))

(defparameter *w31* (vhe '(*input-size* *weight-size*)))
(defparameter *b31* ($parameter (zeros *weight-size*)))
(defparameter *w32* (vhe '(*weight-size* *weight-size*)))
(defparameter *b32* ($parameter (zeros *weight-size*)))
(defparameter *w33* (vhe '(*weight-size* *weight-size*)))
(defparameter *b33* ($parameter (zeros *weight-size*)))
(defparameter *w34* (vhe '(*weight-size* *weight-size*)))
(defparameter *b34* ($parameter (zeros *weight-size*)))
(defparameter *w35* (vhe '(*weight-size* *weight-size*)))
(defparameter *b35* ($parameter (zeros *weight-size*)))
(defparameter *w36* (vhe '(*weight-size* *output-size*)))
(defparameter *b36* ($parameter (zeros *output-size*)))

(defparameter *p04* (list *w31* *b31* *w32* *b32* *w33* *b33*
                          *w34* *b34* *w35* *b35* *w36* *b36*))

(defparameter *w41* (vhe '(*input-size* *weight-size*)))
(defparameter *b41* ($parameter (zeros *weight-size*)))
(defparameter *w42* (vhe '(*weight-size* *weight-size*)))
(defparameter *b42* ($parameter (zeros *weight-size*)))
(defparameter *w43* (vhe '(*weight-size* *weight-size*)))
(defparameter *b43* ($parameter (zeros *weight-size*)))
(defparameter *w44* (vhe '(*weight-size* *weight-size*)))
(defparameter *b44* ($parameter (zeros *weight-size*)))
(defparameter *w45* (vhe '(*weight-size* *weight-size*)))
(defparameter *b45* ($parameter (zeros *weight-size*)))
(defparameter *w46* (vhe '(*weight-size* *output-size*)))
(defparameter *b46* ($parameter (zeros *output-size*)))

(defparameter *p05* (list *w41* *b41* *w42* *b42* *w43* *b43*
                          *w44* *b44* *w45* *b45* *w46* *b46*))


(defun single-step (params x)
  (let ((w1 ($ params 0))
        (b1 ($ params 1))
        (w2 ($ params 2))
        (b2 ($ params 3))
        (w3 ($ params 4))
        (b3 ($ params 5))
        (w4 ($ params 6))
        (b4 ($ params 7))
        (w5 ($ params 8))
        (b5 ($ params 9))
        (w6 ($ params 10))
        (b6 ($ params 11)))
    (-> x
        ($affine w1 b1)
        ($relu)
        ($affine w2 b2)
        ($relu)
        ($affine w3 b3)
        ($relu)
        ($affine w4 b4)
        ($relu)
        ($affine w5 b5)
        ($relu)
        ($affine w6 b6)
        ($softmax))))

(defun single-step-bn (params aps x)
  (let ((w1 ($ params 0))
        (b1 ($ params 1))
        (w2 ($ params 2))
        (b2 ($ params 3))
        (w3 ($ params 4))
        (b3 ($ params 5))
        (w4 ($ params 6))
        (b4 ($ params 7))
        (w5 ($ params 8))
        (b5 ($ params 9))
        (w6 ($ params 10))
        (b6 ($ params 11))
        (g1 ($ params 12))
        (e1 ($ params 13))
        (rm1 ($ aps 0))
        (rv1 ($ aps 1))
        (sm1 ($ aps 2))
        (sd1 ($ aps 3))
        (g2 ($ params 14))
        (e2 ($ params 15))
        (rm2 ($ aps 4))
        (rv2 ($ aps 5))
        (sm2 ($ aps 6))
        (sd2 ($ aps 7))
        (g3 ($ params 16))
        (e3 ($ params 17))
        (rm3 ($ aps 8))
        (rv3 ($ aps 9))
        (sm3 ($ aps 10))
        (sd3 ($ aps 11))
        (g4 ($ params 18))
        (e4 ($ params 19))
        (rm4 ($ aps 12))
        (rv4 ($ aps 13))
        (sm4 ($ aps 14))
        (sd4 ($ aps 15))
        (g5 ($ params 20))
        (e5 ($ params 21))
        (rm5 ($ aps 16))
        (rv5 ($ aps 17))
        (sm5 ($ aps 18))
        (sd5 ($ aps 19)))
    (-> x
        ($affine w1 b1)
        ($bn g1 e1 rm1 rv1 sm1 sd1)
        ($relu)
        ($affine w2 b2)
        ($bn g2 e2 rm2 rv2 sm2 sd2)
        ($relu)
        ($affine w3 b3)
        ($bn g3 e3 rm3 rv3 sm3 sd3)
        ($relu)
        ($affine w4 b4)
        ($bn g4 e4 rm4 rv4 sm4 sd4)
        ($relu)
        ($affine w5 b5)
        ($bn g5 e5 rm5 rv5 sm5 sd5)
        ($relu)
        ($affine w6 b6)
        ($softmax))))

(defun single-step-snn (params x)
  (let ((w1 ($ params 0))
        (b1 ($ params 1))
        (w2 ($ params 2))
        (b2 ($ params 3))
        (w3 ($ params 4))
        (b3 ($ params 5))
        (w4 ($ params 6))
        (b4 ($ params 7))
        (w5 ($ params 8))
        (b5 ($ params 9))
        (w6 ($ params 10))
        (b6 ($ params 11)))
    (-> x
        ($affine w1 b1)
        ($selu)
        ($affine w2 b2)
        ($selu)
        ($affine w3 b3)
        ($selu)
        ($affine w4 b4)
        ($selu)
        ($affine w5 b5)
        ($selu)
        ($affine w6 b6)
        ($softmax))))

(defun single-step-swish (params x)
  (let ((w1 ($ params 0))
        (b1 ($ params 1))
        (w2 ($ params 2))
        (b2 ($ params 3))
        (w3 ($ params 4))
        (b3 ($ params 5))
        (w4 ($ params 6))
        (b4 ($ params 7))
        (w5 ($ params 8))
        (b5 ($ params 9))
        (w6 ($ params 10))
        (b6 ($ params 11)))
    (-> x
        ($affine w1 b1)
        ($swish)
        ($affine w2 b2)
        ($swish)
        ($affine w3 b3)
        ($swish)
        ($affine w4 b4)
        ($swish)
        ($affine w5 b5)
        ($swish)
        ($affine w6 b6)
        ($softmax))))

(defun single-step-mish (params x)
  (let ((w1 ($ params 0))
        (b1 ($ params 1))
        (w2 ($ params 2))
        (b2 ($ params 3))
        (w3 ($ params 4))
        (b3 ($ params 5))
        (w4 ($ params 6))
        (b4 ($ params 7))
        (w5 ($ params 8))
        (b5 ($ params 9))
        (w6 ($ params 10))
        (b6 ($ params 11)))
    (-> x
        ($affine w1 b1)
        ($mish)
        ($affine w2 b2)
        ($mish)
        ($affine w3 b3)
        ($mish)
        ($affine w4 b4)
        ($mish)
        ($affine w5 b5)
        ($mish)
        ($affine w6 b6)
        ($softmax))))

($cg! *p01*)
(let ((y* (single-step *p01* *x-train*)))
  ($cee y* *y-train*)
  ($cg! *p01*))

($cg! *p02*)
(let ((y* (single-step-bn *p02* *a02* *x-train*)))
  ($cee y* *y-train*)
  ($cg! *p02*))

($cg! *p03*)
(let ((y* (single-step-snn *p03* *x-train*)))
  ($cee y* *y-train*)
  ($cg! *p03*))

($cg! *p04*)
(let ((y* (single-step-swish *p04* *x-train*)))
  ($cee y* *y-train*)
  ($cg! *p04*))

($cg! *p05*)
(let ((y* (single-step-mish *p05* *x-train*)))
  ($cee y* *y-train*)
  ($cg! *p05*))

(defparameter *epochs* 500)

;; 49.22, 52.18 secs
(progn
  ($cg! *p01*)
  (loop :for epoch :from 1 :to *epochs*
        :do (loop :for xb :in *x-batches*
                  :for yb :in *y-batches*
                  :for i :from 0
                  :for y* = (single-step *p01* xb)
                  :for l = ($cee y* yb)
                  :do (progn
                        (when (and (zerop (rem epoch 100))
                                   (zerop i))
                          (prn (format nil "[~A] ~A" epoch l)))
                        ($adgd! *p01*)))))

;; 66.33, 66.29 secs
(progn
  ($cg! *p02*)
  (loop :for epoch :from 1 :to *epochs*
        :do (loop :for xb :in *x-batches*
                  :for yb :in *y-batches*
                  :for i :from 0
                  :for y* = (single-step-bn *p02* *a02* xb)
                  :for l = ($cee y* yb)
                  :do (progn
                        (when (and (zerop (rem epoch 100))
                                   (zerop i))
                          (prn (format nil "[~A] ~A" epoch l)))
                        ($adgd! *p02*)))))

;; 51.37, 57.52 secs
(progn
  ($cg! *p03*)
  (loop :for epoch :from 1 :to *epochs*
        :do (loop :for xb :in *x-batches*
                  :for yb :in *y-batches*
                  :for i :from 0
                  :for y* = (single-step-snn *p03* xb)
                  :for l = ($cee y* yb)
                  :do (progn
                        (when (and (zerop (rem epoch 100))
                                   (zerop i))
                          (prn (format nil "[~A] ~A" epoch l)))
                        ($adgd! *p03*)))))

;; 50.82, 51.67 secs
(progn
  ($cg! *p04*)
  (loop :for epoch :from 1 :to *epochs*
        :do (loop :for xb :in *x-batches*
                  :for yb :in *y-batches*
                  :for i :from 0
                  :for y* = (single-step-swish *p04* xb)
                  :for l = ($cee y* yb)
                  :do (progn
                        (when (and (zerop (rem epoch 100))
                                   (zerop i))
                          (prn (format nil "[~A] ~A" epoch l)))
                        ($adgd! *p04*)))))

;; 70.26, 69.76 secs
(progn
  ($cg! *p05*)
  (loop :for epoch :from 1 :to *epochs*
        :do (loop :for xb :in *x-batches*
                  :for yb :in *y-batches*
                  :for i :from 0
                  :for y* = (single-step-mish *p05* xb)
                  :for l = ($cee y* yb)
                  :do (progn
                        (when (and (zerop (rem epoch 100))
                                   (zerop i))
                          (prn (format nil "[~A] ~A" epoch l)))
                        ($adgd! *p05*)))))
