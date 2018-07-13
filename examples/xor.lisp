(defpackage :xor-example
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :xor-example)

;; because there's no such neural network library without xor example

;; direct, without using ad.
(defun fwd (input weight) ($sigmoid! ($@ input weight)))
(defun dwb (delta output) ($* delta output ($- 1 output)))

(let* ((X (tensor '((0 0 1) (0 1 1) (1 0 1) (1 1 1))))
       (Y (tensor '((0) (1) (1) (0))))
       (w1 (rndn 3 3))
       (w2 (rndn 3 1))
       (lr 1))
  (loop :for i :from 0 :below 1000
        :do (let* ((l1 (fwd X w1))
                   (l2 (fwd l1 w2))
                   (l2d (dwb ($- l2 y) l2))
                   (l1d (dwb ($@ l2d ($transpose w2)) l1))
                   (dw2 ($@ ($transpose l1) l2d))
                   (dw1 ($@ ($transpose X) l1d)))
              ($sub! w1 ($* lr dw1))
              ($sub! w2 ($* lr dw2))))
  (prn (fwd (fwd X w1) w2)))

;; using ad or autograd
(let* ((w1 ($variable (rndn 3 3)))
       (w2 ($variable (rndn 3 1)))
       (X ($constant '((0 0 1) (0 1 1) (1 0 1) (1 1 1))))
       (Y ($constant '(0 1 1 0)))
       (lr 1))
  (loop :for i :from 0 :below 1000
        :do (let* ((l1 ($sigmoid ($mm X w1)))
                   (l2 ($sigmoid ($mm l1 w2)))
                   (d ($sub l2 Y))
                   (out ($dot d d)))
              ($bp! out)
              ($gd! out lr)))
  (prn ($sigmoid ($mm ($sigmoid ($mm X w1)) w2))))

(let* ((w1 ($variable (rndn 3 3)))
       (w2 ($variable (rndn 3 1)))
       (X ($constant '((0 0 1) (0 1 1) (1 0 1) (1 1 1))))
       (Y ($constant '(0 1 1 0)))
       (lr 1))
  (loop :for i :from 0 :below 1000
        :do (let* ((l1 ($sigmoid ($mm X w1)))
                   (l2 ($sigmoid ($mm l1 w2)))
                   (d ($sub l2 Y))
                   (out ($dot d d)))
              ($bp! out)
              ($mgd! out lr)))
  (prn ($sigmoid ($mm ($sigmoid ($mm X w1)) w2))))

(let* ((w1 ($variable (rndn 3 3)))
       (w2 ($variable (rndn 3 1)))
       (X ($constant '((0 0 1) (0 1 1) (1 0 1) (1 1 1))))
       (Y ($constant '(0 1 1 0)))
       (lr 1))
  (loop :for i :from 0 :below 1000
        :do (let* ((l1 ($sigmoid ($mm X w1)))
                   (l2 ($sigmoid ($mm l1 w2)))
                   (d ($sub l2 Y))
                   (out ($dot d d)))
              ($bp! out)
              ($agd! out lr)))
  (prn ($sigmoid ($mm ($sigmoid ($mm X w1)) w2))))

(let* ((w1 ($variable (rndn 3 3)))
       (w2 ($variable (rndn 3 1)))
       (X ($constant '((0 0 1) (0 1 1) (1 0 1) (1 1 1))))
       (Y ($constant '(0 1 1 0)))
       (lr 0.01))
  (loop :for i :from 0 :below 1000
        :do (let* ((l1 ($sigmoid ($mm X w1)))
                   (l2 ($sigmoid ($mm l1 w2)))
                   (d ($sub l2 Y))
                   (out ($dot d d)))
              ($bp! out)
              ($amgd! out lr)))
  (prn ($sigmoid ($mm ($sigmoid ($mm X w1)) w2))))


(let* ((w1 ($variable (rndn 3 3)))
       (w2 ($variable (rndn 3 1)))
       (X ($constant '((0 0 1) (0 1 1) (1 0 1) (1 1 1))))
       (Y ($constant '(0 1 1 0)))
       (lr 0.1))
  (loop :for i :from 0 :below 1000
        :do (let* ((l1 ($sigmoid ($mm X w1)))
                   (l2 ($sigmoid ($mm l1 w2)))
                   (d ($sub l2 Y))
                   (out ($dot d d)))
              ($bp! out)
              ($rmgd! out lr)))
  (prn ($sigmoid ($mm ($sigmoid ($mm X w1)) w2))))

(let* ((w1 ($variable (rndn 3 3)))
       (w2 ($variable (rndn 3 1)))
       (X ($constant '((0 0 1) (0 1 1) (1 0 1) (1 1 1))))
       (Y ($constant '(0 1 1 0))))
  (loop :for i :from 0 :below 1000
        :do (let* ((l1 ($sigmoid ($mm X w1)))
                   (l2 ($sigmoid ($mm l1 w2)))
                   (d ($sub l2 Y))
                   (out ($dot d d)))
              ($bp! out)
              ($adgd! out)))
  (prn ($sigmoid ($mm ($sigmoid ($mm X w1)) w2))))
