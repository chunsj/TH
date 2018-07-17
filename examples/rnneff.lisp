;; from
;; http://karpathy.github.io/2015/05/21/rnn-effectiveness/
;;
;; other than rnn-char in the scratch folder, this will be a test for lstm.
;; my intention is using 2 layered-lstm, could this generate better sample than
;; vanilla rnn in the scratch?
;;
;; things I want to try
;; 1. 2 layered-lstm, of course
;; 2. pre-generation of training data
;; 3. code for batch training (I think using $broadcast will do)
;;
;; as with the case of rnn-char, my code in th assumes that vectors are in row,
;; this code as well should use x * w, instead of w * x.

(defpackage :rnneff
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :rnneff)

(defparameter *data* (format nil "~{~A~^~%~}"(read-lines-from "data/tinyshakespeare.txt")))
(defparameter *chars* (remove-duplicates (coerce *data* 'list)))
(defparameter *data-size* ($count *data*))
(defparameter *vocab-size* ($count *chars*))

(defparameter *char-to-idx* (let ((ht #{}))
                              (loop :for i :from 0 :below *vocab-size*
                                    :for ch = ($ *chars* i)
                                    :do (setf ($ ht ch) i))
                              ht))
(defparameter *idx-to-char* *chars*)

(defparameter *hidden-size* 128)
(defparameter *sequence-length* 50)

(defparameter *wa1* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *ua1* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *ba1* ($variable (zeros 1 *hidden-size*)))

(defparameter *wi1* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *ui1* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *bi1* ($variable (zeros 1 *hidden-size*)))

(defparameter *wf1* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *uf1* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *bf1* ($variable (zeros 1 *hidden-size*)))

(defparameter *wo1* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *uo1* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *bo1* ($variable (zeros 1 *hidden-size*)))

(defparameter *wa2* ($variable ($* 0.01 (rndn *hidden-size* *vocab-size*))))
(defparameter *ua2* ($variable ($* 0.01 (rndn *vocab-size* *vocab-size*))))
(defparameter *ba2* ($variable (zeros 1 *vocab-size*)))

(defparameter *wi2* ($variable ($* 0.01 (rndn *hidden-size* *vocab-size*))))
(defparameter *ui2* ($variable ($* 0.01 (rndn *vocab-size* *vocab-size*))))
(defparameter *bi2* ($variable (zeros 1 *vocab-size*)))

(defparameter *wf2* ($variable ($* 0.01 (rndn *hidden-size* *vocab-size*))))
(defparameter *uf2* ($variable ($* 0.01 (rndn *vocab-size* *vocab-size*))))
(defparameter *bf2* ($variable (zeros 1 *vocab-size*)))

(defparameter *wo2* ($variable ($* 0.01 (rndn *hidden-size* *vocab-size*))))
(defparameter *uo2* ($variable ($* 0.01 (rndn *vocab-size* *vocab-size*))))
(defparameter *bo2* ($variable (zeros 1 *vocab-size*)))

;; test with single example
(let ((input (zeros *sequence-length* *vocab-size*)))
  (loop :for k :from 0 :below *sequence-length*
        :do (setf ($ input k (random *vocab-size*)) 1))
  (prn input)
  (let ((pout1 ($constant (zeros 1 *hidden-size*)))
        (ps1 ($constant (zeros 1 *hidden-size*)))
        (pout2 ($constant (zeros 1 *vocab-size*)))
        (ps2 ($constant (zeros 1 *vocab-size*))))
    (loop :for i :from 0 :below *sequence-length*
          :for xt = ($constant ($index input 0 i))
          :for at1 = ($tanh ($+ ($@ xt *wa1*) ($@ pout1 *ua1*) *ba1*))
          :for it1 = ($sigmoid ($+ ($@ xt *wi1*) ($@ pout1 *ui1*) *bi1*))
          :for ft1 = ($sigmoid ($+ ($@ xt *wf1*) ($@ pout1 *uf1*) *bf1*))
          :for ot1 = ($sigmoid ($+ ($@ xt *wo1*) ($@ pout1 *uo1*) *bo1*))
          :for st1 = ($+ ($* at1 it1) ($* ft1 ps1))
          :for out1 = ($* ($tanh st1) ot1)
          :for at2 = ($tanh ($+ ($@ out1 *wa2*) ($@ pout2 *ua2*) *ba2*))
          :for it2 = ($sigmoid ($+ ($@ out1 *wi2*) ($@ pout2 *ui2*) *bi2*))
          :for ft2 = ($sigmoid ($+ ($@ out1 *wf2*) ($@ pout2 *uf2*) *bf2*))
          :for ot2 = ($sigmoid ($+ ($@ out1 *wo2*) ($@ pout2 *uo2*) *bo2*))
          :for st2 = ($+ ($* at2 it2) ($* ft2 ps2))
          :for out2 = ($* ($tanh st2) ot2)
          :for yt = ($softmax out2)
          :do (prn yt))))
