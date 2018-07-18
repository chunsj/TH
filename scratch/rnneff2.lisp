(defpackage :rnneff
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :rnneff)

(defparameter *data* (format nil "~{~A~^~%~}" (read-lines-from "data/tinyshakespeare.txt")))
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

(defparameter *wa* ($* 0.01 (rndn *vocab-size* *hidden-size*)))
(defparameter *ua* ($* 0.01 (rndn *hidden-size* *hidden-size*)))
(defparameter *ba* (zeros 1 *hidden-size*))

(defparameter *wi* ($* 0.01 (rndn *vocab-size* *hidden-size*)))
(defparameter *ui* ($* 0.01 (rndn *hidden-size* *hidden-size*)))
(defparameter *bi* (zeros 1 *hidden-size*))

(defparameter *wf* ($* 0.01 (rndn *vocab-size* *hidden-size*)))
(defparameter *uf* ($* 0.01 (rndn *hidden-size* *hidden-size*)))
(defparameter *bf* (zeros 1 *hidden-size*))

(defparameter *wo* ($* 0.01 (rndn *vocab-size* *hidden-size*)))
(defparameter *uo* ($* 0.01 (rndn *hidden-size* *hidden-size*)))
(defparameter *bo* (zeros 1 *hidden-size*))

(defparameter *wy* ($* 0.01 (rndn *hidden-size* *vocab-size*)))
(defparameter *by* (zeros 1 *vocab-size*))

(loop :for p :from 0 :below (min 1 (- *data-size* *sequence-length* 1)) :by *sequence-length*
      :for input = (let ((m (zeros *sequence-length* *vocab-size*)))
                     (loop :for i :from p :below (+ p *sequence-length*)
                           :for ch = ($ *data* i)
                           :do (setf ($ m (- i p) ($ *char-to-idx* ch)) 1))
                     m)
      :for target = (let ((m (zeros *sequence-length* *vocab-size*)))
                      (loop :for i :from (1+ p) :below (+ p *sequence-length* 1)
                            :for ch = ($ *data* i)
                            :do (setf ($ m (- i p 1) ($ *char-to-idx* ch)) 1))
                      m)
      :do (let ((pout (zeros 1 *hidden-size*))
                (ps (zeros 1 *hidden-size*))
                (outs nil)
                (states nil)
                (losses nil))
            (loop :for i :from 0 :below *sequence-length*
                  :for xt = ($index input 0 i)
                  :for at = ($tanh ($+ ($@ xt *wa*) ($@ pout *ua*) *ba*))
                  :for it = ($sigmoid ($+ ($@ xt *wi*) ($@ pout *ui*) *bi*))
                  :for ft = ($sigmoid ($+ ($@ xt *wf*) ($@ pout *uf*) *bf*))
                  :for ot = ($sigmoid ($+ ($@ xt *wo*) ($@ pout *uo*) *bo*))
                  :for st = ($+ ($* at it) ($* ft ps))
                  :for out = ($* ($tanh st) ot)
                  :for p = ($+ ($@ out *wy*) *by*)
                  :for yt = ($softmax p)
                  :for y = ($index target 0 i)
                  :for l = ($cee yt y)
                  :do (progn
                        (setf ps st)
                        (push st states)
                        (push out outs)
                        (setf pout out)
                        (push l losses)))
            (gcf)))
