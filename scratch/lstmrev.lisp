(defpackage :lstmrev
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :lstmrev)

;; use same data
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

;; weight seed values - to compare
(defparameter *wa1s* ($* 0.01 (rndn *vocab-size* *hidden-size*)))
(defparameter *ua1s* ($* 0.01 (rndn *hidden-size* *hidden-size*)))
(defparameter *ba1s* (zeros 1 *hidden-size*))

(defparameter *wi1s* ($* 0.01 (rndn *vocab-size* *hidden-size*)))
(defparameter *ui1s* ($* 0.01 (rndn *hidden-size* *hidden-size*)))
(defparameter *bi1s* (zeros 1 *hidden-size*))

(defparameter *wf1s* ($* 0.01 (rndn *vocab-size* *hidden-size*)))
(defparameter *uf1s* ($* 0.01 (rndn *hidden-size* *hidden-size*)))
(defparameter *bf1s* (zeros 1 *hidden-size*))

(defparameter *wo1s* ($* 0.01 (rndn *vocab-size* *hidden-size*)))
(defparameter *uo1s* ($* 0.01 (rndn *hidden-size* *hidden-size*)))
(defparameter *bo1s* (zeros 1 *hidden-size*))

(defparameter *wa2s* ($* 0.01 (rndn *hidden-size* *vocab-size*)))
(defparameter *ua2s* ($* 0.01 (rndn *vocab-size* *vocab-size*)))
(defparameter *ba2s* (zeros 1 *vocab-size*))

(defparameter *wi2s* ($* 0.01 (rndn *hidden-size* *vocab-size*)))
(defparameter *ui2s* ($* 0.01 (rndn *vocab-size* *vocab-size*)))
(defparameter *bi2s* (zeros 1 *vocab-size*))

(defparameter *wf2s* ($* 0.01 (rndn *hidden-size* *vocab-size*)))
(defparameter *uf2s* ($* 0.01 (rndn *vocab-size* *vocab-size*)))
(defparameter *bf2s* (zeros 1 *vocab-size*))

(defparameter *wo2s* ($* 0.01 (rndn *hidden-size* *vocab-size*)))
(defparameter *uo2s* ($* 0.01 (rndn *vocab-size* *vocab-size*)))
(defparameter *bo2s* (zeros 1 *vocab-size*))

;; weights
(defparameter *wa1* ($variable ($clone *wa1s*)))
(defparameter *ua1* ($variable ($clone *ua1s*)))
(defparameter *ba1* ($variable ($clone *ba1s*)))

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
