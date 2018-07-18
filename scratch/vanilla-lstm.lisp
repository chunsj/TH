(defpackage :vanilla-lstm
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :vanilla-lstm)

(defparameter *data-lines* (read-lines-from "data/tinyshakespeare.txt"))
(defparameter *data* (format nil "~{~A~^~%~}" *data-lines*))
(defparameter *chars* (remove-duplicates (coerce *data* 'list)))
(defparameter *data-size* ($count *data*))
(defparameter *vocab-size* ($count *chars*))

(defparameter *char-to-idx* (let ((ht #{}))
                              (loop :for i :from 0 :below *vocab-size*
                                    :for ch = ($ *chars* i)
                                    :do (setf ($ ht ch) i))
                              ht))
(defparameter *idx-to-char* *chars*)

(defparameter *hidden-size* 100)
(defparameter *sequence-length* 25)
(defparameter *learning-rate* 1E-1)
(defparameter *weight-sd* 0.1)
(defparameter *z-size* (+ *hidden-size* *vocab-size*))

(defun d$sigmoid (y) ($* y ($- 1 y)))
(defun d$tanh (y) ($- 1 ($* y y)))

(defparameter *wf* ($+ ($* (rndn *z-size* *hidden-size*) *weight-sd*) 0.5))
(defparameter *bf* (zeros 1 *hidden-size*))

(defparameter *wi* ($+ ($* (rndn *z-size* *hidden-size*) *weight-sd*) 0.5))
(defparameter *bi* (zeros 1 *hidden-size*))

(defparameter *wc* ($* (rndn *z-size* *hidden-size*) *weight-sd*))
(defparameter *bc* (zeros 1 *hidden-size*))

(defparameter *wo* ($+ ($* (rndn *z-size* *hidden-size*) *weight-sd*) 0.5))
(defparameter *bo* (zeros 1 *hidden-size*))

(defparameter *wv* ($+ ($* (rndn *hidden-size* *vocab-size*) *weight-sd*) 0.5))
(defparameter *bv* (zeros 1 *vocab-size*))

(defparameter *dwf* (zeros *z-size* *hidden-size*))
(defparameter *dbf* (zeros 1 *hidden-size*))

(defparameter *dwi* (zeros *z-size* *hidden-size*))
(defparameter *dbi* (zeros 1 *hidden-size*))

(defparameter *dwc* (zeros *z-size* *hidden-size*))
(defparameter *dbc* (zeros 1 *hidden-size*))

(defparameter *dwo* (zeros *z-size* *hidden-size*))
(defparameter *dbo* (zeros 1 *hidden-size*))

(defparameter *dwv* (zeros *hidden-size* *vocab-size*))
(defparameter *dbv* (zeros 1 *vocab-size*))

(defun zero-grads ()
  ($zero! *dwf*)
  ($zero! *dbf*)
  ($zero! *dwi*)
  ($zero! *dbi*)
  ($zero! *dwc*)
  ($zero! *dbc*)
  ($zero! *dwo*)
  ($zero! *dbo*)
  ($zero! *dwv*)
  ($zero! *dbv*))

(loop :for iter :from 1 :to 1
      :for n = 0
      :for upto = (min 1 (- *data-size* *sequence-length* 1))
      :do (loop :for p :from 0 :below upto :by *sequence-length*
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
                :do (let* ((ph (zeros 1 *hidden-size*))
                           (pc (zeros 1 *hidden-size*))
                           (dhn (zeros 1 *hidden-size*))
                           (dcn (zeros 1 *hidden-size*))
                           (hs (list ph))
                           (cs (list pc))
                           (os nil)
                           (ds nil)
                           (zs nil)
                           (is nil)
                           (cbs nil)
                           (losses nil)
                           (tloss 0))
                      (loop :for i :from 0 :below (min 1 ($size input 0))
                            :for xt = ($index input 0 i)
                            :for zt = ($cat ph xt 1)
                            :for ft = ($sigmoid ($+ ($@ zt *wf*) *bf*))
                            :for it = ($sigmoid ($+ ($@ zt *wi*) *bi*))
                            :for cbt = ($tanh ($+ ($@ zt *wc*) *bc*))
                            :for ct = ($+ ($* ft pc) ($* it cbt))
                            :for ot = ($sigmoid ($+ ($@ zt *wo*) *bo*))
                            :for ht = ($* ot ($tanh ct))
                            :for vt = ($+ ($@ ht *wv*) *bv*)
                            :for yt = ($softmax vt)
                            :for y = ($index target 0 i)
                            :for l = ($cee yt y)
                            :for d = ($- yt y)
                            :do (progn
                                  (push ht hs)
                                  (push ct cs)
                                  (push ot os)
                                  (push zt zs)
                                  (push it is)
                                  (push cbt cbs)
                                  (push l losses)
                                  (push d ds)
                                  (incf tloss l)))
                      (zero-grads)
                      (loop :for i :from 0 :below (min 1 ($size input 0))
                            :for dvt = ($ ds i)
                            :for ht = ($ hs i)
                            :for ct = ($ cs i)
                            :for ot = ($ os i)
                            :for zt = ($ zs i)
                            :for cbt = ($ cbs i)
                            :for it = ($ is i)
                            :do (let ((dht nil)
                                      (dot nil)
                                      (dct nil)
                                      (dcbt nil)
                                      (dit))
                                  ($add! *dwv* ($@ ($transpose ht) dvt))
                                  ($add! *dbv* dvt)
                                  (setf dht ($@ dvt ($transpose *wv*)))
                                  (setf dht ($+ dht dhn))
                                  (setf dot ($* (d$sigmoid ot) dht ($tanh ct)))
                                  ($add! *dwo* ($@ ($transpose zt) dot))
                                  ($add! *dbo* dot)
                                  (setf dct ($+ dcn ($* dht ot (d$tanh ($tanh ct)))))
                                  (setf dcbt ($* (d$tanh cbt) dct it))
                                  ($add! *dwc* ($@ ($transpose zt) dcbt))
                                  ($add! *dbc* dcbt)
                                  (setf dit ($* (d$sigmoid it) dct cbt))
                                  ($add! *dwi* ($@ ($transpose zt) dit))
                                  ($add! *dbi* dit)
                                  (prn dcbt)))
                      (incf n))))
