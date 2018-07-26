;; from
;; http://karpathy.github.io/2015/05/21/rnn-effectiveness/

(defpackage :genchars2
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :genchars2)

;;
;; 2-layer lstm
;;

(defparameter *data-lines* (read-lines-from "data/tinyshakespeare.txt"))
;;(defparameter *data-lines* (read-lines-from "/Users/Sungjin/TXTDB/shakespeare.txt"))
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

(defun choose (probs)
  (let* ((sprobs ($sum probs))
         (probs ($div probs sprobs)))
    ($ ($reshape! ($multinomial probs 1) ($count probs)) 0)))

(defparameter *hidden-size* 128)
(defparameter *sequence-length* 50)

(defparameter *lstm2* (parameters))

(defparameter *wa1* ($parameter *lstm2* ($- ($* 0.16 (rnd *vocab-size* *hidden-size*)) 0.08)))
(defparameter *ua1* ($parameter *lstm2* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *ba1* ($parameter *lstm2* ($- ($* 0.16 (rnd 1 *hidden-size*)) 0.08)))

(defparameter *wi1* ($parameter *lstm2* ($- ($* 0.16 (rnd *vocab-size* *hidden-size*)) 0.08)))
(defparameter *ui1* ($parameter *lstm2* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bi1* ($parameter *lstm2* ($- ($* 0.16 (rnd 1 *hidden-size*)) 0.08)))

(defparameter *wf1* ($parameter *lstm2* ($- ($* 0.16 (rnd *vocab-size* *hidden-size*)) 0.08)))
(defparameter *uf1* ($parameter *lstm2* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bf1* ($parameter *lstm2* (ones 1 *hidden-size*)))

(defparameter *wo1* ($parameter *lstm2* ($- ($* 0.16 (rnd *vocab-size* *hidden-size*)) 0.08)))
(defparameter *uo1* ($parameter *lstm2* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bo1* ($parameter *lstm2* ($- ($* 0.16 (rnd 1 *hidden-size*)) 0.08)))

(defparameter *wa2* ($parameter *lstm2* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *ua2* ($parameter *lstm2* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *ba2* ($parameter *lstm2* ($- ($* 0.16 (rnd 1 *hidden-size*)) 0.08)))

(defparameter *wi2* ($parameter *lstm2* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *ui2* ($parameter *lstm2* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bi2* ($parameter *lstm2* ($- ($* 0.16 (rnd 1 *hidden-size*)) 0.08)))

(defparameter *wf2* ($parameter *lstm2* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *uf2* ($parameter *lstm2* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bf2* ($parameter *lstm2* (ones 1 *hidden-size*)))

(defparameter *wo2* ($parameter *lstm2* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *uo2* ($parameter *lstm2* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bo2* ($parameter *lstm2* ($- ($* 0.16 (rnd 1 *hidden-size*)) 0.08)))

(defparameter *wy* ($parameter *lstm2* ($- ($* 0.16 (rndn *hidden-size* *vocab-size*)) 0.08)))
(defparameter *by* ($parameter *lstm2* ($- ($* 0.16 (rnd 1 *vocab-size*)) 0.08)))

(defun sample (h1 o1 h2 o2 seed-idx n &optional (temperature 1))
  (let ((x (zeros 1 *vocab-size*))
        (indices (list seed-idx))
        (hs1 ($constant h1))
        (os1 ($constant o1))
        (hs2 ($constant h2))
        (os2 ($constant o2)))
    (setf ($ x 0 seed-idx) 1)
    (loop :for i :from 0 :below n
          :for xt = ($constant x)
          :for at1 = ($tanh ($+ ($@ xt *wa1*) ($@ os1 *ua1*) *ba1*))
          :for it1 = ($sigmoid ($+ ($@ xt *wi1*) ($@ os1 *ui1*) *bi1*))
          :for ft1 = ($sigmoid ($+ ($@ xt *wf1*) ($@ os1 *uf1*) *bf1*))
          :for ot1 = ($sigmoid ($+ ($@ xt *wo1*) ($@ os1 *uo1*) *bo1*))
          :for st1 = ($+ ($* at1 it1) ($* ft1 hs1))
          :for out1 = ($* ($tanh st1) ot1)
          :for at2 = ($tanh ($+ ($@ out1 *wa2*) ($@ os2 *ua2*) *ba2*))
          :for it2 = ($sigmoid ($+ ($@ out1 *wi2*) ($@ os2 *ui2*) *bi2*))
          :for ft2 = ($sigmoid ($+ ($@ out1 *wf2*) ($@ os2 *uf2*) *bf2*))
          :for ot2 = ($sigmoid ($+ ($@ out1 *wo2*) ($@ os2 *uo2*) *bo2*))
          :for st2 = ($+ ($* at2 it2) ($* ft2 hs2))
          :for out2 = ($* ($tanh st2) ot2)
          :for yt = ($+ ($@ out2 *wy*) *by*)
          :for ps = ($softmax ($/ yt ($constant temperature)))
          :for nidx = (choose ($data ps))
          :do (progn
                (setf hs1 st1)
                (setf os1 out1)
                (setf hs2 st2)
                (setf os2 out2)
                (unless (typep nidx 'number)
                  (prn nidx))
                (push nidx indices)
                ($zero! x)
                (setf ($ x 0 nidx) 1)))
    ($cg! *lstm2*)
    (coerce (mapcar (lambda (i) ($ *idx-to-char* i)) (reverse indices)) 'string)))

(loop :for iter :from 1 :to 1
      :for n = 0
      :for upto = (max 1 (- *data-size* *sequence-length* 1))
      :do (loop :for p :from 0 :below upto :by *sequence-length*
                :for input-str = (subseq *data* p (+ p *sequence-length*))
                :for target-str = (subseq *data* (1+ p) (+ p *sequence-length* 1))
                :for input = (let ((m (zeros *sequence-length* *vocab-size*)))
                               (loop :for i :from 0 :below *sequence-length*
                                     :for ch = ($ input-str i)
                                     :do (setf ($ m i ($ *char-to-idx* ch)) 1))
                               m)
                :for target = (let ((m (zeros *sequence-length* *vocab-size*)))
                                (loop :for i :from 0 :below *sequence-length*
                                      :for ch = ($ target-str i)
                                      :do (setf ($ m i ($ *char-to-idx* ch)) 1))
                                m)
                :do (let ((hs1 ($constant (zeros 1 *hidden-size*)))
                          (os1 ($constant (zeros 1 *hidden-size*)))
                          (hs2 ($constant (zeros 1 *hidden-size*)))
                          (os2 ($constant (zeros 1 *hidden-size*)))
                          (losses nil)
                          (tloss 0))
                      ($cg! *lstm2*)
                      (loop :for i :from 0 :below ($size input 0)
                            :for xt = ($constant ($index input 0 i))
                            :for at1 = ($tanh ($+ ($@ xt *wa1*) ($@ os1 *ua1*) *ba1*))
                            :for it1 = ($sigmoid ($+ ($@ xt *wi1*) ($@ os1 *ui1*) *bi1*))
                            :for ft1 = ($sigmoid ($+ ($@ xt *wf1*) ($@ os1 *uf1*) *bf1*))
                            :for ot1 = ($sigmoid ($+ ($@ xt *wo1*) ($@ os1 *uo1*) *bo1*))
                            :for st1 = ($+ ($* at1 it1) ($* ft1 hs1))
                            :for out1 = ($* ($tanh st1) ot1)
                            :for at2 = ($tanh ($+ ($@ out1 *wa2*) ($@ os2 *ua2*) *ba2*))
                            :for it2 = ($sigmoid ($+ ($@ out1 *wi2*) ($@ os2 *ui2*) *bi2*))
                            :for ft2 = ($sigmoid ($+ ($@ out1 *wf2*) ($@ os2 *uf2*) *bf2*))
                            :for ot2 = ($sigmoid ($+ ($@ out1 *wo2*) ($@ os2 *uo2*) *bo2*))
                            :for st2 = ($+ ($* at2 it2) ($* ft2 hs2))
                            :for out2 = ($* ($tanh st2) ot2)
                            :for yt = ($+ ($@ out2 *wy*) *by*)
                            :for ps = ($softmax yt)
                            :for y = ($constant ($index target 0 i))
                            :for l = ($cee ps y)
                            :do (progn
                                  (setf hs1 st1)
                                  (setf os1 out1)
                                  (setf hs2 st2)
                                  (setf os2 out2)
                                  (incf tloss ($data l))
                                  (push l losses)))
                      ($adgd! *lstm2*)
                      (when (zerop (rem n 10))
                        (prn "")
                        (prn "[ITER]" n (/ tloss (* 1.0 *sequence-length*)))
                        (prn (sample ($data hs1)
                                     ($data os1)
                                     ($data hs2)
                                     ($data os2)
                                     ($ *char-to-idx* ($ input-str 0))
                                     72))
                        (prn "")
                        (gcf))
                      (incf n))))

(prn (sample (zeros 1 *hidden-size*) (zeros 1 *hidden-size*)
             (zeros 1 *hidden-size*) (zeros 1 *hidden-size*)
             (random *vocab-size*) 200))
