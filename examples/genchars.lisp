;; from
;; http://karpathy.github.io/2015/05/21/rnn-effectiveness/

(defpackage :genchars
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :genchars)

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

(defun choose (probs)
  (let* ((sprobs ($sum probs))
         (probs ($div probs sprobs)))
    ($ ($reshape! ($multinomial probs 1) ($count probs)) 0)))

;;
;; vanilla rnn
;;

(defparameter *hidden-size* 100)
(defparameter *sequence-length* 25)

(defparameter *rnn* (parameters))
(defparameter *wx* ($parameter *rnn* ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *wh* ($parameter *rnn* ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *wy* ($parameter *rnn* ($* 0.01 (rndn *hidden-size* *vocab-size*))))
(defparameter *bh* ($parameter *rnn* (zeros 1 *hidden-size*)))
(defparameter *by* ($parameter *rnn* (zeros 1 *vocab-size*)))

(defun sample (h seed-idx n &optional (temperature 1))
  (let ((x (zeros 1 *vocab-size*))
        (indices (list seed-idx))
        (ph ($constant h)))
    (setf ($ x 0 seed-idx) 1)
    (loop :for i :from 0 :below n
          :for xt = ($constant x)
          :for ht = ($tanh ($+ ($@ xt *wx*) ($@ ph *wh*) *bh*))
          :for yt = ($+ ($@ ht *wy*) *by*)
          :for ps = ($softmax ($/ yt ($constant temperature)))
          :for nidx = (choose ($data ps))
          :do (progn
                (setf ph ht)
                (push nidx indices)
                ($zero! x)
                (setf ($ x 0 nidx) 1)))
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
                :do (let ((ph ($constant (zeros 1 *hidden-size*)))
                          (losses nil)
                          (tloss 0))
                      (loop :for i :from 0 :below ($size input 0)
                            :for xt = ($constant ($index input 0 i))
                            :for ht = ($tanh ($+ ($@ xt *wx*) ($@ ph *wh*) *bh*))
                            :for yt = ($+ ($@ ht *wy*) *by*)
                            :for ps = ($softmax yt)
                            :for y = ($constant ($index target 0 i))
                            :for l = ($cee ps y)
                            :do (progn
                                  (setf ph ht)
                                  (incf tloss ($data l))
                                  (push l losses)))
                      ($adgd! *rnn*)
                      (when (zerop (rem n 100))
                        (prn "")
                        (prn "[ITER]" n (/ tloss (* 1.0 *sequence-length*)))
                        (prn (sample ($data ph) ($ *char-to-idx* ($ input-str 0)) 72))
                        (prn "")
                        (gcf))
                      (incf n))))

(prn (sample (zeros 1 *hidden-size*) (random *vocab-size*) 200 0.5))

;;
;; 1-layer lstm
;;

(defparameter *hidden-size* 200)
(defparameter *sequence-length* 50)

(defparameter *lstm1* (parameters))

(defparameter *wa* ($parameter *lstm1* ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *ua* ($parameter *lstm1* ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *ba* ($parameter *lstm1* (zeros 1 *hidden-size*)))

(defparameter *wi* ($parameter *lstm1* ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *ui* ($parameter *lstm1* ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *bi* ($parameter *lstm1* (zeros 1 *hidden-size*)))

(defparameter *wf* ($parameter *lstm1* ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *uf* ($parameter *lstm1* ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *bf* ($parameter *lstm1* (ones 1 *hidden-size*)))

(defparameter *wo* ($parameter *lstm1* ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *uo* ($parameter *lstm1* ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *bo* ($parameter *lstm1* (zeros 1 *hidden-size*)))

(defparameter *wy* ($parameter *lstm1* ($* 0.01 (rndn *hidden-size* *vocab-size*))))
(defparameter *by* ($parameter *lstm1* (zeros 1 *vocab-size*)))

(defun sample (h c seed-idx n &optional (temperature 1))
  (let ((x (zeros 1 *vocab-size*))
        (indices (list seed-idx))
        (ph ($constant h))
        (pc ($constant c)))
    (setf ($ x 0 seed-idx) 1)
    (loop :for i :from 0 :below n
          :for xt = ($constant x)
          :for ft = ($sigmoid ($+ ($@ xt *wf*) ($@ ph *uf*) *bf*))
          :for it = ($sigmoid ($+ ($@ xt *wi*) ($@ ph *ui*) *bi*))
          :for ot = ($sigmoid ($+ ($@ xt *wo*) ($@ ph *uo*) *bo*))
          :for at = ($tanh ($+ ($@ xt *wa*) ($@ ph *ua*) *ba*))
          :for ct = ($+ ($* at it) ($* ft pc))
          :for ht = ($* ($tanh ct) ot)
          :for yt = ($+ ($@ ht *wy*) *by*)
          :for ps = ($softmax ($/ yt ($constant temperature)))
          :for nidx = (choose ($data ps))
          :do (progn
                (setf ph ht)
                (setf pc ct)
                (push nidx indices)
                ($zero! x)
                (setf ($ x 0 nidx) 1)))
    (coerce (mapcar (lambda (i) ($ *idx-to-char* i)) (reverse indices)) 'string)))

($cg! *lstm1*)

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
                :do (let ((ph ($constant(zeros 1 *hidden-size*)))
                          (pc ($constant (zeros 1 *hidden-size*)))
                          (losses nil)
                          (tloss 0))
                      (loop :for i :from 0 :below ($size input 0)
                            :for xt = ($constant ($index input 0 i))
                            :for it = ($sigmoid ($+ ($@ xt *wi*) ($@ ph *ui*) *bi*))
                            :for ft = ($sigmoid ($+ ($@ xt *wf*) ($@ ph *uf*) *bf*))
                            :for ot = ($sigmoid ($+ ($@ xt *wo*) ($@ ph *uo*) *bo*))
                            :for at = ($tanh ($+ ($@ xt *wa*) ($@ ph *ua*) *ba*))
                            :for ct = ($+ ($* at it) ($* ft pc))
                            :for ht = ($* ($tanh ct) ot)
                            :for yt = ($+ ($@ ht *wy*) *by*)
                            :for ps = ($softmax yt)
                            :for y = ($constant ($index target 0 i))
                            :for l = ($cee ps y)
                            :do (progn
                                  (setf ph ht)
                                  (setf pc ct)
                                  (incf tloss ($data l))
                                  (push l losses)))
                      ($adgd! *lstm1*)
                      (when (zerop (rem n 50))
                        (prn "")
                        (prn "[ITER]" n (/ tloss (* 1.0 *sequence-length*)))
                        (prn (sample ($data ph) ($data pc)
                                     ($ *char-to-idx* ($ input-str 0))
                                     72))
                        (prn "")
                        (gcf))
                      (incf n))))

(prn (sample (zeros 1 *hidden-size*) (zeros 1 *hidden-size*)
             (random *vocab-size*) 800))

;;
;; 2-layer lstm
;;

(defparameter *hidden-size* 512)
(defparameter *sequence-length* 128)

(defparameter *wa1* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *ua1* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *ba1* ($variable (zeros 1 *hidden-size*)))

(defparameter *wi1* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *ui1* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *bi1* ($variable (zeros 1 *hidden-size*)))

(defparameter *wf1* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *uf1* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *bf1* ($variable (ones 1 *hidden-size*)))

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
(defparameter *bf2* ($variable (ones 1 *vocab-size*)))

(defparameter *wo2* ($variable ($* 0.01 (rndn *hidden-size* *vocab-size*))))
(defparameter *uo2* ($variable ($* 0.01 (rndn *vocab-size* *vocab-size*))))
(defparameter *bo2* ($variable (zeros 1 *vocab-size*)))

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
          :for ps = ($softmax ($/ out2 ($constant temperature)))
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
                          (hs2 ($constant (zeros 1 *vocab-size*)))
                          (os2 ($constant (zeros 1 *vocab-size*)))
                          (losses nil)
                          (tloss 0))
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
                            :for yt = ($softmax out2)
                            :for y = ($constant ($index target 0 i))
                            :for l = ($cee yt y)
                            :do (progn
                                  (setf hs1 st1)
                                  (setf os1 out1)
                                  (setf hs2 st2)
                                  (setf os2 out2)
                                  (incf tloss ($data l))
                                  (push l losses)))
                      ($adgd! (list *wa1* *ua1* *ba1* *wi1* *ui1* *bi1* *wf1* *uf1* *bf1*
                                    *wo1* *uo1* *bo1*
                                    *wa2* *ua2* *ba2* *wi2* *ui2* *bi2* *wf2* *uf2* *bf2*
                                    *wo2* *uo2* *bo2*))
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
             (zeros 1 *vocab-size*) (zeros 1 *vocab-size*)
             (random *vocab-size*) 800))
