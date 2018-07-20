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

;;
;; simple rnn
;;
(defparameter *wx* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *wh* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *wy* ($variable ($* 0.01 (rndn *hidden-size* *vocab-size*))))
(defparameter *bh* ($variable (zeros 1 *hidden-size*)))
(defparameter *by* ($variable (zeros 1 *vocab-size*)))

(let ((n 0))
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
              ($bptt! losses)
              ($adgd! ($0 losses))
              (when (zerop (rem n 100))
                ;;(prn p tloss (sample ph (round ($ input 0 0)) 72))
                (gcf))
              (incf n))))

;;
;; simple rnn - revised
;;
(defparameter *wx* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *wh* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *wy* ($variable ($* 0.01 (rndn *hidden-size* *vocab-size*))))
(defparameter *bh* ($variable (zeros 1 *hidden-size*)))
(defparameter *by* ($variable (zeros 1 *vocab-size*)))

(let ((n 0))
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
        :do (let ((ph ($constant (zeros 1 *hidden-size*)))
                  (hs nil)
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
                          (setf ph ($state ht))
                          (push (list ht ph) hs)
                          (incf tloss ($data l))
                          (push l losses)))
              ($bptt! losses)
              ($bpst! hs)
              ($adgd! ($0 losses))
              (when (zerop (rem n 100))
                ;;(prn p tloss (sample ph (round ($ input 0 0)) 72))
                (gcf))
              (incf n))))


;;
;; gru trial
;;
(defparameter *wz* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *uz* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *bz* ($variable (zeros 1 *hidden-size*)))

(defparameter *wr* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *ur* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *br* ($variable (zeros 1 *hidden-size*)))

(defparameter *wh* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *uh* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *bh* ($variable (zeros 1 *hidden-size*)))

(defparameter *wy* ($variable ($* 0.01 (rndn *hidden-size* *vocab-size*))))
(defparameter *by* ($variable (zeros 1 *vocab-size*)))

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
      :do (let ((ph ($constant (zeros 1 *hidden-size*)))
                (one ($constant (ones 1 *hidden-size*)))
                (losses nil))
            (loop :for i :from 0 :below *sequence-length*
                  :for xt = ($constant ($index input 0 i))
                  :for zt = ($sigmoid ($+ ($@ xt *wz*) ($@ ph *uz*) *bz*))
                  :for rt = ($sigmoid ($+ ($@ xt *wr*) ($@ ph *ur*) *br*))
                  :for ht = ($+ ($* ($- one zt) ph)
                                ($* zt ($sigmoid ($+ ($@ xt *wh*)
                                                     ($@ ($* rt ph) *uh*)
                                                     *bh*))))
                  :for p = ($+ ($@ ht *wy*) *by*)
                  :for yt = ($softmax p)
                  :for y = ($constant ($index target 0 i))
                  :for l = ($cee yt y)
                  :do (progn
                        ;;(setf ph ht)
                        (setf ($data ph) ($data ht)) ;; no, this is wrong, for speed
                        (push l losses)))
            ($bptt! losses)
            ($adgd! ($0 losses))
            (gcf)))

;;
;; even simple 1 layer lstm takes too much time.
;;
(defparameter *wa* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *ua* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *ba* ($variable (zeros 1 *hidden-size*)))

(defparameter *wi* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *ui* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *bi* ($variable (zeros 1 *hidden-size*)))

(defparameter *wf* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *uf* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *bf* ($variable (zeros 1 *hidden-size*)))

(defparameter *wo* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *uo* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *bo* ($variable (zeros 1 *hidden-size*)))

(defparameter *wy* ($variable ($* 0.01 (rndn *hidden-size* *vocab-size*))))
(defparameter *by* ($variable (zeros 1 *vocab-size*)))

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
      :do (let ((pout ($constant (zeros 1 *hidden-size*)))
                (ps ($constant (zeros 1 *hidden-size*)))
                (losses nil))
            (loop :for i :from 0 :below (min 5 *sequence-length*)
                  :for xt = ($constant ($index input 0 i))
                  :for at = ($tanh ($+ ($@ xt *wa*) ($@ pout *ua*) *ba*))
                  :for it = ($sigmoid ($+ ($@ xt *wi*) ($@ pout *ui*) *bi*))
                  :for ft = ($sigmoid ($+ ($@ xt *wf*) ($@ pout *uf*) *bf*))
                  :for ot = ($sigmoid ($+ ($@ xt *wo*) ($@ pout *uo*) *bo*))
                  :for st = ($+ ($* at it) ($* ft ps))
                  :for out = ($* ($tanh st) ot)
                  :for p = ($+ ($@ out *wy*) *by*)
                  :for yt = ($softmax p)
                  :for y = ($constant ($index target 0 i))
                  :for l = ($cee yt y)
                  :do (progn
                        (setf ps st)
                        (setf pout out)
                        (push l losses)))
            ($bptt! losses)
            ($adgd! ($0 losses))
            (gcf)))

;;
;; even simple 1 layer lstm - revised
;;
(defparameter *wa* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *ua* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *ba* ($variable (zeros 1 *hidden-size*)))

(defparameter *wi* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *ui* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *bi* ($variable (zeros 1 *hidden-size*)))

(defparameter *wf* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *uf* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *bf* ($variable (zeros 1 *hidden-size*)))

(defparameter *wo* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *uo* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *bo* ($variable (zeros 1 *hidden-size*)))

(defparameter *wy* ($variable ($* 0.01 (rndn *hidden-size* *vocab-size*))))
(defparameter *by* ($variable (zeros 1 *vocab-size*)))

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
      :do (let ((pout ($constant (zeros 1 *hidden-size*)))
                (ps ($constant (zeros 1 *hidden-size*)))
                (outs nil)
                (states nil)
                (losses nil))
            (loop :for i :from 0 :below *sequence-length*
                  :for xt = ($constant ($index input 0 i))
                  :for at = ($tanh ($+ ($@ xt *wa*) ($@ pout *ua*) *ba*))
                  :for it = ($sigmoid ($+ ($@ xt *wi*) ($@ pout *ui*) *bi*))
                  :for ft = ($sigmoid ($+ ($@ xt *wf*) ($@ pout *uf*) *bf*))
                  :for ot = ($sigmoid ($+ ($@ xt *wo*) ($@ pout *uo*) *bo*))
                  :for st = ($+ ($* at it) ($* ft ps))
                  :for out = ($* ($tanh st) ot)
                  :for p = ($+ ($@ out *wy*) *by*)
                  :for yt = ($softmax p)
                  :for y = ($constant ($index target 0 i))
                  :for l = ($cee yt y)
                  :do (progn
                        (setf ps ($state st))
                        (setf pout ($state out))
                        (push (list st ps) states)
                        (push (list out pout) outs)
                        (push l losses)))
            ($bptt! losses)
            ($bpst! states outs)
            ($adgd! ($0 losses))
            (gcf)))

;;
;; 2 layer - very slow even for single sample iteration
;;
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
      :do (let ((pout1 ($constant (zeros 1 *hidden-size*)))
                (ps1 ($constant (zeros 1 *hidden-size*)))
                (pout2 ($constant (zeros 1 *vocab-size*)))
                (ps2 ($constant (zeros 1 *vocab-size*)))
                (losses nil))
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
                  :for y = ($constant ($index target 0 i))
                  :for l = ($cee yt y)
                  :do (progn
                        (setf ps1 st1)
                        (setf ps2 st2)
                        (setf pout1 out1)
                        (setf pout2 out2)
                        (push l losses)))
            ($bptt! losses)
            ($adgd! ($0 losses))
            (gcf)))

;;
;; 2 layer lstm - revised
;;
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
      :do (let ((pout1 ($constant (zeros 1 *hidden-size*)))
                (ps1 ($constant (zeros 1 *hidden-size*)))
                (pout2 ($constant (zeros 1 *vocab-size*)))
                (ps2 ($constant (zeros 1 *vocab-size*)))
                (outs1 nil)
                (states1 nil)
                (outs2 nil)
                (states2 nil)
                (losses nil))
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
                  :for y = ($constant ($index target 0 i))
                  :for l = ($cee yt y)
                  :do (progn
                        (setf ps1 ($state st1))
                        (setf ps2 ($state st2))
                        (setf pout1 ($state out1))
                        (setf pout2 ($state out2))
                        (push (list st1 ps1) states1)
                        (push (list out1 pout1) outs1)
                        (push (list out2 pout2) outs2)
                        (push l losses)))
            ($bptt! losses)
            ($bpst! states2 outs2 states1 outs1)
            ($adgd! ($0 losses))
            (gcf)))
