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
  (let ((choices (sort (loop :for i :from 0 :below ($size probs 1)
                             :collect (list i ($ probs 0 i)))
                       (lambda (a b) (> (cadr a) (cadr b))))))
    (labels ((make-ranges ()
               (loop :for (datum probability) :in choices
                     :sum (coerce probability 'double-float) :into total
                     :collect (list datum total)))
             (pick (ranges)
               (declare (optimize (speed 3) (safety 0) (debug 0)))
               (loop :with random = (random 1D0)
                     :for (datum below) :of-type (t double-float) :in ranges
                     :when (< random below)
                       :do (return datum))))
      (pick (make-ranges)))))

;;
;; vanilla rnn
;;

(defparameter *hidden-size* 100)
(defparameter *sequence-length* 25)

(defparameter *wx* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *wh* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *wy* ($variable ($* 0.01 (rndn *hidden-size* *vocab-size*))))
(defparameter *bh* ($variable (zeros 1 *hidden-size*)))
(defparameter *by* ($variable (zeros 1 *vocab-size*)))

(defun sample (h seed-idx n)
  (let ((x (zeros 1 *vocab-size*))
        (indices (list seed-idx))
        (hs ($state h)))
    (setf ($ x 0 seed-idx) 1)
    (loop :for i :from 0 :below n
          :for xt = ($constant x)
          :for ht = ($tanh ($+ ($@ xt *wx*) ($@ ($prev hs) *wh*) *bh*))
          :for yt = ($+ ($@ ht *wy*) *by*)
          :for ps = ($softmax yt)
          :for nidx = (choose ($data ps))
          :do (progn
                ($update! hs ht)
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
                :do (let ((hs ($state (zeros 1 *hidden-size*)))
                          (losses nil)
                          (tloss 0))
                      (loop :for i :from 0 :below ($size input 0)
                            :for xt = ($constant ($index input 0 i))
                            :for ht = ($tanh ($+ ($@ xt *wx*) ($@ ($prev hs) *wh*) *bh*))
                            :for yt = ($+ ($@ ht *wy*) *by*)
                            :for ps = ($softmax yt)
                            :for y = ($constant ($index target 0 i))
                            :for l = ($cee ps y)
                            :do (progn
                                  ($update! hs ht)
                                  (incf tloss ($data l))
                                  (push l losses)))
                      ($bptt! losses hs)
                      ($adgd! ($0 losses))
                      (when (zerop (rem n 100))
                        (prn "")
                        (prn "[ITER]" n tloss)
                        (prn (sample ($data ($prev hs)) ($ *char-to-idx* ($ input-str 0)) 72))
                        (prn "")
                        (gcf))
                      (incf n))))

(prn (sample (zeros 1 *hidden-size*) (random *vocab-size*) 800))

;;
;; 1-layer lstm
;;

(defparameter *hidden-size* 128)
(defparameter *sequence-length* 50)

(defparameter *wa* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *ua* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *ba* ($variable (ones 1 *hidden-size*)))

(defparameter *wi* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *ui* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *bi* ($variable (ones 1 *hidden-size*)))

(defparameter *wf* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *uf* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *bf* ($variable (ones 1 *hidden-size*)))

(defparameter *wo* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *uo* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *bo* ($variable (ones 1 *hidden-size*)))

(defparameter *wy* ($variable ($* 0.01 (rndn *hidden-size* *vocab-size*))))
(defparameter *by* ($variable (ones 1 *vocab-size*)))

(defun sample (h o seed-idx n)
  (let ((x (zeros 1 *vocab-size*))
        (indices (list seed-idx))
        (hs ($state h))
        (os ($state o)))
    (setf ($ x 0 seed-idx) 1)
    (loop :for i :from 0 :below n
          :for xt = ($constant x)
          :for at = ($tanh ($+ ($@ xt *wa*) ($@ ($prev os) *ua*) *ba*))
          :for it = ($sigmoid ($+ ($@ xt *wi*) ($@ ($prev os) *ui*) *bi*))
          :for ft = ($sigmoid ($+ ($@ xt *wf*) ($@ ($prev os) *uf*) *bf*))
          :for ot = ($sigmoid ($+ ($@ xt *wo*) ($@ ($prev os) *uo*) *bo*))
          :for st = ($+ ($* at it) ($* ft ($prev hs)))
          :for out = ($* ($tanh st) ot)
          :for yt = ($+ ($@ out *wy*) *by*)
          :for ps = ($softmax yt)
          :for nidx = (choose ($data ps))
          :do (progn
                ($update! hs st)
                ($update! os out)
                (push nidx indices)
                ($zero! x)
                (setf ($ x 0 nidx) 1)))
    (coerce (mapcar (lambda (i) ($ *idx-to-char* i)) (reverse indices)) 'string)))

(prn (sample (zeros 1 *hidden-size*) (zeros 1 *hidden-size*)
             (random *vocab-size*) 400))

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
                :do (let ((hs ($state (zeros 1 *hidden-size*)))
                          (os ($state (zeros 1 *hidden-size*)))
                          (losses nil)
                          (tloss 0))
                      (loop :for i :from 0 :below ($size input 0)
                            :for xt = ($constant ($index input 0 i))
                            :for at = ($tanh ($+ ($@ xt *wa*) ($@ ($prev os) *ua*) *ba*))
                            :for it = ($sigmoid ($+ ($@ xt *wi*) ($@ ($prev os) *ui*) *bi*))
                            :for ft = ($sigmoid ($+ ($@ xt *wf*) ($@ ($prev os) *uf*) *bf*))
                            :for ot = ($sigmoid ($+ ($@ xt *wo*) ($@ ($prev os) *uo*) *bo*))
                            :for st = ($+ ($* at it) ($* ft ($prev hs)))
                            :for out = ($* ($tanh st) ot)
                            :for yt = ($+ ($@ out *wy*) *by*)
                            :for ps = ($softmax yt)
                            :for y = ($constant ($index target 0 i))
                            :for l = ($cee ps y)
                            :do (progn
                                  ($update! hs st)
                                  ($update! os out)
                                  (incf tloss ($data l))
                                  (push l losses)))
                      ($bptt! losses os hs)
                      ($adgd! ($0 losses))
                      (when (zerop (rem n 100))
                        (prn "")
                        (prn "[ITER]" n tloss)
                        (prn (sample ($data ($prev hs))
                                     ($data ($prev os))
                                     ($ *char-to-idx* ($ input-str 0))
                                     72))
                        (prn "")
                        (gcf))
                      (incf n))))

;;
;; 2-layer lstm - current implementation does not work well (maybe there's a problem in bptt)
;;

(defparameter *hidden-size* 128)
(defparameter *sequence-length* 50)

(defparameter *wa1* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *ua1* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *ba1* ($variable (ones 1 *hidden-size*)))

(defparameter *wi1* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *ui1* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *bi1* ($variable (ones 1 *hidden-size*)))

(defparameter *wf1* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *uf1* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *bf1* ($variable (ones 1 *hidden-size*)))

(defparameter *wo1* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *uo1* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *bo1* ($variable (ones 1 *hidden-size*)))

(defparameter *wa2* ($variable ($* 0.01 (rndn *hidden-size* *vocab-size*))))
(defparameter *ua2* ($variable ($* 0.01 (rndn *vocab-size* *vocab-size*))))
(defparameter *ba2* ($variable (ones 1 *vocab-size*)))

(defparameter *wi2* ($variable ($* 0.01 (rndn *hidden-size* *vocab-size*))))
(defparameter *ui2* ($variable ($* 0.01 (rndn *vocab-size* *vocab-size*))))
(defparameter *bi2* ($variable (ones 1 *vocab-size*)))

(defparameter *wf2* ($variable ($* 0.01 (rndn *hidden-size* *vocab-size*))))
(defparameter *uf2* ($variable ($* 0.01 (rndn *vocab-size* *vocab-size*))))
(defparameter *bf2* ($variable (ones 1 *vocab-size*)))

(defparameter *wo2* ($variable ($* 0.01 (rndn *hidden-size* *vocab-size*))))
(defparameter *uo2* ($variable ($* 0.01 (rndn *vocab-size* *vocab-size*))))
(defparameter *bo2* ($variable (ones 1 *vocab-size*)))

(defun sample (h1 o1 h2 o2 seed-idx n)
  (let ((x (zeros 1 *vocab-size*))
        (indices (list seed-idx))
        (hs1 ($state h1))
        (os1 ($state o1))
        (hs2 ($state h2))
        (os2 ($state o2)))
    (setf ($ x 0 seed-idx) 1)
    (loop :for i :from 0 :below n
          :for xt = ($constant x)
          :for at1 = ($tanh ($+ ($@ xt *wa1*) ($@ ($prev os1) *ua1*) *ba1*))
          :for it1 = ($sigmoid ($+ ($@ xt *wi1*) ($@ ($prev os1) *ui1*) *bi1*))
          :for ft1 = ($sigmoid ($+ ($@ xt *wf1*) ($@ ($prev os1) *uf1*) *bf1*))
          :for ot1 = ($sigmoid ($+ ($@ xt *wo1*) ($@ ($prev os1) *uo1*) *bo1*))
          :for st1 = ($+ ($* at1 it1) ($* ft1 ($prev hs1)))
          :for out1 = ($* ($tanh st1) ot1)
          :for at2 = ($tanh ($+ ($@ out1 *wa2*) ($@ ($prev os2) *ua2*) *ba2*))
          :for it2 = ($sigmoid ($+ ($@ out1 *wi2*) ($@ ($prev os2) *ui2*) *bi2*))
          :for ft2 = ($sigmoid ($+ ($@ out1 *wf2*) ($@ ($prev os2) *uf2*) *bf2*))
          :for ot2 = ($sigmoid ($+ ($@ out1 *wo2*) ($@ ($prev os2) *uo2*) *bo2*))
          :for st2 = ($+ ($* at2 it2) ($* ft2 ($prev hs2)))
          :for out2 = ($* ($tanh st2) ot2)
          :for ps = ($softmax out2)
          :for nidx = (choose ($data ps))
          :do (progn
                ($update! hs1 st1)
                ($update! os1 out1)
                ($update! hs2 st2)
                ($update! os2 out2)
                (push nidx indices)
                ($zero! x)
                (setf ($ x 0 nidx) 1)))
    (coerce (mapcar (lambda (i) ($ *idx-to-char* i)) (reverse indices)) 'string)))

(prn (sample (zeros 1 *hidden-size*) (zeros 1 *hidden-size*)
             (zeros 1 *vocab-size*) (zeros 1 *vocab-size*)
             (random *vocab-size*) 800))

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
                :do (let ((hs1 ($state (zeros 1 *hidden-size*)))
                          (os1 ($state (zeros 1 *hidden-size*)))
                          (hs2 ($state (zeros 1 *vocab-size*)))
                          (os2 ($state (zeros 1 *vocab-size*)))
                          (losses nil)
                          (tloss 0))
                      (loop :for i :from 0 :below ($size input 0)
                            :for xt = ($constant ($index input 0 i))
                            :for at1 = ($tanh ($+ ($@ xt *wa1*) ($@ ($prev os1) *ua1*) *ba1*))
                            :for it1 = ($sigmoid ($+ ($@ xt *wi1*) ($@ ($prev os1) *ui1*) *bi1*))
                            :for ft1 = ($sigmoid ($+ ($@ xt *wf1*) ($@ ($prev os1) *uf1*) *bf1*))
                            :for ot1 = ($sigmoid ($+ ($@ xt *wo1*) ($@ ($prev os1) *uo1*) *bo1*))
                            :for st1 = ($+ ($* at1 it1) ($* ft1 ($prev hs1)))
                            :for out1 = ($* ($tanh st1) ot1)
                            :for at2 = ($tanh ($+ ($@ out1 *wa2*) ($@ ($prev os2) *ua2*) *ba2*))
                            :for it2 = ($sigmoid ($+ ($@ out1 *wi2*) ($@ ($prev os2) *ui2*) *bi2*))
                            :for ft2 = ($sigmoid ($+ ($@ out1 *wf2*) ($@ ($prev os2) *uf2*) *bf2*))
                            :for ot2 = ($sigmoid ($+ ($@ out1 *wo2*) ($@ ($prev os2) *uo2*) *bo2*))
                            :for st2 = ($+ ($* at2 it2) ($* ft2 ($prev hs2)))
                            :for out2 = ($* ($tanh st2) ot2)
                            :for yt = ($softmax out2)
                            :for y = ($constant ($index target 0 i))
                            :for l = ($cee yt y)
                            :do (progn
                                  ($update! hs1 st1)
                                  ($update! os1 out1)
                                  ($update! hs2 st2)
                                  ($update! os2 out2)
                                  (incf tloss ($data l))
                                  (push l losses)))
                      ($bptt! losses os2 hs2 os1 hs1)
                      ($adgd! ($0 losses))
                      (when (zerop (rem n 10))
                        (prn "")
                        (prn "[ITER]" n tloss)
                        (prn (sample ($data ($prev hs1))
                                     ($data ($prev os1))
                                     ($data ($prev hs2))
                                     ($data ($prev os2))
                                     ($ *char-to-idx* ($ input-str 0))
                                     72))
                        (prn "")
                        (gcf))
                      (incf n))))
