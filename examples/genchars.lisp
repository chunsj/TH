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
