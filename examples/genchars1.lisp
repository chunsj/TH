(defpackage :genchars1
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :genchars1)

;;
;; 1-layer lstm
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

(defparameter *batch-size* 100)

(defparameter *max-epochs* 50)

(defparameter *learning-rate* 2E-3)
(defparameter *learning-rate-decay* 0.97)
(defparameter *learning-rate-decay-after* 10)
(defparameter *decay-rate* 0.95)

(defparameter *input* (let* ((sz *data-size*)
                             (nbatch *batch-size*)
                             (nseq *sequence-length*)
                             (nlen (* nbatch nseq (floor (/ sz (* nbatch nseq)))))
                             (data (tensor.byte nlen)))
                        (prn "NLEN:" nlen)
                        (loop :for i :from 0 :below nlen
                              :do (setf ($ data i) ($ *char-to-idx* ($ *data* i))))
                        data))
(defparameter *target* (let ((ydata ($clone *input*)))
                         (setf ($subview ydata 0 (1- ($size ydata 0)))
                               ($subview *input* 1 (1- ($size *input* 0))))
                         (setf ($ ydata (1- ($size *input* 0))) ($ *input* 0))
                         ydata))
(defparameter *input-batches* ($split ($view *input* *batch-size*
                                             (/ ($size *input* 0) *batch-size*))
                                      *sequence-length* 1))
(defparameter *target-batches* ($split ($view *target* *batch-size*
                                              (/ ($size *target* 0) *batch-size*))
                                       *sequence-length* 1))
(defparameter *nbatches* ($count *input-batches*))

(defparameter *lstm* (parameters))

(defparameter *wa* ($parameter *lstm* ($- ($* 0.16 (rnd *vocab-size* *hidden-size*)) 0.08)))
(defparameter *ua* ($parameter *lstm* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *ba* ($parameter *lstm* ($- ($* 0.16 (rnd 1 *hidden-size*)) 0.08)))

(defparameter *wi* ($parameter *lstm* ($- ($* 0.16 (rnd *vocab-size* *hidden-size*)) 0.08)))
(defparameter *ui* ($parameter *lstm* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bi* ($parameter *lstm* ($- ($* 0.16 (rnd 1 *hidden-size*)) 0.08)))

(defparameter *wf* ($parameter *lstm* ($- ($* 0.16 (rnd *vocab-size* *hidden-size*)) 0.08)))
(defparameter *uf* ($parameter *lstm* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bf* ($parameter *lstm* (ones 1 *hidden-size*)))

(defparameter *wo* ($parameter *lstm* ($- ($* 0.16 (rnd *vocab-size* *hidden-size*)) 0.08)))
(defparameter *uo* ($parameter *lstm* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bo* ($parameter *lstm* ($- ($* 0.16 (rnd 1 *hidden-size*)) 0.08)))

(defparameter *wy* ($parameter *lstm* ($- ($* 0.16 (rnd *hidden-size* *vocab-size*)) 0.08)))
(defparameter *by* ($parameter *lstm* ($- ($* 0.16 (rnd 1 *vocab-size*)) 0.08)))

(defun sample (ph pc seed-idx n &optional (temperature 1))
  (let ((x (zeros 1 *vocab-size*))
        (indices (list seed-idx))
        (ph ($constant ph))
        (pc ($constant pc)))
    (setf ($ x 0 seed-idx) 1)
    (loop :for i :from 0 :below n
          :for xt = ($constant x)
          :for it = ($sigmoid ($+ ($@ xt *wi*) ($@ ph *ui*) *bi*))
          :for ft = ($sigmoid ($+ ($@ xt *wf*) ($@ ph *uf*) *bf*))
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
                (unless (typep nidx 'number)
                  (prn nidx))
                (push nidx indices)
                ($zero! x)
                (setf ($ x 0 nidx) 1)))
    ($cg! *lstm*)
    (coerce (mapcar (lambda (i) ($ *idx-to-char* i)) (reverse indices)) 'string)))

(defun sigmoid-gate (xt ph w u b)
  (let* ((tx ($@ xt w))
         (tp ($@ ph u))
         (tb ($@ ($constant (ones ($size xt 0) 1)) b))
         (res ($sigmoid ($+ tx tp tb))))
    res))

(defun tanh-gate (xt ph w u b)
  (let* ((tx ($@ xt w))
         (tp ($@ ph u))
         (tb ($@ ($constant (ones ($size xt 0) 1)) b))
         (res ($tanh ($+ tx tp tb))))
    res))

(defun affine (x w b)
  (let* ((tx ($@ x w))
         (tb ($@ ($constant (ones ($size x 0) 1)) b)))
    ($+ tx tb)))

($cg! *lstm*)

(loop :for epoch :from 1 :to *max-epochs*
      :do (progn
            (loop :for bidx :from 0
                  :for input :in (subseq *input-batches* 0 1)
                  :for target :in (subseq *target-batches* 0 1)
                  :do (let ((ph1 ($constant (zeros ($size input 0) *hidden-size*)))
                            (pc1 ($constant (zeros ($size input 0) *hidden-size*)))
                            (loss 0))
                        (loop :for time :from 0 :below (min 1 ($size input 1))
                              :for xt = (let ((m (zeros *batch-size* *vocab-size*)))
                                          (loop :for i :from 0 :below *batch-size*
                                                :do (setf ($ m i ($ input i time)) 1))
                                          ($constant m))
                              :for it1 = (sigmoid-gate xt ph1 *wi1* *ui1* *bi1*)
                              :for ft1 = (sigmoid-gate xt ph1 *wf1* *uf1* *bf1*)
                              :for ot1 = (sigmoid-gate xt ph1 *wo1* *uo1* *bo1*)
                              :for at1 = (tanh-gate xt ph1 *wa1* *ua1* *ba1*)
                              :for ct1 = ($+ ($* at1 it1) ($* ft1 pc1))
                              :for ht1 = ($* ($tanh ct1) ot1)
                              :for it2 = (sigmoid-gate ht1 ph2 *wi2* *ui2* *bi2*)
                              :for ft2 = (sigmoid-gate ht1 ph2 *wf2* *uf2* *bf2*)
                              :for ot2 = (sigmoid-gate ht1 ph2 *wo2* *uo2* *bo2*)
                              :for at2 = (tanh-gate ht1 ph2 *wa2* *ua2* *ba2*)
                              :for ct2 = ($+ ($* at2 it2) ($* ft2 pc2))
                              :for ht2 = ($* ($tanh ct2) ot2)
                              :for yt = ($logsoftmax (affine ht2 *wy* *by*))
                              :for y = (let ((m ($index target 1 time)))
                                         ($constant ($reshape m *batch-size*)))
                              :for l = ($cnll yt y)
                              :do (progn
                                    (setf ph1 ht1)
                                    (setf pc1 ct1)
                                    (setf ph2 ht2)
                                    (setf pc2 ht2)
                                    (incf loss ($data l))))
                        ($rmgd! *lstm2* *learning-rate* *decay-rate*)
                        (when (and (= bidx 0) (>= epoch *learning-rate-decay-after*))
                          (setf *learning-rate* (* *learning-rate* *learning-rate-decay*))
                          (prn "DECAYED LR:" *learning-rate*))
                        ;;($adgd! *lstm2*)
                        (when (zerop (rem bidx 50))
                          (prn "")
                          (prn "[BTCH/ITER]" bidx "/" epoch (* loss (/ 1.0 *sequence-length*)))
                          (prn (sample ($index ($data ph1) 0 0) ($index ($data pc1) 0 0)
                                       ($index ($data ph2) 0 0) ($index ($data pc2) 0 0)
                                       (random *vocab-size*) 72))
                          (prn ""))))
            (gcf)))

(prn (sample (zeros 1 *hidden-size*) (zeros 1 *hidden-size*)
             (random *vocab-size*) 100 1))
