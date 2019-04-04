;; from
;; http://karpathy.github.io/2015/05/21/rnn-effectiveness/

(defpackage :gench-lstm2
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :gench-lstm2)

;;
;; 2-layer lstm
;;

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

(defparameter *hidden-size* 128)
(defparameter *sequence-length* 50)

(defparameter *batch-size* 100)

(defparameter *max-epochs* 50)

(defparameter *learning-rate* 0.002)

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

(defparameter *wy* ($parameter *lstm2* ($- ($* 0.16 (rnd *hidden-size* *vocab-size*)) 0.08)))
(defparameter *by* ($parameter *lstm2* ($- ($* 0.16 (rnd 1 *vocab-size*)) 0.08)))

(defun sample (ph1 pc1 ph2 pc2 seed-idx n &optional (temperature 1))
  (let ((x (zeros 1 *vocab-size*))
        (indices (list seed-idx))
        (ph1 ($constant ph1))
        (pc1 ($constant pc1))
        (ph2 ($constant ph2))
        (pc2 ($constant pc2)))
    (setf ($ x 0 seed-idx) 1)
    (loop :for i :from 0 :below n
          :for xt = ($constant x)
          :for it1 = ($sigmoid ($+ ($@ xt *wi1*) ($@ ph1 *ui1*) *bi1*))
          :for ft1 = ($sigmoid ($+ ($@ xt *wf1*) ($@ ph1 *uf1*) *bf1*))
          :for ot1 = ($sigmoid ($+ ($@ xt *wo1*) ($@ ph1 *uo1*) *bo1*))
          :for at1 = ($tanh ($+ ($@ xt *wa1*) ($@ ph1 *ua1*) *ba1*))
          :for ct1 = ($+ ($* at1 it1) ($* ft1 pc1))
          :for ht1 = ($* ($tanh ct1) ot1)
          :for it2 = ($sigmoid ($+ ($@ ht1 *wi2*) ($@ ph2 *ui2*) *bi2*))
          :for ft2 = ($sigmoid ($+ ($@ ht1 *wf2*) ($@ ph2 *uf2*) *bf2*))
          :for ot2 = ($sigmoid ($+ ($@ ht1 *wo2*) ($@ ph2 *uo2*) *bo2*))
          :for at2 = ($tanh ($+ ($@ ht1 *wa2*) ($@ ph2 *ua2*) *ba2*))
          :for ct2 = ($+ ($* at2 it2) ($* ft2 pc2))
          :for ht2 = ($* ($tanh ct2) ot2)
          :for yt = ($+ ($@ ht2 *wy*) *by*)
          :for ps = ($softmax ($/ yt ($constant temperature)))
          :for nidx = (choose ($data ps))
          :do (progn
                (setf ph1 ht1)
                (setf pc1 ct1)
                (setf ph2 ht2)
                (setf pc2 ct2)
                (unless (typep nidx 'number)
                  (prn nidx))
                (push nidx indices)
                ($zero! x)
                (setf ($ x 0 nidx) 1)))
    ($cg! *lstm2*)
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

($cg! *lstm2*)

(setf *max-epochs* 1)

(time
 (with-foreign-memory-limit
     (loop :for epoch :from 1 :to *max-epochs*
           :do (progn
                 (loop :for bidx :from 0
                       :for input :in *input-batches*
                       :for target :in *target-batches*
                       :do (let ((ph1 ($constant (zeros ($size input 0) *hidden-size*)))
                                 (pc1 ($constant (zeros ($size input 0) *hidden-size*)))
                                 (ph2 ($constant (zeros ($size input 0) *hidden-size*)))
                                 (pc2 ($constant (zeros ($size input 0) *hidden-size*)))
                                 (loss 0))
                             (loop :for time :from 0 :below ($size input 1)
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
                             ($amgd! *lstm2* *learning-rate*)
                             (when (zerop (rem bidx 10))
                               (prn "")
                               (prn "[BTCH/ITER]" bidx "/" epoch (* loss (/ 1.0 *sequence-length*)))
                               (prn (sample ($index ($data ph1) 0 0) ($index ($data pc1) 0 0)
                                            ($index ($data ph2) 0 0) ($index ($data pc2) 0 0)
                                            (random *vocab-size*) 72))
                               (prn ""))))))))

(time
 (loop :for epoch :from 1 :to *max-epochs*
       :do (progn
             (loop :for bidx :from 0
                   :for input :in *input-batches*
                   :for target :in *target-batches*
                   :do (let ((ph1 ($constant (zeros ($size input 0) *hidden-size*)))
                             (pc1 ($constant (zeros ($size input 0) *hidden-size*)))
                             (ph2 ($constant (zeros ($size input 0) *hidden-size*)))
                             (pc2 ($constant (zeros ($size input 0) *hidden-size*)))
                             (loss 0))
                         (loop :for time :from 0 :below ($size input 1)
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
                         ($amgd! *lstm2* *learning-rate*)
                         (when (and (> bidx 0) (zerop (rem bidx 5))) (gcf))
                         (when (zerop (rem bidx 10))
                           (prn "")
                           (prn "[BTCH/ITER]" bidx "/" epoch (* loss (/ 1.0 *sequence-length*)))
                           (prn (sample ($index ($data ph1) 0 0) ($index ($data pc1) 0 0)
                                        ($index ($data ph2) 0 0) ($index ($data pc2) 0 0)
                                        (random *vocab-size*) 72))
                           (prn "")))))))

(gcf)

(prn (sample (zeros 1 *hidden-size*) (zeros 1 *hidden-size*)
             (zeros 1 *hidden-size*) (zeros 1 *hidden-size*)
             (random *vocab-size*) 800 0.5))
