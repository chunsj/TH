;; from
;; http://karpathy.github.io/2015/05/21/rnn-effectiveness/

(defpackage :gench-do-lstm3-obama
  (:use #:common-lisp
        #:mu
        #:th
        #:th.ex.data))

(in-package :gench-do-lstm3-obama)

;;
;; 3-layer lstm with dropout
;;

(defparameter *data-lines* (text-lines :obama))
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

(defparameter *hidden-size* 256)
(defparameter *sequence-length* 100)

(defparameter *dropout* 0.5)

(defparameter *batch-size* 50)

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

(defparameter *lstm3* (parameters))

(defparameter *wa1* ($push *lstm3* ($- ($* 0.16 (rnd *vocab-size* *hidden-size*)) 0.08)))
(defparameter *ua1* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *ba1* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size*)) 0.08)))

(defparameter *wi1* ($push *lstm3* ($- ($* 0.16 (rnd *vocab-size* *hidden-size*)) 0.08)))
(defparameter *ui1* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bi1* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size*)) 0.08)))

(defparameter *wf1* ($push *lstm3* ($- ($* 0.16 (rnd *vocab-size* *hidden-size*)) 0.08)))
(defparameter *uf1* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bf1* ($push *lstm3* (ones *hidden-size*)))

(defparameter *wo1* ($push *lstm3* ($- ($* 0.16 (rnd *vocab-size* *hidden-size*)) 0.08)))
(defparameter *uo1* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bo1* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size*)) 0.08)))

(defparameter *wa2* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *ua2* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *ba2* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size*)) 0.08)))

(defparameter *wi2* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *ui2* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bi2* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size*)) 0.08)))

(defparameter *wf2* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *uf2* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bf2* ($push *lstm3* (ones *hidden-size*)))

(defparameter *wo2* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *uo2* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bo2* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size*)) 0.08)))

(defparameter *wa3* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *ua3* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *ba3* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size*)) 0.08)))

(defparameter *wi3* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *ui3* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bi3* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size*)) 0.08)))

(defparameter *wf3* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *uf3* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bf3* ($push *lstm3* (ones *hidden-size*)))

(defparameter *wo3* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *uo3* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bo3* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size*)) 0.08)))

(defparameter *wy* ($push *lstm3* ($- ($* 0.16 (rnd *hidden-size* *vocab-size*)) 0.08)))
(defparameter *by* ($push *lstm3* ($- ($* 0.16 (rnd *vocab-size*)) 0.08)))

(defun lstm3-write-weight-to (w fname)
  (let ((f (file.disk fname "w")))
    ($fwrite ($data w) f)
    ($fclose f)))

(defun lstm3-write-weights ()
  (lstm3-write-weight-to *wa1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wa1.dat")
  (lstm3-write-weight-to *ua1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-ua1.dat")
  (lstm3-write-weight-to *ba1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-ba1.dat")
  (lstm3-write-weight-to *wi1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wi1.dat")
  (lstm3-write-weight-to *ui1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-ui1.dat")
  (lstm3-write-weight-to *bi1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-bi1.dat")
  (lstm3-write-weight-to *wf1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wf1.dat")
  (lstm3-write-weight-to *uf1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-uf1.dat")
  (lstm3-write-weight-to *bf1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-bf1.dat")
  (lstm3-write-weight-to *wo1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wo1.dat")
  (lstm3-write-weight-to *uo1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-uo1.dat")
  (lstm3-write-weight-to *bo1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-bo1.dat")

  (lstm3-write-weight-to *wa2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wa2.dat")
  (lstm3-write-weight-to *ua2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-ua2.dat")
  (lstm3-write-weight-to *ba2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-ba2.dat")
  (lstm3-write-weight-to *wi2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wi2.dat")
  (lstm3-write-weight-to *ui2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-ui2.dat")
  (lstm3-write-weight-to *bi2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-bi2.dat")
  (lstm3-write-weight-to *wf2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wf2.dat")
  (lstm3-write-weight-to *uf2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-uf2.dat")
  (lstm3-write-weight-to *bf2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-bf2.dat")
  (lstm3-write-weight-to *wo2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wo2.dat")
  (lstm3-write-weight-to *uo2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-uo2.dat")
  (lstm3-write-weight-to *bo2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-bo2.dat")

  (lstm3-write-weight-to *wa3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wa3.dat")
  (lstm3-write-weight-to *ua3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-ua3.dat")
  (lstm3-write-weight-to *ba3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-ba3.dat")
  (lstm3-write-weight-to *wi3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wi3.dat")
  (lstm3-write-weight-to *ui3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-ui3.dat")
  (lstm3-write-weight-to *bi3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-bi3.dat")
  (lstm3-write-weight-to *wf3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wf3.dat")
  (lstm3-write-weight-to *uf3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-uf3.dat")
  (lstm3-write-weight-to *bf3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-bf3.dat")
  (lstm3-write-weight-to *wo3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wo3.dat")
  (lstm3-write-weight-to *uo3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-uo3.dat")
  (lstm3-write-weight-to *bo3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-bo3.dat")

  (lstm3-write-weight-to *wy* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wy.dat")
  (lstm3-write-weight-to *by* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-by.dat"))

(defun lstm3-read-weight-from (w fname)
  (let ((f (file.disk fname "r")))
    ($fread ($data w) f)
    ($fclose f)))

(defun lstm3-read-weights ()
  (lstm3-read-weight-from *wa1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wa1.dat")
  (lstm3-read-weight-from *ua1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-ua1.dat")
  (lstm3-read-weight-from *ba1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-ba1.dat")
  (lstm3-read-weight-from *wi1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wi1.dat")
  (lstm3-read-weight-from *ui1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-ui1.dat")
  (lstm3-read-weight-from *bi1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-bi1.dat")
  (lstm3-read-weight-from *wf1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wf1.dat")
  (lstm3-read-weight-from *uf1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-uf1.dat")
  (lstm3-read-weight-from *bf1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-bf1.dat")
  (lstm3-read-weight-from *wo1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wo1.dat")
  (lstm3-read-weight-from *uo1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-uo1.dat")
  (lstm3-read-weight-from *bo1* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-bo1.dat")

  (lstm3-read-weight-from *wa2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wa2.dat")
  (lstm3-read-weight-from *ua2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-ua2.dat")
  (lstm3-read-weight-from *ba2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-ba2.dat")
  (lstm3-read-weight-from *wi2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wi2.dat")
  (lstm3-read-weight-from *ui2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-ui2.dat")
  (lstm3-read-weight-from *bi2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-bi2.dat")
  (lstm3-read-weight-from *wf2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wf2.dat")
  (lstm3-read-weight-from *uf2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-uf2.dat")
  (lstm3-read-weight-from *bf2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-bf2.dat")
  (lstm3-read-weight-from *wo2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wo2.dat")
  (lstm3-read-weight-from *uo2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-uo2.dat")
  (lstm3-read-weight-from *bo2* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-bo2.dat")

  (lstm3-read-weight-from *wa3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wa3.dat")
  (lstm3-read-weight-from *ua3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-ua3.dat")
  (lstm3-read-weight-from *ba3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-ba3.dat")
  (lstm3-read-weight-from *wi3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wi3.dat")
  (lstm3-read-weight-from *ui3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-ui3.dat")
  (lstm3-read-weight-from *bi3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-bi3.dat")
  (lstm3-read-weight-from *wf3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wf3.dat")
  (lstm3-read-weight-from *uf3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-uf3.dat")
  (lstm3-read-weight-from *bf3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-bf3.dat")
  (lstm3-read-weight-from *wo3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wo3.dat")
  (lstm3-read-weight-from *uo3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-uo3.dat")
  (lstm3-read-weight-from *bo3* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-bo3.dat")

  (lstm3-read-weight-from *wy* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-wy.dat")
  (lstm3-read-weight-from *by* "examples/weights/gench-do-lstm3-obama/gench-do-lstm3-obama-by.dat"))

(defun sample (ph1 pc1 ph2 pc2 ph3 pc3 seed-idx n &optional (temperature 1))
  (let ((x (zeros 1 *vocab-size*))
        (indices (list seed-idx)))
    (setf ($ x 0 seed-idx) 1)
    (loop :for i :from 0 :below n
          :for xt = x
          :for it1 = ($sigmoid ($affine2 xt *wi1* ph1 *ui1* *bi1*))
          :for ft1 = ($sigmoid ($affine2 xt *wf1* ph1 *uf1* *bf1*))
          :for ot1 = ($sigmoid ($affine2 xt *wo1* ph1 *uo1* *bo1*))
          :for at1 = ($tanh ($affine2 xt *wa1* ph1 *ua1* *ba1*))
          :for ct1 = ($addm2 at1 it1 ft1 pc1)
          :for ht1 = ($mul ($tanh ct1) ot1)
          :for it2 = ($sigmoid ($affine2 ht1 *wi2* ph2 *ui2* *bi2*))
          :for ft2 = ($sigmoid ($affine2 ht1 *wf2* ph2 *uf2* *bf2*))
          :for ot2 = ($sigmoid ($affine2 ht1 *wo2* ph2 *uo2* *bo2*))
          :for at2 = ($tanh ($affine2 ht1 *wa2* ph2 *ua2* *ba2*))
          :for ct2 = ($addm2 at2 it2 ft2 pc2)
          :for ht2 = ($mul ($tanh ct2) ot2)
          :for it3 = ($sigmoid ($affine2 ht2 *wi3* ph3 *ui3* *bi3*))
          :for ft3 = ($sigmoid ($affine2 ht2 *wf3* ph3 *uf3* *bf3*))
          :for ot3 = ($sigmoid ($affine2 ht2 *wo3* ph3 *uo3* *bo3*))
          :for at3 = ($tanh ($affine2 ht2 *wa3* ph3 *ua3* *ba3*))
          :for ct3 = ($addm2 at3 it3 ft3 pc3)
          :for ht3 = ($mul ($tanh ct3) ot3)
          :for yt = ($affine ht3 *wy* *by*)
          :for ps = ($softmax ($/ yt temperature))
          :for nidx = (choose ($data ps))
          :do (progn
                (setf ph1 ht1)
                (setf pc1 ct1)
                (setf ph2 ht2)
                (setf pc2 ct2)
                (setf ph3 ht3)
                (setf pc3 ct3)
                (unless (typep nidx 'number)
                  (prn nidx))
                (push nidx indices)
                ($zero! x)
                (setf ($ x 0 nidx) 1)))
    ($cg! *lstm3*)
    (coerce (mapcar (lambda (i) ($ *idx-to-char* i)) (reverse indices)) 'string)))

(defun sigmoid-gate (xt ph w u b) ($sigmoid ($affine2 xt w ph u b)))
(defun tanh-gate (xt ph w u b) ($tanh ($affine2 xt w ph u b)))
(defun cell-state (at it ft pc) ($addm2 at it ft pc))

($cg! *lstm3*)

(setf *max-epochs* 10)

;;(setf *learning-rate* 0.00001) ;; for continued training

(time
 (loop :for epoch :from 1 :to *max-epochs*
       :do (progn
             (loop :for bidx :from 0
                   :for input :in *input-batches*
                   :for target :in *target-batches*
                   :do (let ((ph1 (zeros ($size input 0) *hidden-size*))
                             (pc1 (zeros ($size input 0) *hidden-size*))
                             (ph2 (zeros ($size input 0) *hidden-size*))
                             (pc2 (zeros ($size input 0) *hidden-size*))
                             (ph3 (zeros ($size input 0) *hidden-size*))
                             (pc3 (zeros ($size input 0) *hidden-size*))
                             (loss 0))
                         (loop :for time :from 0 :below ($size input 1)
                               :for xt = (let ((m (zeros *batch-size* *vocab-size*)))
                                           (loop :for i :from 0 :below *batch-size*
                                                 :do (setf ($ m i ($ input i time)) 1))
                                           m)
                               :for it1 = (sigmoid-gate xt ph1 *wi1* *ui1* *bi1*)
                               :for ft1 = (sigmoid-gate xt ph1 *wf1* *uf1* *bf1*)
                               :for ot1 = (sigmoid-gate xt ph1 *wo1* *uo1* *bo1*)
                               :for at1 = (tanh-gate xt ph1 *wa1* *ua1* *ba1*)
                               :for ct1 = (cell-state at1 it1 ft1 pc1)
                               :for ht1 = ($mul ($tanh ct1) ot1)
                               :for ht1i = ($dropout ht1 T *dropout*)
                               :for it2 = (sigmoid-gate ht1i ph2 *wi2* *ui2* *bi2*)
                               :for ft2 = (sigmoid-gate ht1i ph2 *wf2* *uf2* *bf2*)
                               :for ot2 = (sigmoid-gate ht1i ph2 *wo2* *uo2* *bo2*)
                               :for at2 = (tanh-gate ht1i ph2 *wa2* *ua2* *ba2*)
                               :for ct2 = (cell-state at2 it2 ft2 pc2)
                               :for ht2 = ($mul ($tanh ct2) ot2)
                               :for ht2i = ($dropout ht2 T *dropout*)
                               :for it3 = (sigmoid-gate ht2i ph3 *wi3* *ui3* *bi3*)
                               :for ft3 = (sigmoid-gate ht2i ph3 *wf3* *uf3* *bf3*)
                               :for ot3 = (sigmoid-gate ht2i ph3 *wo3* *uo3* *bo3*)
                               :for at3 = (tanh-gate ht2i ph3 *wa3* *ua3* *ba3*)
                               :for ct3 = (cell-state at3 it3 ft3 pc3)
                               :for ht3 = ($mul ($tanh ct3) ot3)
                               :for ht3i = ($dropout ht3 T *dropout*)
                               :for yt = ($logsoftmax ($affine ht3i *wy* *by*))
                               :for y = (let ((m ($index target 1 time)))
                                          ($reshape m *batch-size*))
                               :for l = ($cnll yt y)
                               :do (progn
                                     (setf ph1 ht1)
                                     (setf pc1 ct1)
                                     (setf ph2 ht2)
                                     (setf pc2 ht2)
                                     (setf ph3 ht3)
                                     (setf pc3 ht3)
                                     (incf loss ($data l))))
                         ($amgd! *lstm3* *learning-rate*)
                         (when (zerop (rem bidx 10))
                           (prn "")
                           (prn "[BTCH/ITER]" bidx "/" epoch (* loss (/ 1.0 *sequence-length*)))
                           (prn (sample ($index ($data ph1) 0 0) ($index ($data pc1) 0 0)
                                        ($index ($data ph2) 0 0) ($index ($data pc2) 0 0)
                                        ($index ($data ph3) 0 0) ($index ($data pc3) 0 0)
                                        (random *vocab-size*) 72))
                           (prn "")))))))

(gcf)

(prn (sample (zeros 1 *hidden-size*) (zeros 1 *hidden-size*)
             (zeros 1 *hidden-size*) (zeros 1 *hidden-size*)
             (zeros 1 *hidden-size*) (zeros 1 *hidden-size*)
             (random *vocab-size*) 800 0.8))

(lstm3-write-weights)
(lstm3-read-weights)
