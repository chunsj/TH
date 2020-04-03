(defpackage :gench-lstm
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :gench-lstm)

;;
;; 1-layer lstm
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

(defparameter *wa* ($push *lstm* ($- ($* 0.16 (rnd *vocab-size* *hidden-size*)) 0.08)))
(defparameter *ua* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *ba* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size*)) 0.08)))

(defparameter *wi* ($push *lstm* ($- ($* 0.16 (rnd *vocab-size* *hidden-size*)) 0.08)))
(defparameter *ui* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bi* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size*)) 0.08)))

(defparameter *wf* ($push *lstm* ($- ($* 0.16 (rnd *vocab-size* *hidden-size*)) 0.08)))
(defparameter *uf* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bf* ($push *lstm* (ones *hidden-size*)))

(defparameter *wo* ($push *lstm* ($- ($* 0.16 (rnd *vocab-size* *hidden-size*)) 0.08)))
(defparameter *uo* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bo* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size*)) 0.08)))

(defparameter *wy* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size* *vocab-size*)) 0.08)))
(defparameter *by* ($push *lstm* ($- ($* 0.16 (rnd *vocab-size*)) 0.08)))

(defun sample (ph pc seed-idx n &optional (temperature 1))
  (let ((x (zeros 1 *vocab-size*))
        (indices (list seed-idx)))
    (setf ($ x 0 seed-idx) 1)
    (loop :for i :from 0 :below n
          :for it = ($sigmoid ($affine2 x *wi* ph *ui* *bi*))
          :for ft = ($sigmoid ($affine2 x *wf* ph *uf* *bf*))
          :for ot = ($sigmoid ($affine2 x *wo* ph *uo* *bo*))
          :for at = ($tanh ($affine2 x *wa* ph *ua* *ba*))
          :for ct = ($+ ($* at it) ($* ft pc))
          :for ht = ($* ($tanh ct) ot)
          :for yt = ($affine ht *wy* *by*)
          :for ps = ($softmax ($/ yt temperature))
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
  ($sigmoid ($affine2 xt w ph u b)))

(defun tanh-gate (xt ph w u b)
  ($tanh ($affine2 xt w ph u b)))

(defun affine (x w b) ($affine x w b))

($cg! *lstm*)

;; for test
(setf *max-epochs* 1)

(time
 (loop :for epoch :from 1 :to *max-epochs*
       :do (progn
             (loop :for bidx :from 0
                   :for input :in *input-batches*
                   :for target :in *target-batches*
                   :do (let ((ph (zeros ($size input 0) *hidden-size*))
                             (pc (zeros ($size input 0) *hidden-size*))
                             (loss 0))
                         (loop :for time :from 0 :below ($size input 1)
                               :for xt = (let ((m (zeros *batch-size* *vocab-size*)))
                                           (loop :for i :from 0 :below *batch-size*
                                                 :do (setf ($ m i ($ input i time)) 1))
                                           m)
                               :for it = (sigmoid-gate xt ph *wi* *ui* *bi*)
                               :for ft = (sigmoid-gate xt ph *wf* *uf* *bf*)
                               :for ot = (sigmoid-gate xt ph *wo* *uo* *bo*)
                               :for at = (tanh-gate xt ph *wa* *ua* *ba*)
                               :for ct = ($+ ($* at it) ($* ft pc))
                               :for ht = ($* ($tanh ct) ot)
                               :for yt = ($logsoftmax (affine ht *wy* *by*))
                               :for y = (let ((m ($index target 1 time)))
                                          ($reshape m *batch-size*))
                               :for l = ($cnll yt y)
                               :do (progn
                                     (setf ph ht)
                                     (setf pc ct)
                                     (incf loss ($data l))))
                         ($adgd! *lstm*)
                         (when (zerop (rem bidx 10))
                           (prn "")
                           (prn "[BTCH/ITER]" bidx "/" epoch (* loss (/ 1.0 *sequence-length*)))
                           (prn (sample ($index ($data ph) 0 0) ($index ($data pc) 0 0)
                                        (random *vocab-size*) 72))
                           (prn "")))))))

(gcf)

(prn (sample (zeros 1 *hidden-size*) (zeros 1 *hidden-size*)
             (random *vocab-size*) 800 1.0))
