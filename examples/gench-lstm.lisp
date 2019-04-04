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

(time
 (with-foreign-memory-hack
   (loop :for epoch :from 1 :to *max-epochs*
         :do (progn
               (loop :for bidx :from 0
                     :for input :in *input-batches*
                     :for target :in *target-batches*
                     :do (let ((ph ($constant (zeros ($size input 0) *hidden-size*)))
                               (pc ($constant (zeros ($size input 0) *hidden-size*)))
                               (loss 0))
                           (loop :for time :from 0 :below ($size input 1)
                                 :for xt = (let ((m (zeros *batch-size* *vocab-size*)))
                                             (loop :for i :from 0 :below *batch-size*
                                                   :do (setf ($ m i ($ input i time)) 1))
                                             ($constant m))
                                 :for it = (sigmoid-gate xt ph *wi* *ui* *bi*)
                                 :for ft = (sigmoid-gate xt ph *wf* *uf* *bf*)
                                 :for ot = (sigmoid-gate xt ph *wo* *uo* *bo*)
                                 :for at = (tanh-gate xt ph *wa* *ua* *ba*)
                                 :for ct = ($+ ($* at it) ($* ft pc))
                                 :for ht = ($* ($tanh ct) ot)
                                 :for yt = ($logsoftmax (affine ht *wy* *by*))
                                 :for y = (let ((m ($index target 1 time)))
                                            ($constant ($reshape m *batch-size*)))
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
                             (prn ""))))))))

(time
 (loop :for epoch :from 1 :to *max-epochs*
       :do (progn
             (loop :for bidx :from 0
                   :for input :in *input-batches*
                   :for target :in *target-batches*
                   :do (let ((ph ($constant (zeros ($size input 0) *hidden-size*)))
                             (pc ($constant (zeros ($size input 0) *hidden-size*)))
                             (loss 0))
                         (loop :for time :from 0 :below ($size input 1)
                               :for xt = (let ((m (zeros *batch-size* *vocab-size*)))
                                           (loop :for i :from 0 :below *batch-size*
                                                 :do (setf ($ m i ($ input i time)) 1))
                                           ($constant m))
                               :for it = (sigmoid-gate xt ph *wi* *ui* *bi*)
                               :for ft = (sigmoid-gate xt ph *wf* *uf* *bf*)
                               :for ot = (sigmoid-gate xt ph *wo* *uo* *bo*)
                               :for at = (tanh-gate xt ph *wa* *ua* *ba*)
                               :for ct = ($+ ($* at it) ($* ft pc))
                               :for ht = ($* ($tanh ct) ot)
                               :for yt = ($logsoftmax (affine ht *wy* *by*))
                               :for y = (let ((m ($index target 1 time)))
                                          ($constant ($reshape m *batch-size*)))
                               :for l = ($cnll yt y)
                               :do (progn
                                     (setf ph ht)
                                     (setf pc ct)
                                     (incf loss ($data l))))
                         ($adgd! *lstm*)
                         (when (and (> bidx 0) (rem bidx 5)) (gcf))
                         (when (zerop (rem bidx 10))
                           (prn "")
                           (prn "[BTCH/ITER]" bidx "/" epoch (* loss (/ 1.0 *sequence-length*)))
                           (prn (sample ($index ($data ph) 0 0) ($index ($data pc) 0 0)
                                        (random *vocab-size*) 72))
                           (prn "")))))))

(gcf)

(prn (sample (zeros 1 *hidden-size*) (zeros 1 *hidden-size*)
             (random *vocab-size*) 800 1.0))
