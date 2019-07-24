;; from
;; http://karpathy.github.io/2015/05/21/rnn-effectiveness/

(defpackage :genchars-obama-lstm-l
  (:use #:common-lisp
        #:mu
        #:th
        #:th.ex.data))

(in-package :genchars-obama-lstm-l)

(th::th-set-num-threads 12)
(th::th-set-gc-hard-max (* 10 1024 1024 1024))

(defparameter *data-lines* (remove-if (lambda (line) (< ($count line) 1)) (text-lines :obama)))
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
;; non batched lstm for example
;;

(defparameter *hidden-size* 256)
(defparameter *sequence-length* 100)

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

(defun lstm-write-weight-to (w fname)
  (let ((f (file.disk fname "w")))
    ($fwrite ($data w) f)
    ($fclose f)))

(defun lstm-read-weight-from (w fname)
  (let ((f (file.disk fname "r")))
    ($fread ($data w) f)
    ($fclose f)))

(defun lstm-write-weights ()
  (lstm-write-weight-to *wa* "examples/weights/genchar-obama-lstm-l/lstm-wa.dat")
  (lstm-write-weight-to *ua* "examples/weights/genchar-obama-lstm-l/lstm-ua.dat")
  (lstm-write-weight-to *ba* "examples/weights/genchar-obama-lstm-l/lstm-ba.dat")
  (lstm-write-weight-to *wi* "examples/weights/genchar-obama-lstm-l/lstm-wi.dat")
  (lstm-write-weight-to *ui* "examples/weights/genchar-obama-lstm-l/lstm-ui.dat")
  (lstm-write-weight-to *bi* "examples/weights/genchar-obama-lstm-l/lstm-bi.dat")
  (lstm-write-weight-to *wf* "examples/weights/genchar-obama-lstm-l/lstm-wf.dat")
  (lstm-write-weight-to *uf* "examples/weights/genchar-obama-lstm-l/lstm-uf.dat")
  (lstm-write-weight-to *bf* "examples/weights/genchar-obama-lstm-l/lstm-bf.dat")
  (lstm-write-weight-to *wo* "examples/weights/genchar-obama-lstm-l/lstm-wo.dat")
  (lstm-write-weight-to *uo* "examples/weights/genchar-obama-lstm-l/lstm-uo.dat")
  (lstm-write-weight-to *bo* "examples/weights/genchar-obama-lstm-l/lstm-bo.dat")
  (lstm-write-weight-to *wy* "examples/weights/genchar-obama-lstm-l/lstm-wy.dat")
  (lstm-write-weight-to *by* "examples/weights/genchar-obama-lstm-l/lstm-by.dat"))

(defun lstm-read-weights ()
  (lstm-read-weight-from *wa* "examples/weights/genchar-obama-lstm-l/lstm-wa.dat")
  (lstm-read-weight-from *ua* "examples/weights/genchar-obama-lstm-l/lstm-ua.dat")
  (lstm-read-weight-from *ba* "examples/weights/genchar-obama-lstm-l/lstm-ba.dat")
  (lstm-read-weight-from *wi* "examples/weights/genchar-obama-lstm-l/lstm-wi.dat")
  (lstm-read-weight-from *ui* "examples/weights/genchar-obama-lstm-l/lstm-ui.dat")
  (lstm-read-weight-from *bi* "examples/weights/genchar-obama-lstm-l/lstm-bi.dat")
  (lstm-read-weight-from *wf* "examples/weights/genchar-obama-lstm-l/lstm-wf.dat")
  (lstm-read-weight-from *uf* "examples/weights/genchar-obama-lstm-l/lstm-uf.dat")
  (lstm-read-weight-from *bf* "examples/weights/genchar-obama-lstm-l/lstm-bf.dat")
  (lstm-read-weight-from *wo* "examples/weights/genchar-obama-lstm-l/lstm-wo.dat")
  (lstm-read-weight-from *uo* "examples/weights/genchar-obama-lstm-l/lstm-uo.dat")
  (lstm-read-weight-from *bo* "examples/weights/genchar-obama-lstm-l/lstm-bo.dat")
  (lstm-read-weight-from *wy* "examples/weights/genchar-obama-lstm-l/lstm-wy.dat")
  (lstm-read-weight-from *by* "examples/weights/genchar-obama-lstm-l/lstm-by.dat"))

(defun cindices (str)
  (let ((m (zeros ($count str) *vocab-size*)))
    (loop :for i :from 0 :below ($count str)
          :for ch = ($ str i)
          :do (setf ($ m i ($ *char-to-idx* ch)) 1))
    m))

(defun rstrings (indices) (coerce (mapcar (lambda (i) ($ *idx-to-char* i)) indices) 'string))

(defun seedh (str &optional (temperature 1))
  (let ((input (cindices str))
        (ph (zeros 1 *hidden-size*))
        (pc (zeros 1 *hidden-size*))
        (wa ($data *wa*))
        (ua ($data *ua*))
        (ba ($data *ba*))
        (wi ($data *wi*))
        (ui ($data *ui*))
        (bi ($data *bi*))
        (wf ($data *wf*))
        (uf ($data *uf*))
        (bf ($data *bf*))
        (wo ($data *wo*))
        (uo ($data *uo*))
        (bo ($data *bo*))
        (wy ($data *wy*))
        (by ($data *by*))
        (ncidx 0))
    (loop :for i :from 0 :below ($size input 0)
          :for xt = ($index input 0 i)
          :for (ht ct) = ($lstm xt ph pc wi ui wf uf wo uo wa ua bi bf bo ba)
          :for yt = ($affine ht wy by)
          :for ps = ($softmax ($/ yt temperature))
          :for nidx = (choose ps)
          :do (setf ph ht
                    pc ct
                    ncidx nidx))
    (list ncidx ph pc)))

(defun sample (str n &optional (temperature 1))
  (let ((x (zeros 1 *vocab-size*))
        (indices nil)
        (sh (when str (seedh str temperature)))
        (wa ($data *wa*))
        (ua ($data *ua*))
        (ba ($data *ba*))
        (wi ($data *wi*))
        (ui ($data *ui*))
        (bi ($data *bi*))
        (wf ($data *wf*))
        (uf ($data *uf*))
        (bf ($data *bf*))
        (wo ($data *wo*))
        (uo ($data *uo*))
        (bo ($data *bo*))
        (wy ($data *wy*))
        (by ($data *by*))
        (ph nil)
        (pc nil))
    (if sh
        (let ((idx0 ($0 sh))
              (h ($1 sh))
              (c ($2 sh)))
          (setf ($ x 0 idx0) 1)
          (setf ph h
                pc c)
          (push idx0 indices))
        (let ((idx0 (random *vocab-size*))
              (h (zeros 1 *hidden-size*))
              (c (zeros 1 *hidden-size*)))
          (setf ($ x 0 idx0) 1)
          (setf ph h
                pc c)
          (push idx0 indices)))
    (loop :for i :from 0 :below n
          :for (ht ct) = ($lstm x ph pc wi ui wf uf wo uo wa ua bi bf bo ba)
          :for yt = ($affine ht wy by)
          :for ps = ($softmax ($/ yt temperature))
          :for nidx = (choose ps)
          :do (progn
                (setf ph ht
                      pc ct)
                (push nidx indices)
                ($zero! x)
                (setf ($ x 0 nidx) 1)))
    (concatenate 'string str (rstrings (reverse indices)))))

(defparameter *upto* (- *data-size* *sequence-length* 1))

;; XXX of course, we need better strategy for building data
;; for example, breaking at the word level will be better one.
(defparameter *inputs* (loop :for p :from 0 :below *upto* :by *sequence-length*
                             :for input-str = (subseq *data* p (+ p *sequence-length*))
                             :collect (let ((m (zeros *sequence-length* *vocab-size*)))
                                        (loop :for i :from 0 :below *sequence-length*
                                              :for ch = ($ input-str i)
                                              :do (setf ($ m i ($ *char-to-idx* ch)) 1))
                                        m)))
(defparameter *targets* (loop :for p :from 0 :below *upto* :by *sequence-length*
                              :for target-str = (subseq *data* (1+ p) (+ p *sequence-length* 1))
                              :collect (let ((m (zeros *sequence-length* *vocab-size*)))
                                         (loop :for i :from 0 :below *sequence-length*
                                               :for ch = ($ target-str i)
                                               :do (setf ($ m i ($ *char-to-idx* ch)) 1))
                                         m)))

(defparameter *mloss* (* (- (log (/ 1 *vocab-size*))) *sequence-length*))
(defparameter *min-mloss* *mloss*)

($cg! *lstm*)
(gcf)

(time
 (loop :for iter :from 1 :to 5
       :for n = 0
       :for maxloss = 0
       :for maxloss-pos = -1
       :for max-mloss = 0
       :do (progn
             (loop :for input :in *inputs*
                   :for target :in *targets*
                   :do (let ((ph (zeros 1 *hidden-size*))
                             (pc (zeros 1 *hidden-size*))
                             (tloss 0))
                         (loop :for i :from 0 :below ($size input 0)
                               :for xt = ($index input 0 i)
                               :for (ht ct) = ($lstm xt ph pc *wi* *ui* *wf* *uf* *wo* *uo* *wa* *ua*
                                                     *bi* *bf* *bo* *ba*)
                               :for yt = ($affine ht *wy* *by*)
                               :for ps = ($softmax yt)
                               :for y = ($index target 0 i)
                               :for l = ($cee ps y)
                               :do (progn
                                     (setf ph ht
                                           pc ct)
                                     (incf tloss ($data l))))
                         (when (> tloss maxloss)
                           (setf maxloss-pos n)
                           (setf maxloss tloss))
                         ($rmgd! *lstm*)
                         (setf *mloss* (+ (* 0.999 *mloss*) (* 0.001 tloss)))
                         (when (> *mloss* max-mloss) (setf max-mloss *mloss*))
                         (when (zerop (rem n 100))
                           (prn "[ITER]" iter n *mloss* maxloss maxloss-pos))
                         (incf n)))
             (when (< max-mloss *min-mloss*)
               (prn "*** BETTER MLOSS - WRITE WEIGHTS: FROM" *min-mloss* "TO" max-mloss)
               (setf *min-mloss* max-mloss)
               (lstm-write-weights)))))

(prn (sample "This is not correct." 200 0.5))
(prn (sample "I" 200 0.5))

;;(lstm-write-weights)
(lstm-read-weights)

;; rmgd 0.002 0.99 -  1.31868 - 1.61637
;; adgd - 1.551497 - 1.841827
;; amgd 0.002 - 1.3747485 - 1.70623

(loop :for p :from 0 :below *upto* :by *sequence-length*
      :for n :from 0
      :for instr = (subseq *data* p (+ p *sequence-length*))
      :do (when (member n '(22257 17027 12155 10087 629 362 279 198 89))
            (prn (format nil "~6,d" n) instr)))
