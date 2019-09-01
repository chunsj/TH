;; from
;; http://karpathy.github.io/2015/05/21/rnn-effectiveness/

(defpackage :genchars-obama-lstm2
  (:use #:common-lisp
        #:mu
        #:mdt
        #:th
        #:th.ex.data))

(in-package :genchars-obama-lstm2)

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

(defparameter *wa1* ($push *lstm* ($- ($* 0.16 (rnd *vocab-size* *hidden-size*)) 0.08)))
(defparameter *ua1* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *ba1* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size*)) 0.08)))

(defparameter *wi1* ($push *lstm* ($- ($* 0.16 (rnd *vocab-size* *hidden-size*)) 0.08)))
(defparameter *ui1* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bi1* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size*)) 0.08)))

(defparameter *wf1* ($push *lstm* ($- ($* 0.16 (rnd *vocab-size* *hidden-size*)) 0.08)))
(defparameter *uf1* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bf1* ($push *lstm* (ones *hidden-size*)))

(defparameter *wo1* ($push *lstm* ($- ($* 0.16 (rnd *vocab-size* *hidden-size*)) 0.08)))
(defparameter *uo1* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bo1* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size*)) 0.08)))

(defparameter *wa2* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *ua2* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *ba2* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size*)) 0.08)))

(defparameter *wi2* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *ui2* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bi2* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size*)) 0.08)))

(defparameter *wf2* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *uf2* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bf2* ($push *lstm* (ones *hidden-size*)))

(defparameter *wo2* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *uo2* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size* *hidden-size*)) 0.08)))
(defparameter *bo2* ($push *lstm* ($- ($* 0.16 (rnd *hidden-size*)) 0.08)))

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
  (lstm-write-weight-to *wa1* "examples/weights/genchar-obama-lstm2/lstm-wa1.dat")
  (lstm-write-weight-to *ua1* "examples/weights/genchar-obama-lstm2/lstm-ua1.dat")
  (lstm-write-weight-to *ba1* "examples/weights/genchar-obama-lstm2/lstm-ba1.dat")
  (lstm-write-weight-to *wi1* "examples/weights/genchar-obama-lstm2/lstm-wi1.dat")
  (lstm-write-weight-to *ui1* "examples/weights/genchar-obama-lstm2/lstm-ui1.dat")
  (lstm-write-weight-to *bi1* "examples/weights/genchar-obama-lstm2/lstm-bi1.dat")
  (lstm-write-weight-to *wf1* "examples/weights/genchar-obama-lstm2/lstm-wf1.dat")
  (lstm-write-weight-to *uf1* "examples/weights/genchar-obama-lstm2/lstm-uf1.dat")
  (lstm-write-weight-to *bf1* "examples/weights/genchar-obama-lstm2/lstm-bf1.dat")
  (lstm-write-weight-to *wo1* "examples/weights/genchar-obama-lstm2/lstm-wo1.dat")
  (lstm-write-weight-to *uo1* "examples/weights/genchar-obama-lstm2/lstm-uo1.dat")
  (lstm-write-weight-to *bo1* "examples/weights/genchar-obama-lstm2/lstm-bo1.dat")
  (lstm-write-weight-to *wa2* "examples/weights/genchar-obama-lstm2/lstm-wa2.dat")
  (lstm-write-weight-to *ua2* "examples/weights/genchar-obama-lstm2/lstm-ua2.dat")
  (lstm-write-weight-to *ba2* "examples/weights/genchar-obama-lstm2/lstm-ba2.dat")
  (lstm-write-weight-to *wi2* "examples/weights/genchar-obama-lstm2/lstm-wi2.dat")
  (lstm-write-weight-to *ui2* "examples/weights/genchar-obama-lstm2/lstm-ui2.dat")
  (lstm-write-weight-to *bi2* "examples/weights/genchar-obama-lstm2/lstm-bi2.dat")
  (lstm-write-weight-to *wf2* "examples/weights/genchar-obama-lstm2/lstm-wf2.dat")
  (lstm-write-weight-to *uf2* "examples/weights/genchar-obama-lstm2/lstm-uf2.dat")
  (lstm-write-weight-to *bf2* "examples/weights/genchar-obama-lstm2/lstm-bf2.dat")
  (lstm-write-weight-to *wo2* "examples/weights/genchar-obama-lstm2/lstm-wo2.dat")
  (lstm-write-weight-to *uo2* "examples/weights/genchar-obama-lstm2/lstm-uo2.dat")
  (lstm-write-weight-to *bo2* "examples/weights/genchar-obama-lstm2/lstm-bo2.dat")
  (lstm-write-weight-to *wy* "examples/weights/genchar-obama-lstm2/lstm-wy.dat")
  (lstm-write-weight-to *by* "examples/weights/genchar-obama-lstm2/lstm-by.dat"))

(defun lstm-read-weights ()
  (lstm-read-weight-from *wa1* "examples/weights/genchar-obama-lstm2/lstm-wa1.dat")
  (lstm-read-weight-from *ua1* "examples/weights/genchar-obama-lstm2/lstm-ua1.dat")
  (lstm-read-weight-from *ba1* "examples/weights/genchar-obama-lstm2/lstm-ba1.dat")
  (lstm-read-weight-from *wi1* "examples/weights/genchar-obama-lstm2/lstm-wi1.dat")
  (lstm-read-weight-from *ui1* "examples/weights/genchar-obama-lstm2/lstm-ui1.dat")
  (lstm-read-weight-from *bi1* "examples/weights/genchar-obama-lstm2/lstm-bi1.dat")
  (lstm-read-weight-from *wf1* "examples/weights/genchar-obama-lstm2/lstm-wf1.dat")
  (lstm-read-weight-from *uf1* "examples/weights/genchar-obama-lstm2/lstm-uf1.dat")
  (lstm-read-weight-from *bf1* "examples/weights/genchar-obama-lstm2/lstm-bf1.dat")
  (lstm-read-weight-from *wo1* "examples/weights/genchar-obama-lstm2/lstm-wo1.dat")
  (lstm-read-weight-from *uo1* "examples/weights/genchar-obama-lstm2/lstm-uo1.dat")
  (lstm-read-weight-from *bo1* "examples/weights/genchar-obama-lstm2/lstm-bo1.dat")
  (lstm-read-weight-from *wa2* "examples/weights/genchar-obama-lstm2/lstm-wa2.dat")
  (lstm-read-weight-from *ua2* "examples/weights/genchar-obama-lstm2/lstm-ua2.dat")
  (lstm-read-weight-from *ba2* "examples/weights/genchar-obama-lstm2/lstm-ba2.dat")
  (lstm-read-weight-from *wi2* "examples/weights/genchar-obama-lstm2/lstm-wi2.dat")
  (lstm-read-weight-from *ui2* "examples/weights/genchar-obama-lstm2/lstm-ui2.dat")
  (lstm-read-weight-from *bi2* "examples/weights/genchar-obama-lstm2/lstm-bi2.dat")
  (lstm-read-weight-from *wf2* "examples/weights/genchar-obama-lstm2/lstm-wf2.dat")
  (lstm-read-weight-from *uf2* "examples/weights/genchar-obama-lstm2/lstm-uf2.dat")
  (lstm-read-weight-from *bf2* "examples/weights/genchar-obama-lstm2/lstm-bf2.dat")
  (lstm-read-weight-from *wo2* "examples/weights/genchar-obama-lstm2/lstm-wo2.dat")
  (lstm-read-weight-from *uo2* "examples/weights/genchar-obama-lstm2/lstm-uo2.dat")
  (lstm-read-weight-from *bo2* "examples/weights/genchar-obama-lstm2/lstm-bo2.dat")
  (lstm-read-weight-from *wy* "examples/weights/genchar-obama-lstm2/lstm-wy.dat")
  (lstm-read-weight-from *by* "examples/weights/genchar-obama-lstm2/lstm-by.dat"))

(defun cindices (str)
  (let ((m (zeros ($count str) *vocab-size*)))
    (loop :for i :from 0 :below ($count str)
          :for ch = ($ str i)
          :do (setf ($ m i ($ *char-to-idx* ch)) 1))
    m))

(defun rstrings (indices) (coerce (mapcar (lambda (i) ($ *idx-to-char* i)) indices) 'string))

(defun seedh (str &optional (temperature 1))
  (let ((input (cindices str))
        (ph1 (zeros 1 *hidden-size*))
        (pc1 (zeros 1 *hidden-size*))
        (ph2 (zeros 1 *hidden-size*))
        (pc2 (zeros 1 *hidden-size*))
        (wa1 ($data *wa1*))
        (ua1 ($data *ua1*))
        (ba1 ($data *ba1*))
        (wi1 ($data *wi1*))
        (ui1 ($data *ui1*))
        (bi1 ($data *bi1*))
        (wf1 ($data *wf1*))
        (uf1 ($data *uf1*))
        (bf1 ($data *bf1*))
        (wo1 ($data *wo1*))
        (uo1 ($data *uo1*))
        (bo1 ($data *bo1*))
        (wa2 ($data *wa2*))
        (ua2 ($data *ua2*))
        (ba2 ($data *ba2*))
        (wi2 ($data *wi2*))
        (ui2 ($data *ui2*))
        (bi2 ($data *bi2*))
        (wf2 ($data *wf2*))
        (uf2 ($data *uf2*))
        (bf2 ($data *bf2*))
        (wo2 ($data *wo2*))
        (uo2 ($data *uo2*))
        (bo2 ($data *bo2*))
        (wy ($data *wy*))
        (by ($data *by*))
        (ncidx 0))
    (loop :for i :from 0 :below ($size input 0)
          :for xt = ($index input 0 i)
          :for (ht1 ct1) = ($lstm xt ph1 pc1 wi1 ui1 wf1 uf1 wo1 uo1 wa1 ua1 bi1 bf1 bo1 ba1)
          :for (ht2 ct2) = ($lstm ht1 ph2 pc2 wi2 ui2 wf2 uf2 wo2 uo2 wa2 ua2 bi2 bf2 bo2 ba2)
          :for yt = ($affine ht2 wy by)
          :for ps = ($softmax ($/ yt temperature))
          :for nidx = (choose ps)
          :do (setf ph1 ht1
                    pc1 ct1
                    ph2 ht2
                    pc2 ct2
                    ncidx nidx))
    (list ncidx ph1 pc1 ph2 pc2)))

(defun sample (str n &optional (temperature 1))
  (let ((x (zeros 1 *vocab-size*))
        (indices nil)
        (sh (when str (seedh str temperature)))
        (wa1 ($data *wa1*))
        (ua1 ($data *ua1*))
        (ba1 ($data *ba1*))
        (wi1 ($data *wi1*))
        (ui1 ($data *ui1*))
        (bi1 ($data *bi1*))
        (wf1 ($data *wf1*))
        (uf1 ($data *uf1*))
        (bf1 ($data *bf1*))
        (wo1 ($data *wo1*))
        (uo1 ($data *uo1*))
        (bo1 ($data *bo1*))
        (wa2 ($data *wa2*))
        (ua2 ($data *ua2*))
        (ba2 ($data *ba2*))
        (wi2 ($data *wi2*))
        (ui2 ($data *ui2*))
        (bi2 ($data *bi2*))
        (wf2 ($data *wf2*))
        (uf2 ($data *uf2*))
        (bf2 ($data *bf2*))
        (wo2 ($data *wo2*))
        (uo2 ($data *uo2*))
        (bo2 ($data *bo2*))
        (wy ($data *wy*))
        (by ($data *by*))
        (ph1 nil)
        (pc1 nil)
        (ph2 nil)
        (pc2 nil))
    (if sh
        (let ((idx0 ($0 sh))
              (h1 ($1 sh))
              (c1 ($2 sh))
              (h2 ($3 sh))
              (c2 ($4 sh)))
          (setf ($ x 0 idx0) 1)
          (setf ph1 h1
                pc1 c1
                ph2 h2
                pc2 c2)
          (push idx0 indices))
        (let ((idx0 (random *vocab-size*))
              (h1 (zeros 1 *hidden-size*))
              (c1 (zeros 1 *hidden-size*))
              (h2 (zeros 1 *hidden-size*))
              (c2 (zeros 1 *hidden-size*)))
          (setf ($ x 0 idx0) 1)
          (setf ph1 h1
                pc1 c1
                ph2 h2
                pc2 c2)
          (push idx0 indices)))
    (loop :for i :from 0 :below n
          :for (ht1 ct1) = ($lstm x ph1 pc1 wi1 ui1 wf1 uf1 wo1 uo1 wa1 ua1 bi1 bf1 bo1 ba1)
          :for (ht2 ct2) = ($lstm ht1 ph2 pc2 wi2 ui2 wf2 uf2 wo2 uo2 wa2 ua2 bi2 bf2 bo2 ba2)
          :for yt = ($affine ht2 wy by)
          :for ps = ($softmax ($/ yt temperature))
          :for nidx = (choose ps)
          :do (progn
                (setf ph1 ht1
                      pc1 ct1
                      ph2 ht2
                      pc2 ct2)
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
                                        ($contiguous! m))))
(defparameter *targets* (loop :for p :from 0 :below *upto* :by *sequence-length*
                              :for target-str = (subseq *data* (1+ p) (+ p *sequence-length* 1))
                              :collect (let ((m (zeros *sequence-length* *vocab-size*)))
                                         (loop :for i :from 0 :below *sequence-length*
                                               :for ch = ($ target-str i)
                                               :do (setf ($ m i ($ *char-to-idx* ch)) 1))
                                         ($contiguous! m))))

(defparameter *mloss* (* (- (log (/ 1 *vocab-size*))) *sequence-length*))
(defparameter *min-mloss* *mloss*)

(defparameter *ph1* (zeros 1 *hidden-size*))
(defparameter *pc1* (zeros 1 *hidden-size*))
(defparameter *ph2* (zeros 1 *hidden-size*))
(defparameter *pc2* (zeros 1 *hidden-size*))

(defun train (epochs)
  (loop :for iter :from 1 :to epochs
        :for n = 0
        :for maxloss = 0
        :for maxloss-pos = -1
        :for max-mloss = 0
        :do (progn
              (loop :for input :in *inputs*
                    :for target :in *targets*
                    :do (let ((ph1 *ph1*)
                              (pc1 *pc1*)
                              (ph2 *ph2*)
                              (pc2 *pc2*)
                              (tloss 0))
                          (loop :for i :from 0 :below ($size input 0)
                                :for xt = ($index input 0 i)
                                ;;:for xt = ($reshape ($ input i) 1 86)
                                :for (ht1 ct1) = ($lstm xt ph1 pc1 *wi1* *ui1* *wf1* *uf1*
                                                        *wo1* *uo1* *wa1* *ua1*
                                                        *bi1* *bf1* *bo1* *ba1*)
                                :for (ht2 ct2) = ($lstm ht1 ph2 pc2 *wi2* *ui2* *wf2* *uf2*
                                                        *wo2* *uo2* *wa2* *ua2*
                                                        *bi2* *bf2* *bo2* *ba2*)
                                :for yt = ($affine ht2 *wy* *by*)
                                :for ps = ($softmax yt)
                                :for y = ($index target 0 i)
                                ;;:for y = ($reshape ($ target 0) 1 86)
                                :for l = ($cee ps y)
                                :do (progn
                                      (setf ph1 ht1
                                            pc1 ct1
                                            ph2 ht2
                                            pc2 ct2)
                                      (incf tloss ($data l))))
                          (when (> tloss maxloss)
                            (setf maxloss-pos n)
                            (setf maxloss tloss))
                          ;;($rmgd! *lstm*)
                          ($gd! *lstm* 0.0005)
                          (setf *mloss* (+ (* 0.999 *mloss*) (* 0.001 tloss)))
                          (when (> *mloss* max-mloss) (setf max-mloss *mloss*))
                          (when (zerop (rem n 100))
                            (prn "[ITER]" iter n *mloss* maxloss maxloss-pos (now)))
                          (incf n)))
              (when (< max-mloss *min-mloss*)
                (prn "*** BETTER MLOSS - WRITE WEIGHTS: FROM" *min-mloss* "TO" max-mloss)
                (setf *min-mloss* max-mloss)
                (lstm-write-weights)))))

($cg! *lstm*)
(gcf)

(setf *min-mloss* 103.9918) ;; updated
(setf *mloss* *min-mloss*)

(time (train 40))

(prn (sample "This is not correct." 200 0.5))
(prn (sample "I" 200 0.5))

;;(lstm-write-weights)
(lstm-read-weights)

;; rmgd 0.002 0.99 -  1.31868 - 1.61637
;; adgd - 1.551497 - 1.841827
;; amgd 0.002 - 1.3747485 - 1.70623

(loop :for p :from 0 :below *upto* :by *sequence-length*
      :for n :from 0
      :for input-str = (subseq *data* p (+ p *sequence-length*))
      :do (when (member n '(39773 25383 22257 12208 12155 629 362 171 32))
            (prn (format nil "~6,d" n) input-str)))

;; 20~22% - ccl
;; 52~54% - sbcl

(setf *data-lines* nil
      *data* nil
      *inputs* nil
      *targets* nil)
