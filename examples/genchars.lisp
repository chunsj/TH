;; from
;; http://karpathy.github.io/2015/05/21/rnn-effectiveness/

(defpackage :genchars
  (:use #:common-lisp
        #:mu
        #:th
        #:th.ex.data))

(in-package :genchars)

(defparameter *data-lines* (text-lines :pg))
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
(defparameter *sequence-length* 50)

(defparameter *rnn* (parameters))
(defparameter *wx* ($push *rnn* ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *wh* ($push *rnn* ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *wy* ($push *rnn* ($* 0.01 (rndn *hidden-size* *vocab-size*))))
(defparameter *bh* ($push *rnn* (zeros *hidden-size*)))
(defparameter *by* ($push *rnn* (zeros *vocab-size*)))

(defun rnn-write-weight-to (w fname)
  (let ((f (file.disk fname "w")))
    ($fwrite ($data w) f)
    ($fclose f)))

(defun rnn-read-weight-from (w fname)
  (let ((f (file.disk fname "r")))
    ($fread ($data w) f)
    ($fclose f)))

(defun rnn-write-weights ()
  (rnn-write-weight-to *wx* "examples/weights/genchar/rnn-wx.dat")
  (rnn-write-weight-to *wh* "examples/weights/genchar/rnn-wh.dat")
  (rnn-write-weight-to *wy* "examples/weights/genchar/rnn-wy.dat")
  (rnn-write-weight-to *bh* "examples/weights/genchar/rnn-bh.dat")
  (rnn-write-weight-to *by* "examples/weights/genchar/rnn-by.dat"))

(defun rnn-read-weights ()
  (rnn-read-weight-from *wx* "examples/weights/genchar/rnn-wx.dat")
  (rnn-read-weight-from *wh* "examples/weights/genchar/rnn-wh.dat")
  (rnn-read-weight-from *wy* "examples/weights/genchar/rnn-wy.dat")
  (rnn-read-weight-from *bh* "examples/weights/genchar/rnn-bh.dat")
  (rnn-read-weight-from *by* "examples/weights/genchar/rnn-by.dat"))

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
        (wx ($data *wx*))
        (wh ($data *wh*))
        (bh ($data *bh*))
        (wy ($data *wy*))
        (by ($data *by*))
        (ncidx 0))
    (loop :for i :from 0 :below ($size input 0)
          :for xt = ($index input 0 i)
          :for ht = ($tanh ($affine2 xt wx ph wh bh))
          :for yt = ($affine ht wy by)
          :for ps = ($softmax ($/ yt temperature))
          :for nidx = (choose ps)
          :do (setf ph ht
                    ncidx nidx))
    (cons ncidx ph)))

(defun sample (str n &optional (temperature 1))
  (let ((x (zeros 1 *vocab-size*))
        (indices nil)
        (sh (when str (seedh str temperature)))
        (wx ($data *wx*))
        (wh ($data *wh*))
        (bh ($data *bh*))
        (wy ($data *wy*))
        (by ($data *by*))
        (ph nil))
    (if sh
        (let ((idx0 (car sh))
              (h (cdr sh)))
          (setf ($ x 0 idx0) 1)
          (setf ph h)
          (push idx0 indices))
        (let ((idx0 (random *vocab-size*))
              (h (zeros 1 *hidden-size*)))
          (setf ($ x 0 idx0) 1)
          (setf ph h)
          (push idx0 indices)))
    (loop :for i :from 0 :below n
          :for ht = ($tanh ($affine2 x wx ph wh bh))
          :for yt = ($affine ht wy by)
          :for ps = ($softmax ($/ yt temperature))
          :for nidx = (choose ps)
          :do (progn
                (setf ph ht)
                (push nidx indices)
                ($zero! x)
                (setf ($ x 0 nidx) 1)))
    (concatenate 'string str (rstrings (reverse indices)))))

(defparameter *upto* (- *data-size* *sequence-length* 1))

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

($cg! *rnn*)
(gcf)

(time
 (loop :for iter :from 1 :to 5
       :for n = 0
       :for maxloss = 0
       :do (loop :for input :in *inputs*
                 :for target :in *targets*
                 :do (let ((ph (zeros 1 *hidden-size*))
                           (tloss 0))
                       (loop :for i :from 0 :below ($size input 0)
                             :for xt = ($index input 0 i)
                             :for ht = ($tanh ($affine2 xt *wx* ph *wh* *bh*))
                             :for yt = ($affine ht *wy* *by*)
                             :for ps = ($softmax yt)
                             :for y = ($index target 0 i)
                             :for l = ($cee ps y)
                             :do (progn
                                   (setf ph ht)
                                   (incf tloss ($data l))))
                       (when (> tloss maxloss) (setf maxloss tloss))
                       ($rmgd! *rnn*)
                       (setf *mloss* (+ (* 0.999 *mloss*) (* 0.001 tloss)))
                       (when (zerop (rem n 200))
                         (prn "[ITER]" iter n *mloss* maxloss))
                       (incf n)))))

(prn (sample "This is not correct." 200 0.5))
(prn (sample "I" 200 0.5))

(rnn-write-weights)
(rnn-read-weights)

;; rmgd 0.002 0.99 -  1.31868 - 1.61637
;; adgd - 1.551497 - 1.841827
;; amgd 0.002 - 1.3747485 - 1.70623
