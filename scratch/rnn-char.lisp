;; from
;; http://karpathy.github.io/2015/05/21/rnn-effectiveness/

(defpackage :rnn-char
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :rnn-char)

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

(defparameter *hidden-size* 100)
(defparameter *sequence-length* 25)

(defparameter *wx* ($variable ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *wh* ($variable ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *wy* ($variable ($* 0.01 (rndn *hidden-size* *vocab-size*))))
(defparameter *bh* ($variable (zeros 1 *hidden-size*)))
(defparameter *by* ($variable (zeros 1 *vocab-size*)))

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
               (loop with random = (random 1D0)
                     for (datum below) of-type (t double-float) in ranges
                     when (< random below)
                       do (return datum))))
      (pick (make-ranges)))))

(defun sample (h seed-idx n)
  (let ((x (zeros 1 *vocab-size*))
        (indices nil)
        (ph h))
    (setf ($ x 0 seed-idx) 1)
    (loop :for i :from 0 :below n
          :for xt = ($constant x)
          :for ht = ($tanh ($+ ($@ xt *wx*) ($@ ph *wh*) *bh*))
          :for yt = ($+ ($@ ht *wy*) *by*)
          :for ps = ($softmax yt)
          :for nidx = (choose ($data ps))
          :do (progn
                (setf ph ht)
                (push nidx indices)
                ($zero! x)
                (setf ($ x 0 nidx) 1)))
    (coerce (mapcar (lambda (i) ($ *idx-to-char* i)) (reverse indices)) 'string)))

(loop :for iter :from 1 :to 1
      :for n = 0
      :do (loop :for p :from 0 :below (- *data-size* *sequence-length* 1) :by *sequence-length*
                :for input = (let ((m (zeros *sequence-length* *vocab-size*)))
                               (loop :for i :from p :below (+ p *sequence-length*)
                                     :for ch = ($ *data* i)
                                     :do (setf ($ m (- i p) ($ *char-to-idx* ch)) 1))
                               m)
                :for target = (let ((m (zeros *sequence-length* *vocab-size*)))
                                (loop :for i :from (1+ p) :below (+ p *sequence-length* 1)
                                      :for ch = ($ *data* i)
                                      :do (setf ($ m (- i p 1) ($ *char-to-idx* ch)) 1))
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
                      ($bptt! losses)
                      ($adgd! ($0 losses))
                      (when (zerop (rem n 100))
                        (prn n tloss)
                        (prn (sample ph ($ *char-to-idx* ($ *data* p)) 72))
                        (gcf))
                      (incf n))))

(prn (sample ($constant (zeros 1 *hidden-size*)) (random *vocab-size*) 800))

;; reading/writing network weights - this example comes from dlfs follow-ups
(defun write-weight-to (w fname)
  (let ((f (file.disk fname "w")))
    ($fwrite ($data w) f)
    ($fclose f)))

(defun read-weight-from (w fname)
  (let ((f (file.disk fname "r")))
    ($fread ($data w) f)
    ($fclose f)))

(defun rnn-write-weights ()
  (write-weight-to *wx* "scratch/char-rnn-wx.dat")
  (write-weight-to *wh* "scratch/char-rnn-wh.dat")
  (write-weight-to *wy* "scratch/char-rnn-wy.dat")
  (write-weight-to *bh* "scratch/char-rnn-bh.dat")
  (write-weight-to *by* "scratch/char-rnn-by.dat"))

(defun rnn-read-weights ()
  (read-weight-from *wx* "scratch/char-rnn-wx.dat")
  (read-weight-from *wh* "scratch/char-rnn-wh.dat")
  (read-weight-from *wy* "scratch/char-rnn-wy.dat")
  (read-weight-from *bh* "scratch/char-rnn-bh.dat")
  (read-weight-from *by* "scratch/char-rnn-by.dat"))

(rnn-write-weights)
(rnn-read-weights)
