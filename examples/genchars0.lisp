;; from
;; http://karpathy.github.io/2015/05/21/rnn-effectiveness/

(defpackage :genchars0
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :genchars0)

(setf th::*default-tensor-class* 'tensor.double)

(defparameter *data-lines* (read-lines-from "data/tinyshakespeare.txt"))
(defparameter *data-lines* (read-lines-from "data/pg.txt"))
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

(defparameter *wxh* ($mul! (rndn *vocab-size* *hidden-size*) 0.01))
(defparameter *whh* ($mul! (rndn *hidden-size* *hidden-size*) 0.01))
(defparameter *why* ($mul! (rndn *hidden-size* *vocab-size*) 0.01))
(defparameter *bh* (zeros *hidden-size*))
(defparameter *by* (zeros *vocab-size*))

(defparameter *learning-rate* 1E-1)

(defun lossf (inputs targets hprev)
  (let ((xs nil)
        (hs nil)
        (ys nil)
        (ps nil)
        (loss 0)
        (dwxh ($zero *wxh*))
        (dwhh ($zero *whh*))
        (dwhy ($zero *why*))
        (dbh ($zero *bh*))
        (dby ($zero *by*))
        (dhnext ($zero hprev))
        (steps ($count inputs)))
    (push hprev hs)
    (loop :for time :from 0 :below steps
          :for xt = (let ((xx (zeros 1 *vocab-size*)))
                      (setf ($ xx 0 ($ inputs time)) 1)
                      xx)
          :for tt = (let ((xx (zeros *vocab-size*)))
                      (setf ($ xx ($ targets time)) 1)
                      xx)
          :for ht-1 = (car hs)
          :for ht = ($tanh ($affine2 xt *wxh* ht-1 *whh* *bh*))
          :for yt = ($affine ht *why* *by*)
          :for eyt = ($exp yt)
          :for pt = ($/ eyt ($sum eyt))
          :do (let ((p ($ pt 0 ($ targets time))))
                (decf loss ($log p))
                (push xt xs)
                (push ht hs)
                (push yt ys)
                (push pt ps)))
    (loop :for time :from (1- steps) :downto 0
          :for pt :in ps
          :for (ht ht-1) :on hs
          :for xt :in xs
          :do (let ((dy ($clone pt))
                    (dh nil)
                    (da nil))
                (decf ($ dy 0 ($ targets time)) 1)
                ($add! dwhy ($@ ($transpose ht) dy))
                ($add! dby ($reshape dy *vocab-size*))
                (setf dh ($add! ($@ dy ($transpose dwhy)) dhnext))
                (setf da ($mul! ($- 1 ($* ht ht)) dh))
                ($add! dbh da)
                ($add! dwxh ($@ ($transpose xt) da))
                ($add! dwhh ($@ ($transpose ht-1) da))
                (setf dhnext ($@ da ($transpose *whh*)))))
    (loop :for dp :in (list dwxh dwhh dwhy dbh dby)
          :do ($clamp! dp -5 5))
    (vector loss dwxh dwhh dwhy dbh dby (car hs))))

(defun charidcs (str)
  (coerce (loop :for i :from 0 :below ($count str)
                :collect ($ *char-to-idx* ($ str i)))
          'vector))

(prn (lossf (charidcs "hello") (charidcs "ello.") (zeros 1 *hidden-size*)))
(prn (lossf (charidcs "The biggest component in melody")
            (charidcs "he biggest component in melody.")
            (zeros 1 *hidden-size*)))

(defun sample (h seed-idx n)
  (let ((x (zeros 1 *vocab-size*))
        (indices (list seed-idx))
        (ph h))
    (setf ($ x 0 seed-idx) 1)
    (loop :for i :from 0 :below n
          :for ht = ($tanh ($affine2 x *wxh* ph *whh* *bh*))
          :for yt = ($affine ht *why* *by*)
          :for eyt = ($exp yt)
          :for pt =  ($/ eyt ($sum eyt))
          :for nidx = (choose pt)
          :do (progn
                (setf ph ht)
                (push nidx indices)
                ($zero! x)
                (setf ($ x 0 nidx) 1)))
    (coerce (mapcar (lambda (i) ($ *idx-to-char* i)) (reverse indices)) 'string)))

(prn (sample (zeros 1 *hidden-size*) (random *vocab-size*) 50))

(defparameter *upto* (- *data-size* *sequence-length* 1))

(defparameter *mwxh* ($zero *wxh*))
(defparameter *mwhh* ($zero *whh*))
(defparameter *mwhy* ($zero *why*))
(defparameter *mbh* ($zero *bh*))
(defparameter *mby* ($zero *by*))
(defparameter *smooth-loss* (* (- (log (/ 1 *vocab-size*))) *sequence-length*))

(setf *learning-rate* 0.001)

(time
 (let ((n 0))
   (loop :for iter :from 1 :to 50
         :for upto = *upto*
         :for ph = (zeros 1 *hidden-size*)
         :do (loop :for p :from 0 :below upto :by *sequence-length*
                   :for input-str = (subseq *data* p (+ p *sequence-length*))
                   :for target-str = (subseq *data* (1+ p) (+ p *sequence-length* 1))
                   :for inputs = (charidcs input-str)
                   :for targets = (charidcs target-str)
                   :do (let* ((loss-res (lossf inputs targets ph))
                              (loss ($ loss-res 0))
                              (dwxh ($ loss-res 1))
                              (dwhh ($ loss-res 2))
                              (dwhy ($ loss-res 3))
                              (dbh ($ loss-res 4))
                              (dby ($ loss-res 5))
                              (hprev ($ loss-res 6)))
                         (setf *smooth-loss* (+ (* 0.999 *smooth-loss*)
                                                (* 0.001 loss)))
                         (loop :for p :in (list *wxh* *whh* *why* *bh* *by*)
                               :for dp :in (list dwxh dwhh dwhy dbh dby)
                               :for mp :in (list *mwxh* *mwhh* *mwhy* *mbh* *mby*)
                               :do (progn
                                     ($add! mp ($* dp dp))
                                     ($add! p ($/ ($* dp (- *learning-rate*))
                                                  ($sqrt ($+ mp 1E-8))))))
                         (when (zerop (rem n 500))
                           (prn "")
                           (prn "[ITER]" n *smooth-loss* loss)
                           (prn (sample hprev ($ targets 0) 200))
                           (prn ""))
                         (incf n))))))

(prn (sample (zeros 1 *hidden-size*) (random *vocab-size*) 200))
