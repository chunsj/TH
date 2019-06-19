;; from
;; http://karpathy.github.io/2015/05/21/rnn-effectiveness/

(defpackage :genchars0
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :genchars0)

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

(defun choose (iprobs)
  (let* ((probs ($reshape iprobs ($count iprobs)))
         (sprobs ($sum probs))
         (probs ($div probs sprobs)))
    ($ ($reshape! ($multinomial probs 1) ($count probs)) 0)))

(defun chidcs (str)
  (coerce (loop :for i :from 0 :below ($count str)
                :collect ($ *char-to-idx* ($ str i)))
          'vector))

;;
;; vanilla rnn
;;

(defparameter *hidden-size* 100)
(defparameter *sequence-length* 25)
(defparameter *learning-rate* 1E-1)

(defparameter *wxh* ($mul! (rndn *hidden-size* *vocab-size*) 0.01))
(defparameter *whh* ($mul! (rndn *hidden-size* *hidden-size*) 0.01))
(defparameter *why* ($mul! (rndn *vocab-size* *hidden-size*) 0.01))
(defparameter *bh* (zeros *hidden-size* 1))
(defparameter *by* (zeros *vocab-size* 1))

(defun lossfunc (inputs targets hprev)
  (let ((xs nil)
        (hs nil)
        (ys nil)
        (ps nil)
        (loss 0D0)
        (steps ($count inputs)))
    (push hprev hs)
    (loop :for time :from 0 :below steps
          :for xt = (let ((x (zeros *vocab-size* 1)))
                      (setf ($ x ($ inputs time) 0) 1D0)
                      x)
          :for ht-1 = (car hs)
          :for ht = ($tanh! ($add! ($add! ($@ *wxh* xt) ($@ *whh* ht-1)) *bh*))
          :for yt = ($add! ($@ *why* ht) *by*)
          :for eyt = ($exp yt)
          :for pt = ($/ eyt ($sum eyt))
          :for l = ($- ($log ($ pt ($ targets time) 0)))
          :do (progn
                (incf loss l)
                (push xt xs)
                (push ht hs)
                (push yt ys)
                (push pt ps)))
    (let ((dwxh ($zero *wxh*))
          (dwhh ($zero *whh*))
          (dwhy ($zero *why*))
          (dbh ($zero *bh*))
          (dby ($zero *by*))
          (dhnext ($zero hprev)))
      (loop :for time :from (1- steps) :downto 0
            :for pt :in ps
            :for xt :in xs
            :for (ht ht-1) :on hs
            :do (let ((dy ($clone pt))
                      (dh nil)
                      (da nil))
                  (decf ($ dy ($ targets time) 0) 1D0)
                  ($add! dwhy ($@ dy ($transpose ht)))
                  ($add! dby dy)
                  (setf dh ($+ ($@ ($transpose *why*) dy) dhnext))
                  (setf da ($mul! ($- 1 ($* ht ht)) dh))
                  ($add! dbh da)
                  ($add! dwxh ($@ da ($transpose xt)))
                  ($add! dwhh ($@ da ($transpose ht-1)))
                  (setf dhnext ($@ ($transpose *whh*) da))))
      (loop :for dparam :in (list dwxh dwhh dwhy dwhy dbh dby)
            :do ($clamp! dparam -5 5))
      (vector loss dwxh dwhh dwhy dbh dby (car hs)))))

;;(lossfunc (chidcs "hello") (chidcs "ello.") (zeros *hidden-size* 1))

(defun sample (h seed-idx n)
  (let ((x (zeros *vocab-size* 1))
        (indices (list seed-idx))
        (ht-1 h))
    (setf ($ x seed-idx 0) 1D0)
    (loop :for time :from 0 :below n
          :for ht = ($tanh! ($add! ($add! ($@ *wxh* x) ($@ *whh* ht-1)) *bh*))
          :for yt = ($add! ($@ *why* ht) *by*)
          :for eyt = ($exp yt)
          :for pt = ($/ eyt ($sum eyt))
          :for idx = (choose pt)
          :do (progn
                (setf ht-1 ht)
                (push idx indices)
                ($zero! x)
                (setf ($ x idx 0) 1D0)))
    (coerce (mapcar (lambda (i) ($ *idx-to-char* i)) (reverse indices)) 'string)))

;;(sample (zeros *hidden-size*) (random *vocab-size*) 10)

(defparameter *mwxh* ($zero *wxh*))
(defparameter *mwhh* ($zero *whh*))
(defparameter *mwhy* ($zero *why*))
(defparameter *mbh* ($zero *bh*))
(defparameter *mby* ($zero *by*))
(defparameter *smooth-loss* (* (- (log (/ 1 *vocab-size*))) *sequence-length*))

(defparameter *upto* (- *data-size* *sequence-length* 1))

(time
 (let ((n 0))
   (loop :for iter :from 1 :to 100
         :for upto = *upto*
         :for ph = (zeros *hidden-size* 1)
         :do (loop :for p :from 0 :below upto :by *sequence-length*
                   :for inputs = (chidcs (subseq *data* p (+ p *sequence-length*)))
                   :for targets = (chidcs (subseq *data* (1+ p) (+ (1+ p) *sequence-length*)))
                   :for res =(lossfunc inputs targets ph)
                   :for loss = ($ res 0)
                   :for dwxh = ($ res 1)
                   :for dwhh = ($ res 2)
                   :for dwhy = ($ res 3)
                   :for dbh = ($ res 4)
                   :for dby = ($ res 5)
                   :for hprev = ($ res 6)
                   :do (progn
                         (setf *smooth-loss* (+ (* 0.999 *smooth-loss*)
                                                (* 0.001 loss)))
                         (loop :for param :in (list *wxh* *whh* *why* *bh* *by*)
                               :for dparam :in (list dwxh dwhh dwhy dbh dby)
                               :for mem :in (list *mwxh* *mwhh* *mwhy* *mbh* *mby*)
                               :do (progn
                                     ($add! mem ($* dparam dparam))
                                     ($add! param ($/ ($* (- *learning-rate*) dparam)
                                                      ($sqrt ($+ mem 1E-8))))))
                         (when (zerop (rem n 500))
                           (prn "")
                           (prn "[ITER]" n *smooth-loss*)
                           (prn (sample hprev ($ inputs 0) 200))
                           (prn ""))
                         (incf n))))))
