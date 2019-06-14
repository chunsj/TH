;; from
;; http://karpathy.github.io/2015/05/21/rnn-effectiveness/

(defpackage :genchars0
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :genchars0)

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
          :for xst = (let ((xx (zeros 1 *vocab-size*)))
                       (setf ($ xx 0 ($ *char-to-idx* ($ inputs time))) 1)
                       xx)
          :for hst = ($tanh ($affine2 xst *wxh* (car hs) *whh* *bh*))
          :for yst = ($affine hst *why* *by*)
          :for pst = (let ((exp-yst ($exp ($/ yst ($max yst)))))
                       ($/ exp-yst ($sum exp-yst)))
          :do (let ((p ($ pst 0 ($ *char-to-idx* ($ targets time)))))
                (decf loss ($log p))
                (push xst xs)
                (push hst hs)
                (push yst ys)
                (push pst ps)))
    (loop :for itime :from 0 :below steps
          :for time = (- steps itime 1)
          :for pst :in ps
          :for (hst hstp) :on hs
          :for xst :in xs
          :do (let ((dy ($clone pst))
                    (dh nil)
                    (dhraw nil))
                (decf ($ dy 0 ($ *char-to-idx* ($ targets time))) 1)
                ($add! dwhy ($@ ($transpose hst) dy))
                ($add! dby ($reshape dy *vocab-size*))
                (setf dh ($+ dhnext ($@ dy ($transpose dwhy))))
                (setf dhraw ($* dh ($- 1 ($* hst hst))))
                ($add! dbh dhraw)
                ($add! dwxh ($@ ($transpose xst) dhraw))
                ($add! dwhh ($@ ($transpose hstp) dhraw))
                (setf dhnext ($@ dhraw ($transpose *whh*)))))
    (loop :for dp :in (list dwxh dwhh dwhy dbh dby)
          :do ($clamp! dp -5 5))
    (list loss dwxh dwhh dwhy dbh dby (car hs))))

(lossf "hello" "ello." (zeros 1 *hidden-size*))
(prn (lossf "hello" "ello." (zeros 1 *hidden-size*)))
(prn (lossf "The biggest component in " "he biggest component in m"
            (zeros 1 *hidden-size*)))

(defun sample (h seed-idx n)
  (let ((x (zeros 1 *vocab-size*))
        (indices (list seed-idx))
        (ph h))
    (setf ($ x 0 seed-idx) 1)
    (loop :for i :from 0 :below n
          :for ht = ($tanh ($affine2 x *wxh* ph *whh* *bh*))
          :for yt = ($affine ht *why* *by*)
          :for ps =  (let ((exp-yst ($exp ($/ yt ($max yt)))))
                       ($/ exp-yst ($sum exp-yst)))
          :for nidx = (choose ps)
          :do (progn
                (setf ph ht)
                (push nidx indices)
                ($zero! x)
                (setf ($ x 0 nidx) 1)))
    (coerce (mapcar (lambda (i) ($ *idx-to-char* i)) (reverse indices)) 'string)))

(sample (zeros 1 *hidden-size*) (random *vocab-size*) 50)

(defparameter *upto* (- *data-size* *sequence-length* 1))

(defparameter *mwxh* ($zero *wxh*))
(defparameter *mwhh* ($zero *whh*))
(defparameter *mwhy* ($zero *why*))
(defparameter *mbh* ($zero *bh*))
(defparameter *mby* ($zero *by*))
(defparameter *smooth-loss* (* (- (log (/ 1 *vocab-size*))) *sequence-length*))

(time
 (loop :for iter :from 1 :to 2000
       :for n = 0
       :for upto = *upto*
       :do (loop :for p :from 0 :below upto :by *sequence-length*
                 :for input-str = (subseq *data* p (+ p *sequence-length*))
                 :for target-str = (subseq *data* (1+ p) (+ p *sequence-length* 1))
                 :do (let* ((ph (zeros 1 *hidden-size*))
                            (loss-res (lossf input-str target-str ph))
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
                       (when (zerop (rem n 200))
                         (prn "")
                         (prn "[ITER]" n *smooth-loss*)
                         (prn (sample hprev ($ *char-to-idx* ($ input-str 0)) 200))
                         (prn ""))
                       (incf n)))))

(prn (sample (zeros 1 *hidden-size*) (random *vocab-size*) 200))
