(defpackage :eng-fra
  (:use #:common-lisp
        #:mu
        #:th
        #:th.ex.data
        #:th.layers
        #:th.text))

(in-package :eng-fra)

(defparameter *fra-eng* (mapcar (lambda (pair) (reverse pair)) (eng-fra-small-processed)))
(defparameter *eng-prefixes* '("i am " "i m "
                               "he is " "he s "
                               "she is " "she s "
                               "you are " "you re "
                               "we are " "we re "
                               "they are " "they re "))

(defun starts-with (s prefixes)
  (loop :for k :in prefixes
        :when (and (>= ($count s) ($count k))
                   (string-equal (subseq s 0 ($count k)) k))
          :do (return T)))

(defparameter *pairs* (->> *fra-eng*
                           (filter (lambda (pair)
                                     (let ((fra ($0 pair))
                                           (eng ($1 pair)))
                                       (and (< ($count (split #\Space fra)) 10)
                                            (< ($count (split #\Space eng)) 10)
                                            (starts-with eng *eng-prefixes*)))))
                           (mapcar (lambda (pair)
                                     (let ((fra ($0 pair))
                                           (eng ($1 pair)))
                                       (list (split #\Space fra) (split #\Space eng)))))))

(defparameter *fra-encoder* (word-encoder (append '("SOS" "EOS") (flatten (mapcar #'$0 *pairs*)))))
(defparameter *eng-encoder* (word-encoder (append '("SOS" "EOS") (flatten (mapcar #'$1 *pairs*)))))

(defparameter *input-max-length* (reduce #'max (mapcar (lambda (pair) ($count ($0 pair))) *pairs*)))
(defparameter *output-max-length* (reduce #'max (mapcar (lambda (pair) ($count ($1 pair))) *pairs*)))

(defun fill-eos (sentence max-length)
  (let ((n ($count sentence)))
    (if (< n max-length)
        (append sentence (loop :repeat (- max-length n) :collect "EOS"))
        sentence)))

(defun encode-sentences (encoder sentences max-length)
  (encoder-encode encoder (mapcar (lambda (sentence) (fill-eos sentence max-length)) sentences)))

(defun build-batches (encoder max-length data n)
  (loop :for tail :on data :by (lambda (l) (nthcdr n l))
        :collect (encode-sentences encoder (subseq tail 0 (min ($count tail) n)) max-length)))

(defparameter *batch-size* 100)
(defparameter *hidden-size* 256)

(defparameter *train-xs-batches* (build-batches *fra-encoder* *input-max-length*
                                                (mapcar #'$0 *pairs*)
                                                *batch-size*))
(defparameter *train-ys-batches* (build-batches *eng-encoder* *output-max-length*
                                                (mapcar #'$1 *pairs*)
                                                *batch-size*))

(defparameter *overfit-xs-batches* (subseq *train-xs-batches* 0 1))
(defparameter *overfit-ys-batches* (subseq *train-ys-batches* 0 1))

(defclass encoder-network (layer)
  ((rnn :initform nil :accessor $encoder-rnn)))

(defun encoder-network (encoder nhidden)
  (let ((n (make-instance 'encoder-network))
        (vsize (encoder-vocabulary-size encoder))
        (wvecsz 32))
    (with-slots (rnn) n
      (setf rnn (sequential-layer
                 (recurrent-layer (affine-cell vsize wvecsz
                                               :activation nil
                                               :biasp nil))
                 (recurrent-layer (lstm-cell wvecsz nhidden)))))
    n))

(defmethod $execute ((layer encoder-network) x &key (trainp T))
  (with-slots (rnn) layer
    ($execute rnn x :trainp trainp)))

(defun $encoder-network-state (layer)
  ($cell-state ($cell ($ ($encoder-rnn layer) 1))))

(defclass decoder-network (layer)
  ((rnn :initform nil :accessor $decoder-rnn)))

(defun decoder-network (encoder nhidden)
  (let ((n (make-instance 'encoder-network))
        (vsize (encoder-vocabulary-size encoder))
        (wvecsz 32))
    (with-slots (rnn) n
      (setf rnn (sequential-layer
                 (recurrent-layer (affine-cell vsize wvecsz
                                               :activation :nil
                                               :biasp nil))
                 (recurrent-layer (lstm-cell wvecsz nhidden))
                 (recurrent-layer
                  (sequential-layer
                   (parallel-layer (attention-cell)
                                   (functional-layer
                                    (lambda (q &key (trainp t))
                                     (declare (ignore trainp))
                                     q)))
                   (functional-layer
                    (lambda (c q &key (trainp t))
                     (declare (ignore trainp))
                     ($cat q c 1)))))
                 (recurrent-layer (affine-cell (* 2 nhidden) vsize
                                               :activation :nil)))))
    n))

(defmethod $execute ((layer decoder-network) x &key (trainp T))
  (with-slots (rnn) layer
    ($execute layer x :trainp trainp)))

(defun $update-decoder-network-state! (layer h0)
  (with-slots (rnn) layer
    ($update-cell-state! ($ rnn 1) h0)))

(defun $update-attention-memory! (layer hs)
  (with-slots (rnn) layer
    ($set-memory! ($ ($ ($cell ($ rnn 2)) 0) 0) (concat-sequence hs))))

(defclass seq2seq ()
  ((from-encoder :initform nil :accessor $seq2seq-from-encoder)
   (to-encoder :initform nil :accessor $seq2seq-to-encoder)
   (encoder-network :initform nil :accessor $seq2seq-encoder-network)
   (decoder-network :initform nil :accessor $seq2seq-decoder-network)))

(defun seq2seq (from-encoder to-encoder nhidden)
  (let ((n (make-instance 'seq2seq)))
    (with-slots (encoder-network decoder-network) n
      (setf ($seq2seq-from-encoder n) from-encoder
            ($seq2seq-to-encoder n) to-encoder)
      (setf encoder-network (encoder-network from-encoder nhidden)
            decoder-network (decoder-network to-encoder nhidden)))
    n))

(defun $execute-seq2seq (s2s xs ts)
  (let ((hs ($execute ($seq2seq-encoder-network s2s) xs))
        (h0 ($encoder-network-state ($seq2seq-encoder-network s2s))))
    ($update-decoder-network-state! ($seq2seq-decoder-network s2s) h0)
    ($update-attention-memory! ($seq2seq-decoder-network s2s) hs)
    (with-keeping-state (($seq2seq-decoder-network s2s))
      (let* ((batch-size ($size (car xs) 0))
             (ys (append (list ($fill (tensor.long batch-size) 0)
                               (butlast ts)))))
        ($execute ($seq2seq-decoder-network s2s) ys)))))

(defun $compute-loss (s2s xs ts)
  (let* ((ys ($execute-seq2seq s2s xs ts))
         (losses (mapcar (lambda (y c) ($cec y c)) ys ts)))
    ($div (apply #'$+ losses) ($count losses))))

(defun $generate-seq2seq (s2s hs h0 xs0 n)
  (let ((sampled '())
        (xts xs0)
        (batch-size ($size (car xs0) 0)))
    ($update-decoder-network-state! ($seq2seq-decoder-network s2s) h0)
    ($update-attention-memory! ($seq2seq-decoder-network s2s) hs)
    (with-keeping-state (($seq2seq-decoder-network s2s))
      (loop :for i :from 0 :below n
            :do (let* ((yts ($evaluate ($seq2seq-decoder-network s2s) xts))
                       (rts (encoder-choose ($seq2seq-to-encoder s2s) yts -1)))
                  (push rts sampled)
                  (setf xts (encoder-encode ($seq2seq-to-encoder s2s) rts)))))
    (let ((res (reverse sampled))
          (results (make-list batch-size)))
      (loop :for r :in res
            :do (loop :for v :in r
                      :for i :from 0
                      :do (push v ($ results i))))
      (mapcar #'reverse results))))

(defun $evaluate-seq2seq (s2s xs &optional (n 9))
  (let ((hs ($evaluate ($seq2seq-encoder-network s2s) xs))
        (h0 ($encoder-network-state ($seq2seq-encoder-network s2s))))
    ($generate-seq2seq s2s hs h0
                       (list ($fill (tensor.long ($size (car xs) 0)) 0))
                       n)))

(defun $matches-score (s2s ts ys)
  (let ((tss (encoder-decode ($seq2seq-to-encoder s2s) ts))
        (yss ys))
    (let ((matches (mapcar (lambda (tn yn) (if (string-equal tn yn) 0 1)) tss yss)))
      (* 1D0 (/ (reduce #'+ matches) ($count matches))))))

(defun gd! (s2s fn lr)
  (funcall fn ($seq2seq-decoder-network s2s) lr)
  (funcall fn ($seq2seq-encoder-network s2s) lr))

(defun $train (s2s xss tss &key (epochs 10) (pstep 100) (gdfn #'$adgd!) (lr 1D0))
  (let ((sz ($count xss)))
    (loop :for epoch :from 0 :below epochs
          :do (loop :for xsi :in xss
                    :for ts :in tss
                    :for idx :from 0
                    :for iter = (+ idx (* epoch sz))
                    :for xs = xsi
                    :do (let ((loss ($compute-loss s2s xs ts)))
                          (gd! s2s gdfn lr)
                          (when (zerop (rem iter pstep))
                            (let* ((lv ($data loss))
                                   (ys ($evaluate-seq2seq s2s xs))
                                   (score ($matches-score s2s ts ys)))
                              (prn iter lv score)
                              (prn "TS" (encoder-decode ($seq2seq-to-encoder s2s) ts))
                              (prn "YS" ys)
                              (prn "=="))))))))

(defmethod $reset! ((s2s seq2seq))
  ($reset! ($seq2seq-encoder-network s2s))
  ($reset! ($seq2seq-decoder-network s2s)))

(defparameter *s2s* (seq2seq *fra-encoder* *eng-encoder* 256))
($reset! *s2s*)

($evaluate-seq2seq *s2s* ($0 *overfit-xs-batches*))

(time ($train *s2s* *overfit-xs-batches* *overfit-ys-batches* :epochs 1 :pstep 100))
