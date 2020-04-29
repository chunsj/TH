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
                                       (list (split #\Space fra) (split #\Space eng)))))
                           (alexandria:shuffle)))

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

(defparameter *xs-batches* (build-batches *fra-encoder* *input-max-length*
                                          (mapcar #'$0 *pairs*)
                                          *batch-size*))
(defparameter *ys-batches* (build-batches *eng-encoder* *output-max-length*
                                          (mapcar #'$1 *pairs*)
                                          *batch-size*))

(defparameter *train-xs-batches* (subseq *xs-batches* 0 50))
(defparameter *train-ys-batches* (subseq *ys-batches* 0 50))

(defparameter *test-xs-batches* (subseq *xs-batches* 50 60))
(defparameter *test-ys-batches* (subseq *ys-batches* 50 60))

(defparameter *overfit-xs-batches* (subseq (build-batches *fra-encoder* *input-max-length*
                                                          (mapcar #'$0 *pairs*)
                                                          10)
                                           0 1))
(defparameter *overfit-ys-batches* (subseq (build-batches *eng-encoder* *output-max-length*
                                                          (mapcar #'$1 *pairs*)
                                                          10)
                                           0 1))

(defclass seq2seq ()
  ((from-encoder :initform nil :accessor $seq2seq-from-encoder)
   (to-encoder :initform nil :accessor $seq2seq-to-encoder)
   (encoder-network :initform nil :accessor $seq2seq-encoder-network)
   (decoder-network :initform nil :accessor $seq2seq-decoder-network)))

(defun seq2seq (from-encoder to-encoder nhidden)
  (let ((n (make-instance 'seq2seq))
        (from-vsize (encoder-vocabulary-size from-encoder))
        (to-vsize (encoder-vocabulary-size to-encoder))
        (wvecsz 32))
    (with-slots (encoder-network decoder-network) n
      (setf ($seq2seq-from-encoder n) from-encoder
            ($seq2seq-to-encoder n) to-encoder)
      (setf encoder-network (sequential-layer
                             (recurrent-layer (affine-cell from-vsize wvecsz
                                                           :activation :nil
                                                           :biasp nil))
                             (recurrent-layer (lstm-cell wvecsz nhidden)))
            decoder-network (sequential-layer
                             (recurrent-layer (affine-cell to-vsize wvecsz
                                                           :activation :nil
                                                           :biasp nil))
                             (recurrent-layer (lstm-cell wvecsz nhidden))
                             (recurrent-layer
                              (sequential-layer
                               (parallel-layer
                                (attention-cell)
                                (functional-layer
                                 (lambda (q &key (trainp t))
                                  (declare (ignore trainp))
                                  q)))
                               (functional-layer
                                (lambda (c q &key (trainp t))
                                 (declare (ignore trainp))
                                 ($cat q c 1)))))
                             (recurrent-layer (affine-cell (* 2 nhidden) to-vsize
                                                           :activation :nil)))))
    n))

(defun $encoder-state (s2s)
  (with-slots (encoder-network) s2s
    ($cell-state ($cell ($ encoder-network 1)))))

(defun $update-decoder-network-state! (s2s hs h0)
  (with-slots (decoder-network) s2s
    ($update-cell-state! ($ decoder-network 1) h0)
    ($set-memory! ($ ($ ($cell ($ decoder-network 2)) 0) 0) (concat-sequence hs))))

(defun $execute-seq2seq (s2s xs ts)
  (with-slots (encoder-network decoder-network) s2s
    (let ((hs ($execute encoder-network xs))
          (h0 ($encoder-state s2s)))
      ($update-decoder-network-state! s2s hs h0)
      (with-keeping-state (decoder-network)
        (let* ((batch-size ($size (car xs) 0))
               (ys (append (list ($fill (tensor.long batch-size) 0))
                           (butlast ts))))
          ($execute decoder-network ys))))))

(defun $compute-loss (s2s xs ts)
  (let* ((ys ($execute-seq2seq s2s xs ts))
         (losses (mapcar (lambda (y c) ($cec y c)) ys ts)))
    ($div (apply #'$+ losses) ($count losses))))

(defun $generate-seq2seq (s2s hs h0 xs0 n)
  (with-slots (encoder-network decoder-network to-encoder) s2s
    (let ((sampled '())
          (xts xs0)
          (batch-size ($size (car xs0) 0)))
      ($update-decoder-network-state! s2s hs h0)
      (with-keeping-state (decoder-network)
        (loop :for i :from 0 :below n
              :do (let* ((yts ($evaluate decoder-network xts))
                         (rts (encoder-choose to-encoder yts -1)))
                    (push rts sampled)
                    (setf xts (encoder-encode to-encoder rts)))))
      (let ((res (reverse sampled))
            (results (make-list batch-size)))
        (loop :for r :in res
              :do (loop :for v :in r
                        :for i :from 0
                        :do (push (car v) ($ results i))))
        (mapcar #'reverse results)))))

(defun $evaluate-seq2seq (s2s xs &optional (n 9))
  (with-slots (encoder-network decoder-network) s2s
    (let ((hs ($evaluate encoder-network xs))
          (h0 ($encoder-state s2s)))
      ($generate-seq2seq s2s hs h0
                         (list ($fill (tensor.long ($size (car xs) 0)) 0))
                         n))))

(defun list-equal (l1 l2)
  (and (eq ($count l1) ($count l2))
       (loop :for i :in l1
             :for j :in l2
             :always (string-equal i j))))

(defun $matches-score (s2s ts ys)
  (let ((tss (encoder-decode ($seq2seq-to-encoder s2s) ts))
        (yss ys))
    (let ((matches (mapcar (lambda (tn yn) (if (list-equal tn yn) 0 1)) tss yss)))
      (* 1D0 (/ (reduce #'+ matches) ($count matches))))))

(defun gd! (s2s fn lr)
  (with-slots (encoder-network decoder-network) s2s
    (funcall fn decoder-network lr)
    (funcall fn encoder-network lr)))

(defun replace-eos (lists)
  (loop :for list :in lists
        :collect (loop :for e :in list
                       :collect (if (string-equal e "EOS")
                                    ""
                                    e))))

(defun $train (s2s xss tss &key (epochs 10) (pstep 100) (gdfn #'$adgd!) (lr 1D0) (testp T))
  (let ((sz ($count xss)))
    (with-slots (to-encoder) s2s
      (loop :for epoch :from 0 :below epochs
            :do (loop :for xsi :in xss
                      :for ts :in tss
                      :for idx :from 0
                      :for iter = (+ idx (* epoch sz))
                      :for xs = (reverse xsi)
                      :do (let ((loss ($compute-loss s2s xs ts)))
                            (gd! s2s gdfn lr)
                            (when (zerop (rem iter pstep))
                              (let* ((lv ($data loss))
                                     (tidx (random ($count *test-xs-batches*)))
                                     (txs (if testp
                                              (reverse ($ *test-xs-batches* tidx))
                                              xs))
                                     (tts (if testp
                                              ($ *test-ys-batches* tidx)
                                              ts))
                                     (ys ($evaluate-seq2seq s2s txs))
                                     (score ($matches-score s2s tts ys)))
                                (prn iter lv score)
                                (prn "TS" (replace-eos (encoder-decode to-encoder tts)))
                                (prn "YS" (replace-eos ys))
                                (prn "==")))))))))

(defmethod $reset! ((s2s seq2seq))
  (with-slots (encoder-network decoder-network) s2s
    ($reset! encoder-network)
    ($reset! decoder-network)))

(defparameter *s2s* (seq2seq *fra-encoder* *eng-encoder* 256))
($reset! *s2s*)

;; to check whether the code works
(time ($train *s2s* *overfit-xs-batches* *overfit-ys-batches* :epochs 500 :pstep 100 :testp nil))

;; real one
(time ($train *s2s* *train-xs-batches* *train-ys-batches* :epochs 1000 :pstep 100))

;; testing, checking
(prn (->> '(("tu" "es" "la" "professeur" "." "EOS" "EOS" "EOS" "EOS"))
          (encoder-encode *fra-encoder*)
          ($evaluate-seq2seq *s2s*)))
(prn (encoder-decode *fra-encoder* ($0 *overfit-xs-batches*)))
(prn (encoder-decode *eng-encoder* ($0 *overfit-ys-batches*)))
(prn ($evaluate-seq2seq *s2s* ($0 *overfit-xs-batches*)))
