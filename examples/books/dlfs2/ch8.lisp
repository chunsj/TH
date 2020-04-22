(defpackage :dlfs2-ch8
  (:use #:common-lisp
        #:mu
        #:th
        #:th.ex.data
        #:th.layers
        #:th.text))

(in-package :dlfs2-ch8)

;; data for the chapter 8 example
;;
;; number addition problems
(defparameter *data* (date-data))
(defparameter *data-length* ($count *data*))
(defparameter *encoder* (character-encoder (concatenate 'string "01234567890"
                                                        "abcdefghijklmnopqrstuvwxyz"
                                                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                                        " _-,/")))

;; train and test datasets
(defparameter *train-input-data* (mapcar (lambda (s) (subseq s 0 29)) (subseq *data* 0 40000)))
(defparameter *train-target-data* (mapcar (lambda (s) (subseq s 30)) (subseq *data* 0 40000)))
(defparameter *test-input-data* (mapcar (lambda (s) (subseq s 0 29)) (subseq *data* 40000)))
(defparameter *test-target-data* (mapcar (lambda (s) (subseq s 30)) (subseq *data* 40000)))

;; network parameters
(defparameter *batch-size* 100)
(defparameter *hidden-size* 256)
(defparameter *wvec-size* 16)

;; preparing datasets - an helper function
(defun build-batches (data n)
  (loop :for tail :on data :by (lambda (l) (nthcdr n l))
        :collect (encoder-encode *encoder* (subseq tail 0 (min ($count tail) n)))))

;; for real training
(defparameter *train-xs-batches* (build-batches *train-input-data* *batch-size*))
(defparameter *train-ys-batches* (build-batches *train-target-data* *batch-size*))

;; for overfitting - to check implementation
(defparameter *overfit-xs-batches* (subseq (build-batches *train-input-data* 5) 0 1))
(defparameter *overfit-ys-batches* (subseq (build-batches *train-target-data* 5) 0 1))

;; helper functions for the seq2seq model
;; mostly generation, execution(for training) and evaluation(for running)

;; compute attention context - dot product attention
(defun compute-context (hs q)
  "computes attention context from hs(TxBxD) and q(BxD)"
  (let* ((d ($size q 1))
         (q (-> (apply #'$reshape q (cons 1 ($size q)))
                ($transpose 0 1)))
         (k ($transpose hs 0 1))
         (kt ($transpose k 1 2))
         (qkt ($div ($bmm q kt) ($sqrt d)))
         (a (-> ($softmax ($reshape qkt ($size qkt 0) ($size qkt 2)))
                ($reshape ($size qkt 0) 1 ($size qkt 2))))
         (ctx (-> ($bmm a k)
                  ($reshape ($size k 0) ($size k 2)))))
    ctx))

;; generate a string using the seed string
(defun generate-string (rnn encoder seedstr n &optional (temperature 1D0))
  ($generate-sequence rnn encoder seedstr n temperature))

(defun update-decoder-state! (decoder-rnn h) ($update-cell-state! ($ decoder-rnn 1) h))
(defun update-attention-memory! (decoder-rnn hs)
  ($set-memory! ($ ($ ($cell ($ decoder-rnn 2)) 0) 0) (concat-sequence hs)))

;; execution function for training
(defun execute-seq2seq (encoder-rnn decoder-rnn encoder xs ts)
  (let* ((hs ($execute encoder-rnn xs))
         (h0 ($last hs)))
    (update-decoder-state! decoder-rnn h0)
    (update-attention-memory! decoder-rnn hs)
    (with-keeping-state (decoder-rnn)
      (let* ((batch-size ($size (car xs) 0))
             (ys (append (encoder-encode encoder (loop :repeat batch-size :collect "_"))
                         ts))
             (yts ($execute decoder-rnn ys)))
        (butlast yts)))))

;; loss function using cross entropy
(defun loss-seq2seq (encoder-rnn decoder-rnn encoder xs ts &optional verbose)
  (let* ((ys (execute-seq2seq encoder-rnn decoder-rnn encoder xs ts))
         (losses (mapcar (lambda (y c) ($cec y c)) ys ts))
         (loss ($div (apply #'$+ losses) ($count losses))))
    (when verbose
      (prn "TS" (encoder-decode encoder ts))
      (prn "YS" (encoder-choose encoder ys -1)))
    loss))

;; generate using decoder
(defun generate-decoder (decoder-rnn encoder hs xs0 n)
  (let ((sampled '())
        (h ($last hs))
        (xts xs0)
        (batch-size ($size (car xs0) 0)))
    (update-decoder-state! decoder-rnn h)
    (update-attention-memory! decoder-rnn hs)
    (with-keeping-state (decoder-rnn)
      (loop :for i :from 0 :below n
            :do (let* ((yts ($evaluate decoder-rnn xts))
                       (rts (encoder-choose encoder yts -1)))
                  (push rts sampled)
                  (setf xts (encoder-encode encoder rts)))))
    (let ((res (reverse sampled))
          (results (make-list batch-size)))
      (loop :for r :in res
            :do (loop :for v :in r
                      :for i :from 0
                      :do (push v ($ results i))))
      (mapcar (lambda (rs) (apply #'concatenate 'string (reverse rs))) results))))

;; running the model
(defun evaluate-seq2seq (encoder-rnn decoder-rnn encoder xs &optional (n 10))
  (let ((hs ($evaluate encoder-rnn xs)))
    (generate-decoder decoder-rnn encoder hs
                      (encoder-encode encoder (loop :repeat ($size (car xs) 0) :collect "_"))
                      n)))

;; compare the results - between the generated one and the truth
(defun matches-score (encoder ts ys)
  (let ((tss (encoder-decode encoder ts))
        (yss ys))
    (let ((matches (mapcar (lambda (tn yn) (if (string-equal tn yn) 0 1)) tss yss)))
      (* 1D0 (/ (reduce #'+ matches) ($count matches))))))

;; train seq2seq network
(defun train-seq2seq (encoder-rnn decoder-rnn encoder xss tss epochs pstep)
  (let ((sz ($count xss)))
    (loop :for epoch :from 0 :below epochs
          :do (loop :for xs :in xss
                    :for ts :in tss
                    :for idx :from 0
                    :for iter = (+ idx (* epoch sz))
                    :do (let ((loss (loss-seq2seq encoder-rnn decoder-rnn encoder xs ts)))
                          ($rmgd! decoder-rnn)
                          ($rmgd! encoder-rnn)
                          (when (zerop (rem iter pstep))
                            (let* ((lv ($data loss))
                                   (ys (evaluate-seq2seq encoder-rnn decoder-rnn encoder xs))
                                   (score (matches-score encoder ts ys)))
                              (prn iter lv score)
                              (prn "TS" (encoder-decode encoder ts))
                              (prn "YS" ys))))))))

;; model
(defparameter *encoder-rnn* (let ((vsize (encoder-vocabulary-size *encoder*)))
                              (sequential-layer
                               (recurrent-layer (affine-cell vsize *wvec-size*
                                                             :activation :nil
                                                             :biasp nil))
                               (recurrent-layer (lstm-cell *wvec-size* *hidden-size*)))))

(defparameter *decoder-rnn* (let ((vsize (encoder-vocabulary-size *encoder*)))
                              (sequential-layer
                               (recurrent-layer (affine-cell vsize *wvec-size*
                                                             :activation :nil
                                                             :biasp nil))
                               (recurrent-layer (lstm-cell *wvec-size* *hidden-size*))
                               (recurrent-layer
                                (sequential-layer
                                 (parallel-layer (dot-product-attention-cell)
                                                 (functional-layer
                                                  (lambda (q &key (trainp t))
                                                   (declare (ignore trainp))
                                                   q)))
                                 (functional-layer
                                  (lambda (c q &key (trainp t))
                                   (declare (ignore trainp))
                                   ($cat q c 1)))))
                               (recurrent-layer (affine-cell (* 2 *hidden-size*) vsize
                                                             :activation :nil)))))

($reset! *encoder-rnn*)
($reset! *decoder-rnn*)

;; overfitting for checking implementation
(time (train-seq2seq *encoder-rnn* *decoder-rnn* *encoder*
                     *overfit-xs-batches* *overfit-ys-batches*
                     10000 100))

(prn (car *overfit-xs-batches*))

(prn (encoder-decode *encoder* ($0 *overfit-ys-batches*)))
(prn (evaluate-seq2seq *encoder-rnn* *decoder-rnn* *encoder* ($0 *overfit-xs-batches*)))

;; real training
(time (train-seq2seq *encoder-rnn* *decoder-rnn* *encoder*
                     *train-xs-batches* *train-ys-batches*
                     30 100))

(matches-score *encoder* ($0 *train-ys-batches*)
               (evaluate-seq2seq *encoder-rnn* *decoder-rnn* *encoder* ($0 *train-xs-batches*)))

(prn (encoder-decode *encoder* ($0 *train-ys-batches*)))
(prn (evaluate-seq2seq *encoder-rnn* *decoder-rnn* *encoder* ($0 *train-xs-batches*)))
