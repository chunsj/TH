(defpackage :dlfs2-ch7
  (:use #:common-lisp
        #:mu
        #:th
        #:th.ex.data
        #:th.layers
        #:th.text))

(in-package :dlfs2-ch7)

;; data for the chapter 7 example
;;
;; number addition problems
(defparameter *data* (addition))
(defparameter *data-length* ($count *data*))
(defparameter *encoder* (character-encoder "0123456789 _+-=."))

;; train and test datasets
(defparameter *train-input-data* (mapcar (lambda (s) (subseq s 0 7)) (subseq *data* 0 40000)))
(defparameter *train-target-data* (mapcar (lambda (s) (subseq s 8)) (subseq *data* 0 40000)))
(defparameter *test-input-data* (mapcar (lambda (s) (subseq s 0 7)) (subseq *data* 40000)))
(defparameter *test-target-data* (mapcar (lambda (s) (subseq s 8)) (subseq *data* 40000)))

;; network parameters
(defparameter *batch-size* 100)
(defparameter *hidden-size* 128)
(defparameter *wvec-size* 16)

;; preparing datasets - an helper function
(defun build-batches (data n)
  (loop :for tail :on data :by (lambda (l) (nthcdr n l))
        :collect (encoder-encode *encoder* (subseq tail 0 (min ($count tail) n)))))

;; for real training
(defparameter *train-xs-batches* (build-batches *train-input-data* *batch-size*))
(defparameter *train-ys-batches* (build-batches *train-target-data* *batch-size*))

;; for overfitting - to check implementation
(defparameter *overfit-xs-batches* (subseq (build-batches *train-input-data* 1) 0 1))
(defparameter *overfit-ys-batches* (subseq (build-batches *train-target-data* 1) 0 1))

;; helper functions for the seq2seq model
;; mostly generation, execution(for training) and evaluation(for running)

;; generate a string using the seed string
(defun generate-string (rnn encoder seedstr n &optional (temperature 1D0))
  ($generate-sequence rnn encoder seedstr n temperature))

;; execution function for training - current implementation is wrong XXX
(defun execute-seq2seq (encoder-rnn decoder-rnn encoder xs ts)
  ($execute encoder-rnn xs)
  (let ((h0 ($cell-state encoder-rnn)))
    ($reset-state! decoder-rnn T)
    ($update-cell-state! decoder-rnn h0)
    (let* ((batch-size ($size (car xs) 0))
           (ys (append (encoder-encode encoder (loop :repeat batch-size :collect "_"))
                       ts))
           (yts ($execute decoder-rnn ys)))
      ($reset-state! decoder-rnn nil)
      (butlast yts))))

;; loss function using cross entropy
(defun loss-seq2seq (encoder-rnn decoder-rnn encoder xs ts)
  (let* ((ys (execute-seq2seq encoder-rnn decoder-rnn encoder xs ts))
         (losses (mapcar (lambda (y c) ($cec y c)) ys ts))
         (loss ($div (apply #'$+ losses) ($count losses))))
    (prn "TS" ts)
    (prn "TS" (encoder-decode encoder ts))
    (prn "YS" (encoder-choose encoder ys -1))
    loss))

;; running the model
(defun evaluate-seq2seq (encoder-rnn decoder-rnn encoder xs &optional (n 3))
  ;; xxx here, something wrong, i believe
  ($evaluate encoder-rnn xs)
  (let ((h0 ($cell-state encoder-rnn)))
    ($reset-state! decoder-rnn T)
    ($update-cell-state! decoder-rnn h0)
    (let* ((batch-size ($size (car xs) 0))
           (xts (encoder-encode encoder (loop :repeat batch-size :collect "_")))
           (yts ($evaluate decoder-rnn xts))
           (rts (encoder-choose encoder yts -1))
           (res '()))
      (push rts res)
      (setf xts (encoder-encode encoder rts))
      (loop :for i :from 0 :below n
            :do (let* ((yts ($evaluate decoder-rnn xts))
                       (rts (encoder-choose encoder yts -1)))
                  (push rts res)
                  (setf xts (encoder-encode encoder rts))))
      ($reset-state! decoder-rnn nil)
      (let ((res (reverse res))
            (results (make-list batch-size)))
        (loop :for r :in res
              :do (loop :for v :in r
                        :for i :from 0
                        :do (push v ($ results i))))
        (mapcar (lambda (rs) (apply #'concatenate 'string (reverse rs))) results)))))

;; compare the results - between the generated one and the truth
(defun matches-score (encoder ts ys)
  (let ((tss (->> ts
                  (encoder-decode encoder)
                  (mapcar (lambda (s) (parse-integer s)))))
        (yss (->> ys
                  (mapcar (lambda (s)
                            (handler-case (parse-integer s)
                              (error (c)
                                (declare (ignore c))
                                -1)))))))
    (let ((matches (mapcar (lambda (tn yn) (if (eq tn yn) 0 1)) tss yss)))
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
                              (prn epoch iter lv score)
                              (prn "TS" (encoder-decode encoder ts))
                              (prn "YS" ys))))))))

;; overfitting - testing the implementation
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
                               (recurrent-layer (affine-cell *hidden-size* vsize
                                                             :activation :nil)))))

($reset! *encoder-rnn*)
($reset! *decoder-rnn*)

(progn
  (prn (loss-seq2seq *encoder-rnn* *decoder-rnn* *encoder*
                     (car *overfit-xs-batches*)
                     (car *overfit-ys-batches*)))
  ($cg! *decoder-rnn*)
  ($cg! *encoder-rnn*))

;; overfitting
(time (train-seq2seq *encoder-rnn* *decoder-rnn* *encoder*
                     *overfit-xs-batches* *overfit-ys-batches*
                     1000 200))

(prn (encoder-decode *encoder* ($0 *overfit-ys-batches*)))
(prn (evaluate-seq2seq *encoder-rnn* *decoder-rnn* *encoder* ($0 *overfit-xs-batches*) 1))

;; the real model
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
                               (recurrent-layer (affine-cell *hidden-size* vsize
                                                             :activation :nil)))))

($reset! *encoder-rnn*)
($reset! *decoder-rnn*)

;; real training
(time (train-seq2seq *encoder-rnn* *decoder-rnn* *encoder*
                     *train-xs-batches* *train-ys-batches*
                     30 100))

(matches-score *encoder* ($0 *train-ys-batches*)
               (evaluate-seq2seq *encoder-rnn* *decoder-rnn* *encoder* ($0 *train-xs-batches*)))

(prn (encoder-decode *encoder* ($0 *train-ys-batches*)))
(prn (evaluate-seq2seq *encoder-rnn* *decoder-rnn* *encoder* ($0 *train-xs-batches*)))
