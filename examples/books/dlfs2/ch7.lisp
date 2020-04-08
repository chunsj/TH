(defpackage :dlfs2-ch7
  (:use #:common-lisp
        #:mu
        #:th
        #:th.ex.data
        #:th.layers
        #:th.text))

(in-package :dlfs2-ch7)

(defparameter *data* (addition))
(defparameter *data-length* ($count *data*))
(defparameter *encoder* (character-encoder "0123456789 _+-=."))

(defparameter *train-input-data* (mapcar (lambda (s) (subseq s 0 7)) (subseq *data* 0 40000)))
(defparameter *train-target-data* (mapcar (lambda (s) (subseq s 8)) (subseq *data* 0 40000)))
(defparameter *test-input-data* (mapcar (lambda (s) (subseq s 0 7)) (subseq *data* 40000)))
(defparameter *test-target-data* (mapcar (lambda (s) (subseq s 8)) (subseq *data* 40000)))

(defparameter *batch-size* 100)
(defparameter *hidden-size* 128)
(defparameter *wvec-size* 16)

(defun build-batches (data n)
  (loop :for tail :on data :by (lambda (l) (nthcdr n l))
        :collect (encoder-encode *encoder* (subseq tail 0 (min ($count tail) n)))))

(defparameter *train-xs-batches* (build-batches *train-input-data* *batch-size*))
(defparameter *train-ys-batches* (build-batches *train-target-data* *batch-size*))

(defparameter *overfit-xs-batches* (subseq (build-batches *train-input-data* 3) 0 3))
(defparameter *overfit-ys-batches* (subseq (build-batches *train-target-data* 3) 0 3))

(defun generate-string (rnn encoder seedstr n &optional (temperature 1D0))
  ($generate-sequence rnn encoder seedstr n temperature))

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

(defun execute-seq2seq (encoder-rnn decoder-rnn encoder xs &optional (n 3))
  ($execute encoder-rnn xs)
  (let ((h0 ($cell-state encoder-rnn)))
    ($reset-state! decoder-rnn T)
    ($update-cell-state! decoder-rnn h0)
    (let* ((batch-size ($size (car xs) 0))
           (xts (encoder-encode encoder (loop :repeat batch-size :collect "_")))
           (yts ($execute decoder-rnn xts))
           (rts (encoder-choose encoder yts -1))
           (res '()))
      (push (car yts) res)
      (setf xts (encoder-encode encoder rts))
      (loop :for i :from 0 :below n
            :do (let* ((yts ($execute decoder-rnn xts))
                       (rts (encoder-choose encoder yts -1)))
                  (push (car yts) res)
                  (setf xts (encoder-encode encoder rts))))
      ($reset-state! decoder-rnn nil)
      (reverse res))))

(defun loss-seq2seq (encoder-rnn decoder-rnn encoder xs ts &optional (n 3))
  (let* ((ys (execute-seq2seq encoder-rnn decoder-rnn encoder xs n))
         (losses (mapcar (lambda (y c) ($cec y c)) ys ts))
         (loss ($div (apply #'$+ losses) ($count losses))))
    loss))

(defun evaluate-seq2seq (encoder-rnn decoder-rnn encoder xs &optional (n 3))
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

(defun matches-score (encoder ts ys)
  (let ((tss (->> ts
                  (encoder-decode encoder)
                  (mapcar (lambda (s) (parse-integer s)))))
        (yss (->> ys
                  (mapcar (lambda (s)
                            (handler-case (parse-integer s)
                              (error (c)
                                (prn "NOT A NUMBER:" c)
                                -1)))))))
    (let ((matches (mapcar (lambda (tn yn) (if (eq tn yn) 0 1)) tss yss)))
      (* 1D0 (/ (reduce #'+ matches) ($count matches))))))

($reset! *encoder-rnn*)
($reset! *decoder-rnn*)

;; real training
(time
 (let ((epochs 30)
       (pstep 100))
   (loop :for epoch :from 0 :below epochs
         :do (loop :for xs :in *train-xs-batches*
                   :for ts :in *train-ys-batches*
                   :for iter :from 0
                   :do (let ((loss (loss-seq2seq *encoder-rnn* *decoder-rnn* *encoder* xs ts)))
                         ($rmgd! *encoder-rnn*)
                         ($rmgd! *decoder-rnn*)
                         (when (zerop (rem iter pstep))
                           (prn epoch iter ($data loss))
                           (prn "  "
                                (matches-score *encoder*
                                               ts
                                               (evaluate-seq2seq *encoder-rnn* *decoder-rnn*
                                                                 *encoder* xs)))))))))

(matches-score *encoder* ($0 *train-ys-batches*)
               (evaluate-seq2seq *encoder-rnn* *decoder-rnn* *encoder* ($0 *train-xs-batches*)))

(prn (encoder-decode *encoder* ($0 *train-ys-batches*)))
(prn (evaluate-seq2seq *encoder-rnn* *decoder-rnn* *encoder* ($0 *train-xs-batches*)))

;; overfitting test
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

(time
 (let ((epochs 10000)
       (pstep 500))
   (loop :for epoch :from 0 :below epochs
         :do (loop :for xs :in *overfit-xs-batches*
                   :for ts :in *overfit-ys-batches*
                   :for iter :from 0
                   :do (let ((loss (loss-seq2seq *encoder-rnn* *decoder-rnn* *encoder*
                                                 (reverse xs) ts)))
                         ($rmgd! *encoder-rnn*)
                         ($rmgd! *decoder-rnn*)
                         (when (and (zerop (rem epoch pstep)) (eq iter 0))
                           (prn epoch iter ($data loss))
                           (prn "  "
                                (matches-score *encoder*
                                               ts
                                               (evaluate-seq2seq *encoder-rnn* *decoder-rnn*
                                                                 *encoder* xs 1)))))))))

(prn (encoder-decode *encoder* ($0 *overfit-ys-batches*)))
(prn (evaluate-seq2seq *encoder-rnn* *decoder-rnn* *encoder* ($0 *overfit-xs-batches*) 1))
