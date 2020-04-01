(defpackage :dlfs2-ch5
  (:use #:common-lisp
        #:mu
        #:th
        #:th.ex.data
        #:th.layers
        #:th.text))

(in-package :dlfs2-ch5)

(defparameter *data* (loop :for line :in (ptb :train)
                           :append (->> (strim line)
                                        (split #\space))))
(defparameter *encoder* (word-encoder *data*))

;; simple encoding tests
(prn (encoder-encode *encoder* '(("hello" "world"))))
(prn (encoder-decode *encoder* (encoder-encode *encoder* '(("this" "world")))))

;; encoding tests from encoded vocabularies
(prn (encoder-vocabularies *encoder*))
(prn (subseq *data* 27 127))
(prn (encoder-encode *encoder* (list (subseq *data* 27 127))))
(prn (encoder-decode *encoder* (encoder-encode *encoder* (list (subseq *data* 27 127)))))

(defparameter *xs-data* (loop :for i :from 0 :below 10
                              :for idx = (+ i 27)
                              :collect (subseq *data* idx (+ idx 40))))
(defparameter *ys-data* (loop :for i :from 0 :below 10
                              :for idx = (+ i 28)
                              :collect (subseq *data* idx (+ idx 40))))
(defparameter *xs* (encoder-encode *encoder* *xs-data*))
(defparameter *ys* (encoder-encode *encoder* *ys-data*))

(defparameter *hidden-size* 100)

(defparameter *rnn* (let ((vsize (encoder-vocabulary-size *encoder*)))
                      (sequential-layer
                       (recurrent-layer (rnn-cell vsize *hidden-size*))
                       (recurrent-layer (affine-cell *hidden-size* vsize :activation :nil)))))

;; reset network
($reset! *rnn*)

;; train network
(time
 (with-foreign-memory-limit (4096) ;; for speed
   (let* ((epochs 1000)
          (print-step 10))
     (loop :for iter :from 0 :below epochs
           :do (let* ((outputs ($execute *rnn* *xs*))
                      (losses (mapcar (lambda (y c) ($cec y c)) outputs *ys*))
                      (loss ($div (apply #'$+ losses) ($count losses))))
                 (when (zerop (rem iter print-step))
                   (prn iter ($data loss)))
                 ($rmgd! *rnn*))))))

($reset-state! *rnn* nil)
(prn ($generate-sequence *rnn* *encoder* '("N" "years" "old" "will") 20))
