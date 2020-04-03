;; following code to demonstrate the layer based rnn code
;; this network learns from simple single sentence and
;; generates characters based on it.

(defpackage :demo-genchars
  (:use #:common-lisp
        #:mu
        #:th
        #:th.text
        #:th.layers
        #:th.ex.data))

(in-package :demo-genchars)

;; text encoder
(defparameter *encoder*
  (character-encoder "the quick brown fox jumps over the lazy dog. 12345678901,!?"))

;; train data
(defparameter *data* (list "the quick brown fox jumps over the lazy dog. "
                           "quick brown fox jumps over the lazy dog. the "
                           "brown fox jumps over the lazy dog. the quick "
                           "fox jumps over the lazy dog. the quick brown "
                           "jumps over the lazy dog. the quick brown fox "
                           "over the lazy dog. the quick brown fox jumps "
                           "the lazy dog. the quick brown fox jumps over "))

;; train target
(defparameter *target* (mapcar (lambda (s) (rotate-left-string 1 s)) *data*))

;; network parameters
(defparameter *hidden-size* 100)

;; string generation function
;; encoder-choose function takes relative probabilities, make them normalized
;; probabilities and use it to generate new characters.
;; to use seed string induced state, stateful parameter is turned on.
;; after processing, it should be turned off for fresh restarting.
(defun generate-string (rnn encoder seedstr n &optional (temperature 1D0))
  ($generate-sequence rnn encoder seedstr n temperature))

;; network
;; the output of the network is relative probabilities of each characters and
;; it should be normalized or something to be interpreted as probabilities.
;; generally, softmax is used for it.
(defparameter *rnn* (let ((vsize (encoder-vocabulary-size *encoder*)))
                      (sequential-layer
                       (recurrent-layer (rnn-cell vsize *hidden-size*))
                       (recurrent-layer (affine-cell *hidden-size* vsize :activation :nil)))))

;; reset network
($reset! *rnn*)

;; train network - this is in fact make the network overfit the data.
;; for testing purpose, overfitting is good one :-P
(time
 (let* ((epochs 1000)
        (print-step 50)
        (xs (encoder-encode *encoder* *data*))
        (ts (encoder-encode *encoder* *target*)))
   (loop :for iter :from 0 :below epochs
         :do (let* ((outputs ($execute *rnn* xs))
                    (losses (mapcar (lambda (y c) ($cec y c)) outputs ts))
                    (loss ($div (apply #'$+ losses) ($count losses))))
               (when (zerop (rem iter print-step))
                 (prn iter ($data loss)))
               ($rmgd! *rnn*)))))

;; test trained network - high temperature means more "creativity".
;; if you increase temperature, then you'll know what it means.
(let ((seed-string "the")
      (gen-length 100)
      (temperature 1D0))
  (prn (generate-string *rnn* *encoder* seed-string gen-length temperature)))

;; lstm test
(defparameter *rnn-lstm* (let ((vsize (encoder-vocabulary-size *encoder*)))
                           (sequential-layer
                            (recurrent-layer (lstm-cell vsize *hidden-size*))
                            (recurrent-layer (affine-cell *hidden-size* vsize :activation :nil)))))

($reset! *rnn-lstm*)

(time
 (let* ((epochs 1000)
        (print-step 50)
        (xs (encoder-encode *encoder* *data*))
        (ts (encoder-encode *encoder* *target*)))
   (loop :for iter :from 0 :below epochs
         :do (let* ((outputs ($execute *rnn-lstm* xs))
                    (losses (mapcar (lambda (y c) ($cec y c)) outputs ts))
                    (loss ($div (apply #'$+ losses) ($count losses))))
               (when (zerop (rem iter print-step))
                 (prn iter ($data loss)))
               ($rmgd! *rnn-lstm*)))))

(let ((seed-string "the")
      (gen-length 100)
      (temperature 1D0))
  ($reset-state! *rnn-lstm* nil) ;; in case of being true of statefulp
  (prn (generate-string *rnn-lstm* *encoder* seed-string gen-length temperature)))

;; gru test
(defparameter *rnn-gru* (let ((vsize (encoder-vocabulary-size *encoder*)))
                          (sequential-layer
                           (recurrent-layer (gru-cell vsize *hidden-size*))
                           (recurrent-layer (affine-cell *hidden-size* vsize :activation :nil)))))

($reset! *rnn-gru*)

(time
 (let* ((epochs 1000)
        (print-step 50)
        (xs (encoder-encode *encoder* *data*))
        (ts (encoder-encode *encoder* *target*)))
   (loop :for iter :from 0 :below epochs
         :do (let* ((outputs ($execute *rnn-gru* xs))
                    (losses (mapcar (lambda (y c) ($cec y c)) outputs ts))
                    (loss ($div (apply #'$+ losses) ($count losses))))
               (when (zerop (rem iter print-step))
                 (prn iter ($data loss)))
               ($rmgd! *rnn-gru*)))))

(let ((seed-string "the")
      (gen-length 100)
      (temperature 1D0))
  (prn (generate-string *rnn-gru* *encoder* seed-string gen-length temperature)))
