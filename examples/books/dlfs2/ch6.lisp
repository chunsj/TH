(defpackage :dlfs2-ch6
  (:use #:common-lisp
        #:mu
        #:th
        #:th.ex.data
        #:th.layers
        #:th.text))

(in-package :dlfs2-ch6)

(defparameter *simple-data* '("you" "say" "goodbye" "I" "say" "hello" "."))
(defparameter *encoder* (word-encoder *simple-data*))

(prn (encoder-encode *encoder* '(("hello" "goodbye"))))
(prn (encoder-decode *encoder* (encoder-encode *encoder* '(("hello" "goodbye")))))

(defparameter *xs-data* '(("you" "say" "goodbye" "I" "say" "hello")
                          ("say" "goodbye" "I" "say" "hello" ".")
                          ("goodbye" "I" "say" "hello" "." "you")
                          ("I" "say" "hello" "." "you" "say")))
(defparameter *ys-data* '(("say" "goodbye" "I" "say" "hello" ".")
                          ("goodbye" "I" "say" "hello" "." "you")
                          ("I" "say" "hello" "." "you" "say")
                          ("say" "hello" "." "you" "say" "goodbye")))

(defparameter *xs* (encoder-encode *encoder* *xs-data*))
(defparameter *ys* (encoder-encode *encoder* *ys-data*))

(defparameter *hidden-size* 50)

(defparameter *rnn* (let ((vsize (encoder-vocabulary-size *encoder*)))
                      (sequential-layer
                       (recurrent-layer (lstm-cell vsize *hidden-size*))
                       (recurrent-layer (affine-cell *hidden-size* vsize :activation :nil)))))

;; reset network
($reset! *rnn*)

;; train network
(time
 (with-foreign-memory-limit (4096) ;; for speed
   (let* ((epochs 1000)
          (print-step 100))
     (loop :for iter :from 0 :below epochs
           :do (let* ((outputs ($execute *rnn* *xs*))
                      (losses (mapcar (lambda (y c) ($cec y c)) outputs *ys*))
                      (loss ($div (apply #'$+ losses) ($count losses))))
                 (when (zerop (rem iter print-step))
                   (prn iter ($data loss)))
                 ($rmgd! *rnn*))))))

($reset-state! *rnn* nil)
(prn ($generate-sequence *rnn* *encoder* '("you" "say") 10))


;; more complex data
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
                       (recurrent-layer (lstm-cell vsize *hidden-size*))
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

;; XXX maybe adding more lstm layers could be possible.
;; XXX dropout could be applied as well; but dropout-cell should be implemented to do this.
