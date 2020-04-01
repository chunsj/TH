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
