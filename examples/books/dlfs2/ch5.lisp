(defpackage :dlfs2-ch5
  (:use #:common-lisp
        #:mu
        #:th
        #:th.ex.data
        #:th.layers
        #:th.text))

(in-package :dlfs2-ch5)

(defparameter *encoder* (word-encoder (loop :for line :in (ptb :train)
                                            :append (->> (strim line)
                                                         (split #\space)))))

(prn (encoder-encode *encoder* '(("hello" "world"))))
(prn (encoder-decode *encoder* (encoder-encode *encoder* '(("this" "world")))))
