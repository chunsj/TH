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
