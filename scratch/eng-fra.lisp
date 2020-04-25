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

(defparameter *train-xs-batches* (build-batches *fra-encoder* *input-max-length*
                                                (mapcar #'$0 *pairs*)
                                                *batch-size*))
(defparameter *train-ys-batches* (build-batches *eng-encoder* *output-max-length*
                                                (mapcar #'$1 *pairs*)
                                                *batch-size*))
