;; from
;; http://karpathy.github.io/2015/05/21/rnn-effectiveness/
;;
;; this is for more 'practical' demo of layer based rnn.

(defpackage :demo-genchars2
  (:use #:common-lisp
        #:mu
        #:th
        #:th.text
        #:th.layers
        #:th.ex.data))

(in-package :demo-genchars2)

;; text data from collected text of paul graham.
(defparameter *data* (format nil "~{~A~^~%~}"
                             (remove-if (lambda (line) (< ($count line) 1)) (text-lines :pg))))
(defparameter *encoder* (character-encoder *data*))


;; training data
(defparameter *sequence-length* 50)
(defparameter *batch-size* 10)
(defparameter *data-size* ($count *data*))
(defparameter *upto* (- *data-size* *sequence-length* 1))
(defparameter *input-strings* (loop :for p :from 0 :below *upto* :by *sequence-length*
                                    :collect (subseq *data* p (+ p *sequence-length*))))
(defparameter *target-strings* (loop :for p :from 0 :below *upto* :by *sequence-length*
                                     :collect (subseq *data* (1+ p) (+ p *sequence-length* 1))))
(defparameter *input-size* ($count *input-strings*))

(defun partition (list cell-size)
  (loop :for cell :on list :by (lambda (list) (nthcdr cell-size list))
        :collecting (subseq cell 0 cell-size)))

(defparameter *inputs* (mapcar (lambda (strings) (encoder-encode *encoder* strings))
                               (partition *input-strings* 10)))
(defparameter *targets* (mapcar (lambda (strings) (encoder-encode *encoder* strings))
                                (partition *target-strings* 10)))
(defparameter *batch-count* ($count *inputs*))

;; network parameters
(defparameter *hidden-size* 100)

;; network
(defparameter *rnn* (let ((vsize (encoder-vocabulary-size *encoder*)))
                      (sequential-layer
                       (recurrent-layer (embedding-cell vsize *hidden-size*))
                       (recurrent-layer (affine-cell *hidden-size* vsize :activation :nil)))))

;; string generation function
;; encoder-choose function takes relative probabilities, make them normalized
;; probabilities and use it to generate new characters.
;; to use seed string induced state, stateful parameter is turned on.
;; after processing, it should be turned off for fresh restarting.
(defun generate-string (rnn encoder seedstr n &optional (temperature 1D0))
  (let* ((seedps ($evaluate rnn (encoder-encode encoder (list seedstr))))
         (seedstrs (encoder-choose encoder seedps temperature))
         (laststrs (list (string ($last (car seedstrs)))))
         (resultstr (concatenate 'string seedstr (car laststrs))))
    ($set-stateful rnn T)
    (loop :for i :from 0 :below n
          :for nextseq = (encoder-encode encoder laststrs)
          :for nextoutps = ($evaluate rnn nextseq)
          :for nextoutstrs = (encoder-choose encoder nextoutps temperature)
          :do (progn
                (setf laststrs nextoutstrs)
                (setf resultstr (concatenate 'string resultstr (car nextoutstrs)))))
    ($set-stateful rnn nil)
    resultstr))

;; reset network
($reset! *rnn*)

;; train parameters
(defparameter *epochs* 1000)

;; train network - this is in fact make the network overfit the data.
;; for testing purpose, overfitting is good one :-P
(time
 (with-foreign-memory-limit (32768) ;; for speed
   (let* ((epochs *epochs*)
          (print-step 50))
     (loop :for epoch :from 0 :below epochs
           :do (loop :for xs :in *inputs* :for ts :in *targets*
                     :for iter :from 0
                     :do (let* ((outputs ($execute *rnn* xs))
                                (losses (mapcar (lambda (y c) ($cec y c)) outputs ts))
                                (loss ($div (apply #'$+ losses) ($count losses))))
                           (when (zerop (rem iter print-step))
                             (prn epoch "|" iter ($data loss)))
                           ($rmgd! *rnn*)))))))

;; test trained network - high temperature means more "creativity".
;; if you increase temperature, then you'll know what it means.
(let ((seed-string "the")
      (gen-length 100)
      (temperature 1D0))
  (prn (generate-string *rnn* *encoder* seed-string gen-length temperature)))
