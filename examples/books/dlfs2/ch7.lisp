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

($reset! *encoder-rnn*)
($reset! *decoder-rnn*)

(prn ($last ($evaluate *encoder-rnn* (car *train-xs-batches*))))
(prn (encoder-encode *encoder* '("_")))

;; XXX can i write down what each line of code wants to do?
;;     more humane description would result more clean code.
;; XXX encoder could be a layer as well, right?
;;     more general archiving/unarchiving mechanism required.
(let ((h ($last ($evaluate *encoder-rnn* (car *train-xs-batches*))))
      (xt (tensor.long (loop :repeat *batch-size* :collect 11)))
      (res '())) ;; 11 is for "_"
  ($reset-state! *decoder-rnn* T)
  (with-slots (th.layers::cell) ($1 *decoder-rnn*)
    (with-slots (th.layers::ph) th.layers::cell
      (setf th.layers::ph h)))
  (let* ((out ($evaluate *decoder-rnn* (list xt)))
         (rt (encoder-choose *encoder* out -1)))
    (setf xt (encoder-encode *encoder* rt))
    (push rt res))
  (loop :for i :from 0 :below 3
        :do (let* ((out ($evaluate *decoder-rnn* xt))
                   (rt (encoder-choose *encoder* out -1)))
              (setf xt (encoder-encode *encoder* rt))
              (push rt res)))
  ($reset-state! *decoder-rnn* nil)
  (let ((res (reverse res))
        (results (make-list *batch-size*)))
    (loop :for r :in res
          :do (loop :for v :in r
                    :for i :from 0
                    :do (push v ($ results i))))
    (setf results (mapcar (lambda (rs) (apply #'concatenate 'string (reverse rs))) results))
    (print results)))

;; encoder should have more supportive methods for above implementation
;; softmax output to encoded input and vice versa
