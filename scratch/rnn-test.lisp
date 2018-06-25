;; this is from
;; https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/
;; read the blog, this one is one of the most great blog for neural network.

(defpackage :rnn-trask
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :rnn-trask)

(defun dsigmoid (out) ($* out ($- 1 out)))

(defparameter *binary-dim* 8)
(defparameter *int2binary* #{})

(defun dec->bin (n)
  (multiple-value-bind (r m) (floor n 2)
    (if (= n 0) nil (append (dec->bin r) (list m)))))

(defun bin->dec (binary) (reduce (lambda (x y) (+ (* x 2) y)) binary))

(prn (dec->bin 100))
(prn (bin->dec (dec->bin 123)))

(let* ((largest-number (round (expt 2 *binary-dim*))))
  (loop :for i :from 0 :below largest-number
        :for bin = (dec->bin i)
        :for pad = (loop :for k :from 0 :below (- 8 ($count bin)) :collect 0)
        :do (setf ($ *int2binary* i) (tensor.byte (append pad bin)))))

(loop :for i :from 0 :below (round (expt 2 *binary-dim*))
      :do (prn i ($ *int2binary* i)))

(defparameter *alpha* 0.1)
(defparameter *input-dim* 2)
(defparameter *hidden-dim* 16)
(defparameter *output-dim* 1)

(defparameter *synapse0* ($- ($* 2 (rnd *input-dim* *hidden-dim*)) 1))
(defparameter *synapse1* ($- ($* 2 (rnd *hidden-dim* *output-dim*)) 1))
(defparameter *synapseh* ($- ($* 2 (rnd *hidden-dim* *hidden-dim*)) 1))

(defparameter *synapse0-update* ($zero *synapse0*))
(defparameter *synapse1-update* ($zero *synapse1*))
(defparameter *synapseh-update* ($zero *synapseh*))

(loop :for j :from 0 :below 10000
      :for half-largest-number = (round (expt 2 (1- *binary-dim*)))
      :for a-int = (random half-largest-number)
      :for a = ($ *int2binary* a-int)
      :for b-int = (random half-largest-number)
      :for b = ($ *int2binary* b-int)
      :for c-int = (+ a-int b-int)
      :for c = ($ *int2binary* c-int)
      :for d = ($zero c)
      :for overall-error = 0
      :for layer2-deltas = nil
      :for layer1-histories = (list (zeros 1 *hidden-dim*))
      :do (let ((future-layer1-delta (zeros 1 *hidden-dim*)))
            (loop :for position :from 0 :below *binary-dim*
                  :for x = (tensor (list (list ($ a (- *binary-dim* position 1))
                                               ($ b (- *binary-dim* position 1)))))
                  :for y = ($transpose (tensor (list (list ($ c (- *binary-dim* position 1))))))
                  :for z1 = ($add! ($mm x *synapse0*) ($mm (car layer1-histories) *synapseh*))
                  :for a1 = ($sigmoid z1)
                  :for z2 = ($mm a1 *synapse1*)
                  :for a2 = ($sigmoid z2)
                  :for l2e = ($- y a2)
                  :for l2d = ($* l2e (dsigmoid a2))
                  :do (progn
                        (push l2d layer2-deltas)
                        (incf overall-error (abs ($ l2e 0 0)))
                        (setf ($ d (- *binary-dim* position 1)) (round ($ a2 0 0)))
                        (push a1 layer1-histories)))
            ;; note that layer2-deltas and layer1-histories are in reverse order
            ;; which means most current one is at first
            (loop :for position :from 0 :below *binary-dim*
                  :for x = (tensor (list (list ($ a position) ($ b position))))
                  :for l1 = ($ layer1-histories position) ;; note
                  :for l1p = ($ layer1-histories (1+ position)) ;; note
                  :for l2d = ($ layer2-deltas position) ;; note
                  :for l1d = ($mul! ($add! ($mm future-layer1-delta ($transpose *synapseh*))
                                           ($mm l2d ($transpose *synapse1*)))
                                    (dsigmoid l1))
                  :for s1update = ($mm ($transpose l1) l2d)
                  :for shupdate = ($mm ($transpose l1p) l1d)
                  :for s0update = ($mm ($transpose x) l1d)
                  :do (progn
                        ($add! *synapse1-update* s1update)
                        ($add! *synapseh-update* shupdate)
                        ($add! *synapse0-update* s0update)
                        (setf future-layer1-delta l1d)))
            ($add! *synapse0* ($* *synapse0-update* *alpha*))
            ($add! *synapse1* ($* *synapse1-update* *alpha*))
            ($add! *synapseh* ($* *synapseh-update* *alpha*))
            ($zero! *synapse0-update*)
            ($zero! *synapse1-update*)
            ($zero! *synapseh-update*)
            (when (zerop (rem j 1000))
              (prn "ITR:" j)
              (prn "ERR:" overall-error)
              (prn "PRD:" d)
              (prn "TRU:" c)
              (prn a-int "+" b-int "=" (bin->dec ($list d)) "/" c-int)
              (gcf))))
