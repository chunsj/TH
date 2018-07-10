(defpackage :binary-add
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :binary-add)

(defparameter *binary-dim* 8)
(defparameter *int2binary* #{})

(defun dec->bin (n)
  (multiple-value-bind (r m) (floor n 2)
    (if (= n 0) nil (append (dec->bin r) (list m)))))

(defun bin->dec (binary) (reduce (lambda (x y) (+ (* x 2) y)) binary))

;; binary to decimal and decimal to binary number
;; functions are for generating data
(prn (dec->bin 100))
(prn (bin->dec (dec->bin 123)))

;; for efficiency we will build map for dec-bin
(let* ((largest-number (round (expt 2 *binary-dim*))))
  (loop :for i :from 0 :below largest-number
        :for bin = (dec->bin i)
        :for pad = (loop :for k :from 0 :below (- 8 ($count bin)) :collect 0)
        :do (setf ($ *int2binary* i) (tensor.byte (append pad bin)))))

(defparameter *alpha* 0.1)
(defparameter *input-dim* 2)
(defparameter *hidden-dim* 16)
(defparameter *output-dim* 1)
(defparameter *iterations* 10000)

(defparameter *synapse0* ($variable ($- ($* 2 (rnd *input-dim* *hidden-dim*)) 1)))
(defparameter *synapse1* ($variable ($- ($* 2 (rnd *hidden-dim* *output-dim*)) 1)))
(defparameter *synapseh* ($variable ($- ($* 2 (rnd *hidden-dim* *hidden-dim*)) 1)))

(loop :for j :from 0 :below *iterations*
      :for half-largest-number = (round (expt 2 (1- *binary-dim*)))
      :for a-int = (random half-largest-number)
      :for a = ($ *int2binary* a-int)
      :for b-int = (random half-largest-number)
      :for b = ($ *int2binary* b-int)
      :for c-int = (+ a-int b-int)
      :for c = ($ *int2binary* c-int)
      :do (let ((d ($zero c))
                (overall-error 0)
                (losses nil)
                (ps ($constant (zeros 1 *hidden-dim*))))
            ;; forward propagation
            ;; we start with the least significant bit or the right most bit
            (loop :for position :from (1- *binary-dim*) :downto 0
                  :for x = ($constant (tensor (list (list ($ a position) ($ b position)))))
                  :for z1 = ($add ($mm x *synapse0*) ($mm ps *synapseh*))
                  :for a1 = ($sigmoid z1)
                  :for z2 = ($mm a1 *synapse1*)
                  :for a2 = ($sigmoid z2)
                  :for y = ($constant ($transpose (tensor (list (list ($ c position))))))
                  :for l2e = ($- y a2) ;; cf. if reverted, then, update should be subtracted
                  :for l = ($expt l2e 2)
                  :do (progn
                        (setf ps a1)
                        (push l losses)
                        (incf overall-error (abs ($data ($ l2e 0 0))))
                        (setf ($ d position) (round ($data ($ a2 0 0))))))
            ;; of course, bptt will take losses in given order, so losses should be in reverse order.
            ;; that's why we use push.
            ($bptt! losses)
            ($gd! ($0 losses) *alpha*)
            (when (zerop (rem j 1000))
              (prn "ITR:" j "ERR: " overall-error)
              (prn "PRD:" d)
              (prn "TRU:" c)
              (prn a-int "+" b-int "=" (bin->dec ($list d)) "/" c-int)
              (gcf))))
