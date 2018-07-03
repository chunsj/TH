;; from
;; https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

(defpackage :rnn-names
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :rnn-names)

;; classifying names
(defparameter *names* #{})
(loop :for fp :in (directory "/Users/Sungjin/Documents/MLStudy/PyTorch/names/*.txt")
      :for names = (read-lines-from fp)
      :for fn = (file-namestring fp)
      :do (let ((category (subseq fn 0 (- ($count fn) 4))))
            (setf ($ *names* category) (coerce names 'vector))))
(defparameter *categories* (hash-table-keys *names*))
(defparameter *category-size* ($count *categories*))
(defparameter *categories-index* #{})
(loop :for i :from 0 :below *category-size*
      :for category = ($ *categories* i)
      :do (setf ($ *categories-index* category) i))

(defun category-to-tensor (category)
  (let ((tensor (zeros 1 *category-size*)))
    (setf ($ tensor 0 ($ *categories-index* category)) 1)
    tensor))

(prn (category-to-tensor "Arabic"))

(defparameter *letters* "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,;' ")
(defparameter *letters-length* ($count *letters*))
(defparameter *letters-index* #{})
(loop :for i :from 0 :below *letters-length*
      :for c = ($ *letters* i)
      :do (setf ($ *letters-index* c) i))

(defun letter-to-tensor (ch)
  (let ((tensor (zeros 1 *letters-length*)))
    (setf ($ tensor 0 ($ *letters-index* ch (1- *letters-length*))) 1)
    tensor))

(defun name-to-tensor (name)
  (let ((tensor (zeros ($count name) *letters-length*)))
    (loop :for i :from 0 :below ($count name)
          :for c = ($ name i)
          :for idx = ($ *letters-index* c nil)
          :when idx
            :do (progn
                  (setf ($ tensor i idx) 1)))
    tensor))

(defparameter *hidden-size* 128)

(defparameter *wh* ($variable (rnd (+ *letters-length* *hidden-size*) *hidden-size*)))
(defparameter *bh* ($variable (zeros *hidden-size*)))
(defparameter *wo* ($variable (rnd *hidden-size* *category-size*)))
(defparameter *bo* ($variable (zeros *category-size*)))

(defun predict (tensor hidden-state)
  (let* ((input ($cat tensor hidden-state 1))
         (hidden ($xwpb input *wh* *bh*))
         (ho ($xwpb hidden *wo* *bo*))
         (output ($softmax ho)))
    (setf ($data hidden-state) ($data hidden))
    output))

(let* ((name ($0 ($ *names* "Korean")))
       (name-tensor (name-to-tensor name))
       (category ($constant (category-to-tensor "Korean")))
       (hidden ($constant (zeros 1 *hidden-size*)))
       (output (predict ($constant ($index name-tensor 0 0)) hidden)))
  (prn category output hidden)
  (prn ($cee output category)))

(let* ((name ($0 ($ *names* "Arabic")))
       (name-tensor (name-to-tensor name))
       (category ($constant (category-to-tensor "Arabic")))
       (hidden ($variable (zeros 1 *hidden-size*)))
       (output nil))
  (loop :for i :from 0 :below ($size name-tensor 0)
        :for x = ($constant ($index name-tensor 0 i))
        :do (setf output (predict x hidden)))
  (let ((loss ($cee output category)))
    (prn "L:" loss)
    ($bp! loss)
    ($gd! loss 0.005)))
