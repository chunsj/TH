(defpackage :dlfs-ch3
  (:use #:common-lisp
        #:mu
        #:th
        #:th.ex.data
        #:th.text))

(in-package :dlfs-ch3)

(let* ((c (tensor '((1 0 0 0 0 0 1))))
       (c2 ($select ($nonzero c) 1 1))
       (w (rndn 7 3)))
  (prn ($@ c w))
  (prn ($wimb c2 w))
  (prn ($wemb c w)))

(let ((c0 (tensor '((1 0 0 0 0 0 0))))
      (c1 (tensor '((0 0 1 0 0 0 0))))
      (win (rndn 7 3))
      (wout (rndn 3 7)))
  (let* ((h0 ($wemb c0 win))
         (h1 ($wemb c1 win))
         (h ($* 0.5 ($+ h0 h1)))
         (s ($wemb h wout)))
    (prn ($softmax s))))

(defun preprocess (text)
  (let* ((lowered (string-downcase text))
         (wm (make-word-maps (th.text::collect-words (list lowered))))
         (corpus (make-corpus wm lowered)))
    (list :corpus corpus
          :vocab-size (getf wm :vocab-size)
          :word-to-index(getf wm :word-to-index)
          :index-to-word (getf wm :index-to-word))))

(defparameter *text* "You say goodbye and I say hello.")
(defparameter *data* (preprocess *text*))

(let ((data *data*))
  (prn (getf data :corpus))
  (prn (getf data :index-to-word)))

(defun create-contexts-target (corpus &key (window-size 1))
  (let ((target (subseq corpus window-size (- ($count corpus) window-size)))
        (contexts '()))
    (loop :for idx :from window-size :below (- ($count corpus) window-size)
          :do (let ((cs '()))
                (loop :for i :from (- window-size) :below (1+ window-size)
                      :do (unless (eq i 0)
                            (push ($ corpus (+ idx i)) cs)))
                (push (reverse cs) contexts)))
    (list :contexts (tensor.long (reverse contexts))
          :target (tensor (coerce target 'list)))))

(let ((ct (create-contexts-target (getf *data* :corpus))))
  (prn (getf *data* :corpus))
  (prn (getf ct :contexts))
  (prn (getf ct :target))
  (prn ($ (getf ct :contexts) 0)))

($wimb)
