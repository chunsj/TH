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

(defparameter *ct* (create-contexts-target (getf *data* :corpus)))

(defun convert-one-hot (x sz)
  (cond ((eq ($ndim x) 1)
         (let ((r (zeros ($size x 0) sz)))
           (loop :for i :from 0 :below ($size x 0)
                 :for v = ($ x i)
                 :do (setf ($ r i (round v)) 1))
           r))
        ((eq ($ndim x) 2)
         (let ((r (zeros ($size x 0) ($size x 1) sz)))
           (loop :for i :from 0 :below ($size x 0)
                 :for v = ($ x i)
                 :do (loop :for j :from 0 :below ($size v 0)
                           :for vv = ($ v j)
                           :do (setf ($ r i j (round vv)) 1)))
           r))
        (T (error "cannot convert tensor of ~A dimension" ($ndim x)))))

(let ((ct *ct*))
  (prn (getf ct :target))
  (prn (convert-one-hot (getf ct :target) (getf *data* :vocab-size)))
  (prn (getf ct :contexts))
  (prn (convert-one-hot (getf ct :contexts) (getf *data* :vocab-size))))

(prn ($squeeze ($index (convert-one-hot (getf *ct* :contexts) (getf *data* :vocab-size)) 1 '(1))))

(defparameter *hidden-size* 5)
(defparameter *win* ($parameter (rndn (getf *data* :vocab-size) *hidden-size*)))
(defparameter *wout* ($parameter (rndn *hidden-size* (getf *data* :vocab-size))))

(defun forward (contexts)
  (let ((h0 ($@ ($squeeze ($index contexts 1 '(0))) *win*))
        (h1 ($@ ($squeeze ($index contexts 1 '(1))) *win*)))
    ($@ ($* 0.5 ($+ h0 h1)) *wout*)))

(defun loss (h target) ($cee ($softmax h) target))

($cg! (list *win* *wout*))

(let ((contexts (convert-one-hot (getf *ct* :contexts) (getf *data* :vocab-size)))
      (target (convert-one-hot (getf *ct* :target) (getf *data* :vocab-size))))
  (loop :for epoch :from 0 :below 1000
        :do (let ((loss (loss (forward contexts) target)))
              (prn loss)
              ($amgd! (list *win* *wout*)))))

(gcf)

(loop :for wid :being :the :hash-keys :of (getf *data* :index-to-word)
      :for word = ($ (getf *data* :index-to-word) wid)
      :do (format T "~A~%~A~%" word ($ ($data  *win*) wid)))
