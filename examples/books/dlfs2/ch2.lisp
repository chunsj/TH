(defpackage :dlfs2-ch2
  (:use #:common-lisp
        #:mu
        #:th
        #:th.text
        #:th.ex.data
        #:mplot))

(in-package :dlfs2-ch2)

(defun preprocess (text)
  (let* ((lowered (string-downcase text))
         (wm (make-word-maps (th.text::collect-words (list lowered))))
         (corpus (make-corpus wm lowered)))
    (list :corpus corpus
          :vocab-size (getf wm :vocab-size)
          :word-to-index(getf wm :word-to-index)
          :index-to-word (getf wm :index-to-word))))

(defparameter *text* "You say goodbye and I say hello.")

(let ((data (preprocess *text*)))
  (prn (getf data :corpus))
  (prn (getf data :id-to-word)))

(defparameter *data* (preprocess *text*))

(defparameter *com* (tensor '((0 1 0 0 0 0 0)
                              (1 0 1 0 1 1 0)
                              (0 1 0 1 0 0 0)
                              (0 0 1 0 1 0 0)
                              (0 1 0 1 0 0 0)
                              (0 1 0 0 0 0 1)
                              (0 0 0 0 0 1 0))))

(prn ($ *com* 0))
(prn ($ *com* 4))
(prn ($ *com* (word-to-index *data* "goodbye")))

(defparameter *com* (create-co-matrix (getf *data* :corpus)
                                      (getf *data* :vocab-size)))

(prn *com*)
(prn (words *data*))
(prn (word-to-index *data* "you"))

(let ((c0 ($ *com* (word-to-index *data* "you")))
      (c1 ($ *com* (word-to-index *data* "i"))))
  (prn (cosine-similarity c0 c1)))

(most-similar "you" *data* *com* :top 5)

(prn (ppmi *com*))

(defparameter *w* (ppmi *com*))

(let ((usv ($svd *w*)))
  (defparameter *u* (car usv))
  (defparameter *s* (cadr usv))
  (defparameter *v* (caddr usv)))

(prn ($ *com* 0))
(prn ($ *w* 0))
(prn ($ *u* 0))
(prn ($index ($ *u* 0) 0 '(0 1)))

(let* ((dxy (loop :for word :in (words *data*)
                  :for id = (word-to-index *data* word)
                  :for xy = (list ($ *u* id 0) ($ *u* id 1))
                  :collect (cons word xy)))
       (xrange (let ((xmin (apply #'min (mapcar #'cadr dxy)))
                     (xmax (apply #'max (mapcar #'cadr dxy))))
                 (list xmin xmax)))
       (yrange (let ((ymin (apply #'min (mapcar #'caddr dxy)))
                     (ymax (apply #'max (mapcar #'caddr dxy))))
                 (list ymin ymax))))
  (prn dxy)
  (plot-points (mapcar #'cdr dxy) :xrange xrange :yrange yrange))

(defparameter *ptb* (ptb))
(defparameter *ptb-wm* (make-word-maps (th.text::collect-words *ptb*)))
(defparameter *corpus* (make-corpus *ptb-wm* (format nil "~{~A~^ ~}" *ptb*)))
(defparameter *c* (create-co-matrix *corpus* (vocab-size *ptb-wm*) :window-size 2))

(prn (vocab-size *ptb-wm*))

(index-to-word *ptb-wm* 0)
(index-to-word *ptb-wm* 1)
(index-to-word *ptb-wm* 2)

(word-to-index *ptb-wm* "car")
(word-to-index *ptb-wm* "happy")
(word-to-index *ptb-wm* "lexus")

(defparameter *w* (ppmi *c* :verbose t))

;; this should be done with randomized SVD or very slow
(defparameter *trsvd-usv* ($rsvd *w* 100))
