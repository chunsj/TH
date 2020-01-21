(defpackage :dlfs2-ch2
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :dlfs2-ch2)

(defun replace-all (string part replacement &key (test #'char=))
  "Returns a new string in which all the occurences of the part
is replaced with replacement."
  (with-output-to-string (out)
    (loop :with part-length = (length part)
          :for old-pos = 0 :then (+ pos part-length)
          :for pos = (search part string
                             :start2 old-pos
                             :test test)
          :do (write-string string out
                            :start old-pos
                            :end (or pos (length string)))
          :when pos :do (write-string replacement out)
            :while pos)))

(defun preprocess (text)
  (let* ((lowered (string-downcase text))
         (perioded (replace-all lowered "." " ."))
         (words (split #\space perioded)))
    (let ((word-to-id #{})
          (id-to-word #{}))
      (loop :for word :in words
            :when (null ($ word-to-id word nil))
              :do (let ((new-id ($count word-to-id)))
                    (setf ($ word-to-id word) new-id
                          ($ id-to-word new-id) word)))
      (list :corpus (loop :for word :in words
                          :collect ($ word-to-id word))
            :vocab-size ($count word-to-id)
            :word-to-id word-to-id
            :id-to-word id-to-word))))

(defparameter *text* "You say goodbye and I say hello.")

(let ((data (preprocess *text*)))
  (prn (getf data :corpus))
  (prn (getf data :id-to-word)))

(defparameter *data* (preprocess *text*))

(defparameter *coccurence* (tensor '((0 1 0 0 0 0 0)
                                     (1 0 1 0 1 1 0)
                                     (0 1 0 1 0 0 0)
                                     (0 0 1 0 1 0 0)
                                     (0 1 0 1 0 0 0)
                                     (0 1 0 0 0 0 1)
                                     (0 0 0 0 0 1 0))))

(prn ($ *coccurence* 0))
(prn ($ *coccurence* 4))
(prn ($ *coccurence* ($ (getf *data* :word-to-id) "goodbye")))

(defun create-coccurence-matrix (corpus vocab-size &key (window-size 1))
  (let ((corpus-size ($count corpus))
        (comatrix (tensor (zeros vocab-size vocab-size))))
    (loop :for word-id :in corpus
          :for idx :from 0
          :do (loop :for i :from 1 :below (1+ window-size)
                    :for left-idx = (1- idx)
                    :for right-idx = (1+ idx)
                    :do (progn
                          (when (>= left-idx 0)
                            (let ((left-word-id ($ corpus left-idx)))
                              (setf ($ comatrix word-id left-word-id)
                                    (1+ ($ comatrix word-id left-word-id)))))
                          (when (< right-idx corpus-size)
                            (let ((right-word-id ($ corpus right-idx)))
                              (setf ($ comatrix word-id right-word-id)
                                    (1+ ($ comatrix word-id right-word-id))))))))
    comatrix))

(prn (create-coccurence-matrix (getf *data* :corpus) (getf *data* :vocab-size)))

(defparameter *coccurence* (create-coccurence-matrix (getf *data* :corpus)
                                                     (getf *data* :vocab-size)))

(defun cosine-similarity (x y &key (eps 1E-8))
  (let ((nx ($/ x ($+ ($sqrt ($sum ($square x))) eps)))
        (ny ($/ y ($+ ($sqrt ($sum ($square y))) eps))))
    ($dot nx ny)))

(let ((c0 ($ *coccurence* ($ (getf *data* :word-to-id) "you")))
      (c1 ($ *coccurence* ($ (getf *data* :word-to-id) "i"))))
  (prn (cosine-similarity c0 c1)))

(defun most-similar (query vocab-size word-to-id id-to-word word-matrix &key (top 5))
  (prn top)
  (if (null ($ word-to-id query nil))
      (prn (format nil "cannot find ~A" query))
      (progn
        (prn)
        (prn "[QUERY]" query)
        (let* ((query-id ($ word-to-id query))
               (query-vec ($ word-matrix query-id))
               (similarity (zeros vocab-size)))
          (loop :for i :from 0 :below vocab-size
                :do (setf ($ similarity i)
                          (cosine-similarity ($ word-matrix i) query-vec)))
          (let* ((sorted ($sort similarity 0 T))
                 (sorted-indices (cadr sorted)))
            (loop :for r :from 0 :below ($count similarity)
                  :for i = ($ sorted-indices r)
                  :for count :from 0
                  :while (<= count top)
                  :do (unless (string-equal ($ id-to-word i) query)
                        (prn (format nil "  ~A: ~A" ($ id-to-word i) ($ similarity i))))))))))

(most-similar "you"
              (getf *data*  :vocab-size)
              (getf *data* :word-to-id)
              (getf *data* :id-to-word)
              *coccurence*
              :top 5)

(defun ppmi (coccurrence &key verbose (eps 1E-8))
  (let ((m ($zero coccurrence))
        (n ($sum coccurrence))
        (s ($sum coccurrence 0))
        (total (* ($size coccurrence 0)
                  ($size coccurrence 1)))
        (cnt 0))
    (loop :for i :from 0 :below ($size coccurrence 0)
          :do (loop :for j :from 0 :below ($size coccurrence 1)
                    :for pmi = (log (+ eps (/ (* ($ coccurrence i j) N)
                                              (* ($ s 0 j) ($ s 0 i))))
                                    2)
                    :do (progn
                          (setf ($ m i j) (max 0 pmi))
                          (when verbose
                            (incf cnt)
                            (when (zerop (rem cnt (round (/ total 100))))
                              (prn (format nil "~,1F% completed" (* 100 (/ cnt total)))))))))
    m))

(prn (ppmi *coccurence*))

(defparameter *w* (ppmi *coccurence*))

(prn ($svd *w*))

(let ((usv ($svd *w*)))
  (defparameter *u* (car usv))
  (defparameter *s* (cadr usv))
  (defparameter *v* (caddr usv)))

(prn ($ *coccurence* 0))
(prn ($ *w* 0))
(prn ($ *u* 0))
(prn ($index ($ *u* 0) 0 '(0 1)))

(let ((dxy (loop :for word :in (hash-table-keys (getf *data* :word-to-id))
                 :for id = ($ (getf *data* :word-to-id) word)
                 :for xy = (list ($ *u* id 0) ($ *u* id 1))
                 :collect (cons word xy))))
  (prn dxy)
  (mplot:plot-points (mapcar #'cdr dxy) :xrange '(-0.1 0.8) :yrange '(-0.7 0.1)))
