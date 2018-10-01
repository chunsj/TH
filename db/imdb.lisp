(declaim (optimize (speed 3) (debug 0) (safety 0)))

(defpackage :th.db.imdb
  (:use #:common-lisp
        #:mu
        #:th)
  (:export #:read-imdb-data
           #:read-imdb-data2))

(in-package :th.db.imdb)

(defparameter +imdb-location+ ($concat (namestring (user-homedir-pathname)) ".th/datasets/imdb"))

(defun imdb-reviews (path)
  (let ((filename ($concat path "/reviews-small.txt")))
    (read-lines-from filename)))

(defun imdb-labels (path)
  (let ((filename ($concat path "/labels-small.txt")))
    (read-lines-from filename)))

(defun imdb-reviews-test (path)
  (let ((filename ($concat path "/reviews-small-test.txt")))
    (read-lines-from filename)))

(defun imdb-labels-test (path)
  (let ((filename ($concat path "/labels-small-test.txt")))
    (read-lines-from filename)))

(defun read-imdb-data (&key (path +imdb-location+))
  #{:train-reviews (imdb-reviews path)
    :train-labels (imdb-labels path)
    :test-reviews (imdb-reviews-test path)
    :test-labels (imdb-labels-test path)})

(defun imdb-reviews2 (path)
  (let ((filename ($concat path "/greviews.txt")))
    (read-lines-from filename)))

(defun imdb-labels2 (path)
  (let ((filename ($concat path "/glabels.txt")))
    (read-lines-from filename)))

(defun read-imdb-data2 (&key (path +imdb-location+))
  #{:reviews (imdb-reviews2 path)
    :labels (imdb-labels2 path)})
