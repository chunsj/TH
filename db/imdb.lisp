(defpackage :th.db.imdb
  (:use #:common-lisp
        #:mu
        #:th)
  (:export #:read-imdb-data))

(in-package :th.db.imdb)

(defparameter +imdb-location+ ($concat (namestring (user-homedir-pathname)) "IMDB"))

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
