(defpackage :th.ex.data
  (:use #:common-lisp
        #:mu)
  (:export #:text-lines))

(in-package :th.ex.data)

(defun data-filename (name)
  ($concat (namestring (asdf:system-source-directory :th)) "data/" name))

(defun text-lines (&optional (file :tiny-shakespeare))
  (cond ((eq file :tiny-shakespeare) (read-lines-from (data-filename "tinyshakespeare.txt")))
        ((eq file :obama) (read-lines-from (data-filename "obama.txt")))
        ((eq file :pg) (read-lines-from (data-filename "pg.txt")))
        ((eq file :small-pg) (read-lines-from (data-filename "pgsmall.txt")))))
