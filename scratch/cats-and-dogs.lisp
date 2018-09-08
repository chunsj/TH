(defpackage :cats-and-dogs
  (:use #:common-lisp
        #:mu
        #:th
        #:th.image))

(in-package :cats-and-dogs)

(defparameter *data-directory* "/Users/Sungjin/CatsDogs")

(defun read-train-cat-file (number &optional (h 64) (w 64))
  (when (and (>= number 0) (<= number 10000))
    (let ((filename (format nil "~A/train/cat.~A.jpg" *data-directory* number)))
      (tensor-from-jpeg-file filename :resize-dimension (list h w)))))

(defun read-train-dog-file (number &optional (h 64) (w 64))
  (when (and (>= number 0) (<= number 10000))
    (let ((filename (format nil "~A/train/dog.~A.jpg" *data-directory* number)))
      (tensor-from-jpeg-file filename :resize-dimension (list h w)))))

(defparameter *output-directory* "/Users/Sungjin/Desktop")

(defun write-rgb-png-file (tensor filename)
  (write-tensor-png-file tensor (format nil "~A/~A" *output-directory* filename)))

(defun write-gray-png-file (tensor filename &optional (channel 0))
  (write-tensor-png-file ($ tensor channel) (format nil "~A/~A" *output-directory* filename)))

(write-rgb-png-file (read-train-cat-file 10 64 64) "cat10.png")
(write-gray-png-file (read-train-dog-file 20 64 64) "dog20.png")
