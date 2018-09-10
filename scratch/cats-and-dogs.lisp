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

;;(write-rgb-png-file (read-train-cat-file 10 64 64) "cat10.png")
;;(write-gray-png-file (read-train-dog-file 20 64 64) "dog20.png")

(defparameter *batch-size* 10)
(defparameter *batch-count* 50) ;; almost 20 secs or more
(defparameter *img-size* 50)

(defparameter *train-data*
  (let ((data nil))
    (loop :for bidx :from 0 :below *batch-count*
          :for sidx = (* bidx *batch-size*)
          :do (push (let ((tensor (tensor (* 2 *batch-size*) 3 *img-size* *img-size*)))
                      (loop :for i :from sidx :below (+ sidx *batch-size*)
                            :for cat = (read-train-cat-file i *img-size* *img-size*)
                            :for dog = (read-train-dog-file i *img-size* *img-size*)
                            :do (setf ($ tensor (* 2 (- i sidx))) cat
                                      ($ tensor (1+ (* 2 (- i sidx)))) dog))
                      tensor)
                    data))
    (reverse data)))

(defparameter *train-labels*
  (let ((data nil))
    (loop :for bidx :from 0 :below *batch-count*
          :for sidx = (* bidx *batch-size*)
          :do (push (let ((tensor (tensor (* 2 *batch-size*) 1)))
                      (loop :for i :from sidx :below (+ sidx *batch-size*)
                            :do (setf ($ tensor (* 2 (- i sidx))) 0
                                      ($ tensor (1+ (* 2 (- i sidx)))) 1))
                      tensor)
                    data))
    (reverse data)))

(defparameter *cnd* (parameters))

(defparameter *k1* ($parameter *cnd* ($* 0.01 (rndn 16 3 5 5))))
(defparameter *b1* ($parameter *cnd* (zeros 16)))
(defparameter *k2* ($parameter *cnd* ($* 0.01 (rndn 32 16 5 5))))
(defparameter *b2* ($parameter *cnd* (zeros 32)))
(defparameter *w3* ($parameter *cnd* ($* 0.01 (rndn 51200 1024))))
(defparameter *b3* ($parameter *cnd* (zeros 1 1024)))
(defparameter *w4* ($parameter *cnd* ($* 0.01 (rndn 1024 2))))
(defparameter *b4* ($parameter *cnd* (zeros 1 2)))

(defun network (x)
  (-> x
      ($conv2d *k1* *b1*)
      ($selu)
      ($maxpool2d 2 2)
      ($conv2d *k2* *b2*)
      ($selu)
      ($maxpool2d 2 2)
      ($reshape ($size x 0) 51200)
      ($affine *w3* *b3*)
      ($selu)
      ($affine *w4* *b4*)
      ($softmax)))

($cg! *cnd*)
(gcf)
