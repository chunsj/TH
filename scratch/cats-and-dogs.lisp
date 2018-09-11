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
(defparameter *batch-count* 500) ;; almost 20 secs or more
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
          :do (push (let ((tensor (zeros (* 2 *batch-size*) 2)))
                      (loop :for i :from sidx :below (+ sidx *batch-size*)
                            :do (setf ($ tensor (* 2 (- i sidx)) 0) 1
                                      ($ tensor (1+ (* 2 (- i sidx))) 1) 1))
                      tensor)
                    data))
    (reverse data)))

(defparameter *cnd* (parameters))

(defparameter *k1* ($parameter *cnd* ($* 0.01 (rndn 32 3 5 5))))
(defparameter *b1* ($parameter *cnd* (zeros 32)))
(defparameter *k2* ($parameter *cnd* ($* 0.01 (rndn 64 32 5 5))))
(defparameter *b2* ($parameter *cnd* (zeros 64)))
(defparameter *k3* ($parameter *cnd* ($* 0.01 (rndn 128 64 5 5))))
(defparameter *b3* ($parameter *cnd* (zeros 128)))
(defparameter *k4* ($parameter *cnd* ($* 0.01 (rndn 64 128 5 5))))
(defparameter *b4* ($parameter *cnd* (zeros 64)))
(defparameter *k5* ($parameter *cnd* ($* 0.01 (rndn 32 64 5 5))))
(defparameter *b5* ($parameter *cnd* (zeros 32)))
(defparameter *w6* ($parameter *cnd* (vxavier '(3200 1024))))
(defparameter *b6* ($parameter *cnd* (zeros 1 1024)))
(defparameter *w7* ($parameter *cnd* (vxavier '(1024 2))))
(defparameter *b7* ($parameter *cnd* (zeros 1 2)))

(defun network (x)
  (-> x
      ($conv2d *k1* *b1*)
      ($selu)
      ($maxpool2d 5 5)
      ($conv2d *k2* *b2*)
      ($selu)
      ($maxpool2d 5 5)
      ($conv2d *k3* *b3*)
      ($selu)
      ($maxpool2d 5 5)
      ($conv2d *k4* *b4*)
      ($selu)
      ($maxpool2d 5 5)
      ($conv2d *k5* *b5*)
      ($selu)
      ($maxpool2d 5 5)
      ($reshape ($size x 0) 3200)
      ($affine *w6* *b6*)
      ($relu)
      ($affine *w7* *b7*)
      ($softmax)))

($cg! *cnd*)
(gcf)

(defun opt! (parameters) ($adgd! parameters))

(defparameter *epoch* 5)

(loop :for epoch :from 1 :to *epoch*
      :do (progn
            (loop :for data :in (subseq *train-data* 0 5)
                  :for labels :in (subseq *train-labels* 0 5)
                  :for bidx :from 1
                  :do (let* ((y* (network ($constant data)))
                             (loss ($bce y* ($constant labels))))
                        (prn epoch "|" bidx ($data loss))
                        (opt! *cnd*)
                        (gcf)))))
