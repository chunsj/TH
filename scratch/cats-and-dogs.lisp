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

(defparameter *batch-size* 50)
(defparameter *batch-count* 100)
(defparameter *img-size* 64)

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
          :do (push (let ((tensor (zeros (* 2 *batch-size*))))
                      (loop :for i :from sidx :below (+ sidx *batch-size*)
                            :do (setf ($ tensor (* 2 (- i sidx))) 1))
                      tensor)
                    data))
    (reverse data)))

(defparameter *cnd* (parameters))

(defparameter *k1* ($parameter *cnd* ($* 0.01 (rndn 32 3 3 3))))
(defparameter *b1* ($parameter *cnd* (zeros 32)))
(defparameter *k2* ($parameter *cnd* ($* 0.01 (rndn 32 32 3 3))))
(defparameter *b2* ($parameter *cnd* (zeros 32)))
(defparameter *w3* ($parameter *cnd* (vxavier (list (* 32 58 58) 128))))
(defparameter *b3* ($parameter *cnd* (zeros 1 128)))
(defparameter *w4* ($parameter *cnd* (vxavier '(128 1))))
(defparameter *b4* ($parameter *cnd* (zeros 1 1)))

(defun network (x)
  (-> x
      ($conv2d *k1* *b1*)
      ($relu)
      ($maxpool2d 2 2)
      ($conv2d *k2* *b2*)
      ($relu)
      ($maxpool2d 2 2)
      ($reshape ($size x 0) (* 32 58 58))
      ($affine *w3* *b3*)
      ($relu)
      ($affine *w4* *b4*)
      ($sigmoid)))

($cg! *cnd*)
(gcf)

(defun opt! (parameters) ($amgd! parameters 1E-4))

(defparameter *epoch* 1) ;; 200 is okay?
(defparameter *train-size* (min 100 ($count *train-data*))) ;; to reduce time

;; 1,2,3,4,5,6,7
(loop :for epoch :from 1 :to *epoch*
      :do (progn
            (loop :for data :in (subseq *train-data* 0 *train-size*)
                  :for labels :in (subseq *train-labels* 0 *train-size*)
                  :for bidx :from 1
                  :do (let* ((y* (network ($constant data)))
                             (loss ($bce y* ($constant labels))))
                        (prn epoch "|" bidx ($data loss))
                        (opt! *cnd*)
                        (when (zerop (rem bidx 2))
                          (gcf))))
            (when (zerop (rem epoch 10))
              (let* ((bidx 50)
                     (sidx (* bidx *batch-size*))
                     (tensor (tensor (* 2 *batch-size*) 3 *img-size* *img-size*)))
                (loop :for i :from sidx :below (+ sidx *batch-size*)
                      :for cat = (read-train-cat-file i *img-size* *img-size*)
                      :for dog = (read-train-dog-file i *img-size* *img-size*)
                      :do (setf ($ tensor (* 2 (- i sidx))) cat
                                ($ tensor (1+ (* 2 (- i sidx)))) dog))
                (prn "TEST" ($ge ($data (network ($constant tensor))) 0.5))
                ($cg! *cnd*)))))

(prn ($reshape (tensor.float ($ge ($data (network ($constant (car *train-data*)))) 0.5))
               (* 2 *batch-size*)))
(prn (car *train-labels*))
($cg! *cnd*)

;; test
(let* ((bidx *train-size*)
       (sidx (* bidx *batch-size*))
       (tensor (tensor (* 2 *batch-size*) 3 *img-size* *img-size*)))
  (loop :for i :from sidx :below (+ sidx *batch-size*)
        :for cat = (read-train-cat-file i *img-size* *img-size*)
        :for dog = (read-train-dog-file i *img-size* *img-size*)
        :do (setf ($ tensor (* 2 (- i sidx))) cat
                  ($ tensor (1+ (* 2 (- i sidx)))) dog))
  (prn "ERROR:" (let ((d ($- ($reshape (tensor.float ($ge ($data (network ($constant tensor))) 0.5))
                                    (* 2 *batch-size*))
                          (car *train-labels*))))
               (/ ($dot d d) (* 2 *batch-size*))))
  ($cg! *cnd*))
