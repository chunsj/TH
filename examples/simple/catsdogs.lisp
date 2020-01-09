;; note that this examples is for testing cnn on color images.
;; to get better result, you have to define better network and more data.
;; this example, as is provided, will have 31~40% of error rates.

(defpackage :cats-and-dogs
  (:use #:common-lisp
        #:mu
        #:th
        #:th.image
        #:th.db.cats-and-dogs))

(in-package :cats-and-dogs)

(defparameter *data-directory* ($concat (namestring (user-homedir-pathname))
                                        ".th/datasets/cats-and-dogs"))

(defun read-train-cat-file (number &optional (h 64) (w 64))
  (when (and (>= number 0) (<= number 10000))
    (let ((filename (format nil "~A/train/cat.~A.jpg" *data-directory* number)))
      (tensor-from-jpeg-file filename :resize-dimension (list h w)))))

(defun read-train-dog-file (number &optional (h 64) (w 64))
  (when (and (>= number 0) (<= number 10000))
    (let ((filename (format nil "~A/train/dog.~A.jpg" *data-directory* number)))
      (tensor-from-jpeg-file filename :resize-dimension (list h w)))))

(defparameter *output-directory* ($concat (namestring (user-homedir-pathname))
                                          "Desktop"))

(defun write-rgb-png-file (tensor filename)
  (write-tensor-png-file tensor (format nil "~A/~A" *output-directory* filename)))

(defun write-gray-png-file (tensor filename &optional (channel 0))
  (write-tensor-png-file ($ tensor channel) (format nil "~A/~A" *output-directory* filename)))

;;(write-rgb-png-file (read-train-cat-file 10 64 64) "cat10.png")
;;(write-gray-png-file (read-train-dog-file 20 64 64) "dog20.png")

(defparameter *batch-size* 10)
(defparameter *batch-count* 50)
(defparameter *test-count* 2)
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

(defparameter *test-data*
  (let ((data nil))
    (loop :for bidx :from *batch-count* :below (+ *batch-count* *test-count*)
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

(defparameter *test-labels*
  (let ((data nil))
    (loop :for bidx :from *batch-count* :below (+ *batch-count* *test-count*)
          :for sidx = (* bidx *batch-size*)
          :do (push (let ((tensor (zeros (* 2 *batch-size*))))
                      (loop :for i :from sidx :below (+ sidx *batch-size*)
                            :do (setf ($ tensor (* 2 (- i sidx))) 1))
                      tensor)
                    data))
    (reverse data)))

(defun odim (iw ih kw kh dw dh pw ph)
  (list (1+ (round (/ (+ (- iw kw) (* 2 pw)) dw)))
        (1+ (round (/ (+ (- ih kh) (* 2 ph)) dh)))))

;; k should be (kw kh dw dh pw ph)
(defun compute-size (input-size &rest ks)
  (let ((sz input-size))
    (loop :for k :in ks
          :do (let ((o (apply #'odim (append sz k))))
                (setf sz o)))
    sz))

;; output channel x output dimensions
(defparameter *fdim* (reduce #'* (cons 32 (compute-size (list *img-size* *img-size*)
                                                        '(3 3 1 1 0 0)
                                                        '(2 2 1 1 0 0)
                                                        '(3 3 1 1 0 0)
                                                        '(2 2 1 1 0 0)))))

(defparameter *cnd* (parameters))

(defparameter *k1* ($push *cnd* ($* 0.01 (rndn 32 3 3 3))))
(defparameter *b1* ($push *cnd* (zeros 32)))
(defparameter *k2* ($push *cnd* ($* 0.01 (rndn 32 32 3 3))))
(defparameter *b2* ($push *cnd* (zeros 32)))
(defparameter *w3* ($push *cnd* (vxavier (list *fdim* 128))))
(defparameter *b3* ($push *cnd* (zeros 128)))
(defparameter *w4* ($push *cnd* (vxavier '(128 1))))
(defparameter *b4* ($push *cnd* (zeros 1)))

(defun network (x &optional (trainp t))
  (-> x
      ($conv2d *k1* *b1*)
      ($relu)
      ($maxpool2d 2 2)
      ($conv2d *k2* *b2*)
      ($relu)
      ($maxpool2d 2 2)
      ($reshape ($size x 0) *fdim*)
      ($dropout trainp 0.4)
      ($affine *w3* *b3*)
      ($relu)
      ($affine *w4* *b4*)
      ($sigmoid)))

($cg! *cnd*)
(gcf)

(defun opt! (parameters) ($amgd! parameters 1E-4))

(defparameter *epoch* 60)
(defparameter *train-size* ($count *train-data*))

(setf *epoch* 10)
(time
 (with-foreign-memory-limit ()
   (loop :for epoch :from 1 :to *epoch*
         :do (progn
               (loop :for data :in (subseq *train-data* 0 *train-size*)
                     :for labels :in (subseq *train-labels* 0 *train-size*)
                     :for bidx :from 1
                     :do (let* ((y* (network data))
                                (loss ($bce y* labels)))
                           (prn epoch "|" bidx ($data loss))
                           (opt! *cnd*)))
               (when (zerop (rem epoch 5))
                 (let* ((idx (random *test-count*))
                        (tdata (nth idx *test-data*))
                        (tlbl (nth idx *test-labels*))
                        (res ($data (network tdata nil)))
                        (fres (tensor.float ($ge res 0.5)))
                        (d ($- ($reshape fres (* 2 *batch-size*)) tlbl)))
                   (prn "IDX:" idx "ERROR:" (/ ($dot d d) (* 2 *batch-size*)))
                   ($cg! *cnd*)))))))

;; train check
(let* ((idx (random *train-size*))
       (data (nth idx *train-data*))
       (lbl (car *train-labels*))
       (y (network data nil))
       (res (tensor.float ($ge ($data y) 0.5)))
       (d ($- ($reshape res (* 2 *batch-size*)) lbl)))
  (prn "TRAIN IDX:" idx "ERROR:" (/ ($dot d d) (* 2 *batch-size*)))
  ($cg! *cnd*)
  (gcf))

;; test check
(let* ((idx (random *test-count*))
       (tdata (nth idx *test-data*))
       (tlbl (nth idx *test-labels*))
       (res ($data (network tdata nil)))
       (fres (tensor.float ($ge res 0.5)))
       (d ($- ($reshape fres (* 2 *batch-size*)) tlbl)))
  (prn "TEST IDX:" idx "ERROR:" (/ ($dot d d) (* 2 *batch-size*)))
  ($cg! *cnd*)
  (gcf))

(setf *train-data* nil
      *train-labels* nil
      *test-data* nil
      *test-labels* nil)
(gcf)
