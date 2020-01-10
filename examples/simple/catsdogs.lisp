;; note that this examples is for testing cnn on color images.
;; to get better result, you have to define better network and more data.
;; this example, as is provided, will have 31~40% of error rates.

(defpackage :cats-and-dogs
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers
        #:th.image
        #:th.db.cats-and-dogs))

(in-package :cats-and-dogs)

(defparameter *batch-size* 10)
(defparameter *batch-count* 50)
(defparameter *test-size* 100)
(defparameter *img-size* 64)

(defparameter *data-index* 10)

(defparameter *train-data*
  (let* ((n *data-index*)
         (rng (loop :for i :from 1 :to n :collect i))
         (txs (read-cats-and-dogs-data :indices rng)))
    (when (< (* *batch-count* *batch-size*) (* n 1000))
      (loop :for bidx :from 0 :below *batch-count*
            :for sidx = (* bidx *batch-size*)
            :for eidx = (+ sidx *batch-size*)
            :for indices = (loop :for i :from sidx :below eidx :collect i)
            :collect ($index txs 0 indices)))))

(defparameter *train-labels*
  (let* ((n *data-index*)
         (txs (zeros (* n 1000))))
    (loop :for i :from 0 :below (* n 1000)
          :do (setf ($ txs i) (if (zerop (rem i 2)) 1 0)))
    (when (< (* *batch-count* *batch-size*) (* n 1000))
      (loop :for bidx :from 0 :below *batch-count*
            :for sidx = (* bidx *batch-size*)
            :for eidx = (+ sidx *batch-size*)
            :for indices = (loop :for i :from sidx :below eidx :collect i)
            :collect ($index txs 0 indices)))))

(defparameter *test-data* ($index (read-cats-and-dogs-data :indices '(25))
                                  0
                                  (loop :for i :from 0 :below *test-size* :collect i)))
(defparameter *test-labels* (tensor (loop :for i :from 0 :below ($size *test-data* 0)
                                          :collect (if (zerop (rem i 2)) 1 0))))

(defparameter *output-directory* ($concat (namestring (user-homedir-pathname))
                                          "Desktop"))

(defun write-rgb-png-file (tensor filename)
  (write-tensor-png-file tensor (format nil "~A/~A" *output-directory* filename)))

(defun write-gray-png-file (tensor filename &optional (channel 0))
  (write-tensor-png-file ($ tensor channel) (format nil "~A/~A" *output-directory* filename)))

(defparameter *network* (sequential-layer
                         (convolution-2d-layer 3 32 3 3
                                               :activation :lrelu
                                               :weight-initializer :random-normal
                                               :weight-initialization '(0 0.01))
                         (maxpool-2d-layer 2 2)
                         (convolution-2d-layer 32 32 3 3
                                               :activation :lrelu
                                               :weight-initializer :random-normal
                                               :weight-initialization '(0 0.01))
                         (maxpool-2d-layer 2 2)
                         (flatten-layer)
                         (functional-layer (lambda (x &key (trainp t)) ($dropout x trainp 0.4)))
                         (affine-layer (* 32 58 58) 128
                                       :activation :lrelu
                                       :weight-initializer :random-normal
                                       :weight-initialization '(0 0.01))
                         (affine-layer 128 1
                                       :activation :sigmoid
                                       :weight-initializer :random-normal
                                       :weight-initialization '(0 0.01))))

($reset! *network*)
(gcf)

(defun opt! () ($amgd! *network* 1E-4))

(defparameter *epoch* 60) ;; at least 120
(defparameter *train-size* ($count *train-data*))

(time
 (with-foreign-memory-limit ()
   (loop :for epoch :from 1 :to *epoch*
         :do (progn
               (loop :for data :in (subseq *train-data* 0 *train-size*)
                     :for labels :in (subseq *train-labels* 0 *train-size*)
                     :for bidx :from 1
                     :do (let* ((y* ($execute *network* data))
                                (loss ($bce y* labels)))
                           (prn epoch "|" bidx ($data loss))
                           (opt!)))
               (when (zerop (rem epoch 5))
                 (let* ((res ($evaluate *network* *test-data*))
                        (fres (tensor.float ($ge res 0.5)))
                        (d ($- fres *test-labels*)))
                   (prn "TEST ERROR:" (/ ($dot d d) *test-size*))))))))

;; for testing
(setf *epoch* 1)
(setf *train-size* 1)

;; train check
(let* ((idx (random *train-size*))
       (data (nth idx *train-data*))
       (lbl (nth idx *train-labels*))
       (y ($evaluate *network* data))
       (res (tensor.float ($ge y 0.5)))
       (d ($- res lbl)))
  (prn "TRAIN IDX:" idx "ERROR:" (/ ($dot d d) *batch-size*))
  (gcf))

;; test check
(let* ((res ($evaluate *network* *test-data*))
       (fres (tensor.float ($ge res 0.5)))
       (d ($- fres *test-labels*)))
  (prn "TEST ERROR:" (/ ($dot d d) *test-size*))
  (gcf))
