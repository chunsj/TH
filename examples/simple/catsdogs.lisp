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

(defparameter *test-data* (read-cats-and-dogs-data :indices '(25)))
(defparameter *test-labels* (tensor (loop :for i :from 0 :below ($size *test-data* 0)
                                          :collect (if (zerop (rem i 2)) 1 0))))

(defparameter *output-directory* ($concat (namestring (user-homedir-pathname))
                                          "Desktop"))

(defun write-rgb-png-file (tensor filename)
  (write-tensor-png-file tensor (format nil "~A/~A" *output-directory* filename)))

(defun write-gray-png-file (tensor filename &optional (channel 0))
  (write-tensor-png-file ($ tensor channel) (format nil "~A/~A" *output-directory* filename)))

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

(defparameter *network* (sequential-layer
                         (convolution-2d-layer 3 32 3 3 :activation :relu)
                         (maxpool-2d-layer 2 2)
                         (convolution-2d-layer 32 32 3 3 :activation :relu)
                         (maxpool-2d-layer 2 2)
                         (flatten-layer)
                         (functional-layer
                          (lambda (x &key (trainp t))
                           ($dropout x trainp 0.4)))
                         (affine-layer (* 32 58 58) 128 :activation :relu)
                         (affine-layer 128 1 :activation :sigmoid)))

(defun network (x &optional (trainp t)) ($execute *network* x :trainp trainp))

($reset! *network*)
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
                           (opt! *network*)))
               (when (zerop (rem epoch 5))
                 (let* ((res (network *test-data* nil))
                        (fres (tensor.float ($ge res 0.5)))
                        (d ($- fres *test-labels*)))
                   (prn "TEST ERROR:" (/ ($dot d d) 1000))
                   ($cg! *network*)))))))

;; train check
(let* ((idx (random *train-size*))
       (data (nth idx *train-data*))
       (lbl (nth idx *train-labels*))
       (y (network data nil))
       (res (tensor.float ($ge y 0.5)))
       (d ($- res lbl)))
  (prn "TRAIN IDX:" idx "ERROR:" (/ ($dot d d) *batch-size*))
  (gcf))

;; test check
(let* ((res (network *test-data* nil))
       (fres (tensor.float ($ge res 0.5)))
       (d ($- fres *test-labels*)))
  (prn "TEST ERROR:" (/ ($dot d d) 1000))
  (gcf))
