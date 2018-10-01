(declaim (optimize (speed 3) (debug 1) (safety 0)))

(defpackage :th.db.mnist
  (:use #:common-lisp
        #:mu
        #:th)
  (:export #:read-mnist-train-images
           #:read-mnist-train-labels
           #:read-mnist-t10k-images
           #:read-mnist-t10k-labels
           #:read-mnist-data))

(in-package :th.db.mnist)

(defparameter +idx-types+
  '((#x08 (unsigned-byte 8) 1)
    (#x09 (signed-byte 8) 1)
    ;;(#x0B (unsigned-byte 4))
    (#x0C (signed-byte 32) 4)
    (#x0D single-float 4)
    (#x0E double-float 8)))

(defparameter +mnist-location+ ($concat (namestring (user-homedir-pathname)) ".th/datasets/mnist"))

(defun read-nbyte (n str)
  (let ((ret 0))
    (loop :repeat n :do (setf ret (logior (ash ret 8) (read-byte str))))
    ret))

(defun read-single-image-into-m (m idx s nrow ncol &optional (normalize nil))
  (let* ((sz (* nrow ncol)))
    (dotimes (i sz)
      (let* ((v (read-byte s))
             (rv (if normalize (/ v 255.0) (* 1.0 v))))
        (setf ($ m idx i) rv)))))

(defun read-single-label-into-m (m idx byte onehot?)
  (if onehot?
      (setf ($ m idx byte) 1.0)
      (setf ($ m idx 0) (coerce byte 'single-float))))

(defun read-mnist-images (fname &key (normalize nil) (verbose nil))
  (with-open-file (str fname :element-type '(unsigned-byte 8))
    (assert (loop :repeat 2 :always (= #x00 (read-byte str)))
            nil
            "magic numbers not matched")
    (let* ((type-tag (read-byte str))
           (tagdata (cdr (assoc type-tag +idx-types+)))
           (dtype (car tagdata))
           (nbytes (cadr tagdata))
           (metadata (loop :repeat (read-byte str) :collect (read-nbyte 4 str)))
           (ndata (car metadata))
           (nrow (cadr metadata))
           (ncol (caddr metadata))
           (m (zeros ndata (* nrow ncol))))
      (when verbose
        (format T "~%TYPE: ~A NBYTES: ~A~%" dtype nbytes)
        (format T "NDATA: ~A NROW: ~A NCOL: ~A~%" ndata nrow ncol))
      (loop :for i :from 0 :below ndata
            :do (read-single-image-into-m m i str nrow ncol normalize))
      m)))

(defun read-mnist-labels (fname &key (verbose nil) (onehot nil))
  (with-open-file (str fname :element-type '(unsigned-byte 8))
    (assert (loop :repeat 2 :always (= #x00 (read-byte str)))
            nil
            "magic numbers not matched")
    (let* ((type-tag (read-byte str))
           (tagdata (cdr (assoc type-tag +idx-types+)))
           (dtype (car tagdata))
           (nbytes (cadr tagdata))
           (metadata (loop :repeat (read-byte str) :collect (read-nbyte 4 str)))
           (ndata (car metadata))
           (m (if onehot (zeros ndata 10) (zeros ndata 1))))
      (when verbose
        (format T "~%TYPE: ~A NBYTES: ~A~%" dtype nbytes)
        (format T "NDATA: ~A~%" ndata))
      (loop :for i :from 0 :below ndata
            :do (read-single-label-into-m m i (read-byte str) onehot))
      m)))

(defun read-mnist-train-images (&key (path +mnist-location+) (normalize nil) (verbose nil))
  (read-mnist-images (strcat path "/train-images-idx3-ubyte")
                     :normalize normalize :verbose verbose))

(defun read-mnist-train-labels (&key (path +mnist-location+) (verbose nil) (onehot nil))
  (read-mnist-labels (strcat path "/train-labels-idx1-ubyte")
                     :onehot onehot
                     :verbose verbose))

(defun read-mnist-t10k-images (&key (path +mnist-location+) (normalize nil) (verbose nil))
  (read-mnist-images (strcat path "/t10k-images-idx3-ubyte")
                     :normalize normalize :verbose verbose))

(defun read-mnist-t10k-labels (&key (path +mnist-location+) (onehot nil) (verbose nil))
  (read-mnist-labels (strcat path "/t10k-labels-idx1-ubyte")
                     :onehot onehot
                     :verbose verbose))

(defun read-mnist-data (&key (path +mnist-location+) (normalize T) (onehot T))
  #{:train-images (read-mnist-train-images :path path :normalize normalize)
    :train-labels (read-mnist-train-labels :path path :onehot onehot)
    :test-images (read-mnist-t10k-images :path path :normalize normalize)
    :test-labels (read-mnist-t10k-labels :path path :onehot onehot)})
