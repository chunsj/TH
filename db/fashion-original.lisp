(declaim (optimize (speed 3) (debug 0) (safety 0)))

(defpackage :th.db.fashion-original
  (:use #:common-lisp
        #:mu
        #:th)
  (:export #:read-fashion-train-images
           #:read-fashion-train-labes
           #:read-fashion-t10k-images
           #:read-fashion-t10k-labels
           #:read-fashion-data))

(in-package :th.db.fashion-original)

(defparameter +idx-types+
  '((#x08 (unsigned-byte 8) 1)
    (#x09 (signed-byte 8) 1)
    ;;(#x0B (unsigned-byte 4))
    (#x0C (signed-byte 32) 4)
    (#x0D single-float 4)
    (#x0E double-float 8)))

(defparameter +fmnist-location+ ($concat (namestring (user-homedir-pathname))
                                         ".th/datasets/fashion-mnist"))

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

(defun read-fashion-images (fname &key (normalize nil) (verbose nil))
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

(defun read-fashion-labels (fname &key (verbose nil) (onehot nil))
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

(defun read-fashion-train-images (&key (path +fmnist-location+) (normalize nil) (verbose nil))
  (read-fashion-images (strcat path "/train-images-idx3-ubyte")
                       :normalize normalize :verbose verbose))

(defun read-fashion-train-labels (&key (path +fmnist-location+) (verbose nil) (onehot nil))
  (read-fashion-labels (strcat path "/train-labels-idx1-ubyte")
                       :onehot onehot
                       :verbose verbose))

(defun read-fashion-t10k-images (&key (path +fmnist-location+) (normalize nil) (verbose nil))
  (read-fashion-images (strcat path "/t10k-images-idx3-ubyte")
                       :normalize normalize :verbose verbose))

(defun read-fashion-t10k-labels (&key (path +fmnist-location+) (onehot nil) (verbose nil))
  (read-fashion-labels (strcat path "/t10k-labels-idx1-ubyte")
                       :onehot onehot
                       :verbose verbose))

(defun read-fashion-data (&key (path +fmnist-location+) (normalize T) (onehot T))
  #{:train-images (read-fashion-train-images :path path :normalize normalize)
    :train-labels (read-fashion-train-labels :path path :onehot onehot)
    :test-images (read-fashion-t10k-images :path path :normalize normalize)
    :test-labels (read-fashion-t10k-labels :path path :onehot onehot)})
