(declaim (optimize (speed 3) (debug 1) (safety 0)))

(defpackage :th.db.fashion
  (:use #:common-lisp
        #:mu
        #:th)
  (:export #:read-fashion-data))

(in-package :th.db.fashion)

(defparameter +fashion-location+ ($concat (namestring (user-homedir-pathname))
                                          ".th/datasets/fashion-mnist"))

(defun mfn (n &key (loc +fashion-location+)) (strcat loc "/" n))

(defun generate-fashion-data (&key (loc +fashion-location+))
  (let ((orig-fashion (th.db.fashion-original::read-fashion-data :normalize nil :onehot nil)))
    (let ((train-images ($ orig-fashion :train-images))
          (train-labels ($ orig-fashion :train-labels))
          (test-images ($ orig-fashion :test-images))
          (test-labels ($ orig-fashion :test-labels)))
      (let ((f (file.disk (mfn "fashion-train-images.tensor" :loc loc) "w")))
        (setf ($fbinaryp f) t)
        ($fwrite (tensor.byte train-images) f)
        ($fclose f))
      (let ((f (file.disk (mfn "fashion-train-labels.tensor" :loc loc) "w")))
        (setf ($fbinaryp f) t)
        ($fwrite (tensor.byte train-labels) f)
        ($fclose f))
      (let ((f (file.disk (mfn "fashion-test-images.tensor" :loc loc) "w")))
        (setf ($fbinaryp f) t)
        ($fwrite (tensor.byte test-images) f)
        ($fclose f))
      (let ((f (file.disk (mfn "fashion-test-labels.tensor" :loc loc) "w")))
        (setf ($fbinaryp f) t)
        ($fwrite (tensor.byte test-labels) f)
        ($fclose f)))))

(defun read-fashion-images-tensor (n &key (loc +fashion-location+) (normalize T))
  (let ((f (file.disk (mfn n :loc loc) "r"))
        (m (tensor.byte)))
    (setf ($fbinaryp f) t)
    ($fread m f)
    ($fclose f)
    (if normalize
        ($div! (tensor.float m) 255)
        (tensor.float m))))

(defun read-fashion-labels-tensor (n &key (loc +fashion-location+) (onehot T))
  (let ((f (file.disk (mfn n :loc loc) "r"))
        (m (tensor.byte)))
    (setf ($fbinaryp f) t)
    ($fread m f)
    ($fclose f)
    (if onehot
        (let ((z (zeros ($size m 0) 10)))
          (loop :for i :from 0 :below ($size m 0)
                :do (setf ($ z i ($ m i 0)) 1))
          z)
        (tensor.float m))))

(defun read-fashion-data (&key (path +fashion-location+) (normalize T) (onehot T))
  #{:train-images (read-fashion-images-tensor "fashion-train-images.tensor"
                   :loc path :normalize normalize)
    :train-labels (read-fashion-labels-tensor "fashion-train-labels.tensor"
                   :loc path :onehot onehot)
    :test-images (read-fashion-images-tensor "fashion-test-images.tensor"
                  :loc path :normalize normalize)
    :test-labels (read-fashion-labels-tensor "fashion-test-labels.tensor"
                  :loc path :onehot onehot)})
