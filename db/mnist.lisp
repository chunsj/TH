(declaim (optimize (speed 3) (debug 1) (safety 0)))

(defpackage :th.db.mnist
  (:use #:common-lisp
        #:mu
        #:th)
  (:export #:read-mnist-data))

(in-package :th.db.mnist)

(defparameter +mnist-location+ ($concat (namestring (user-homedir-pathname)) ".th/datasets/mnist"))

(defun mfn (n &key (loc +mnist-location+)) (strcat loc "/" n))

(defun generate-mnist-data (&key (loc +mnist-location+))
  (let ((orig-mnist (th.db.mnist-original::read-mnist-data :normalize nil :onehot nil)))
    (let ((train-images ($ orig-mnist :train-images))
          (train-labels ($ orig-mnist :train-labels))
          (test-images ($ orig-mnist :test-images))
          (test-labels ($ orig-mnist :test-labels)))
      (let ((f (file.disk (mfn "mnist-train-images.tensor" :loc loc) "w")))
        (setf ($fbinaryp f) t)
        ($fwrite (tensor.byte train-images) f)
        ($fclose f))
      (let ((f (file.disk (mfn "mnist-train-labels.tensor" :loc loc) "w")))
        (setf ($fbinaryp f) t)
        ($fwrite (tensor.byte train-labels) f)
        ($fclose f))
      (let ((f (file.disk (mfn "mnist-test-images.tensor" :loc loc) "w")))
        (setf ($fbinaryp f) t)
        ($fwrite (tensor.byte test-images) f)
        ($fclose f))
      (let ((f (file.disk (mfn "mnist-test-labels.tensor" :loc loc) "w")))
        (setf ($fbinaryp f) t)
        ($fwrite (tensor.byte test-labels) f)
        ($fclose f)))))

(defun read-mnist-images-tensor (n &key (loc +mnist-location+) (normalize T))
  (let ((f (file.disk (mfn n :loc loc) "r"))
        (m (tensor.byte)))
    (setf ($fbinaryp f) t)
    ($fread m f)
    ($fclose f)
    (if normalize
        ($div! (tensor.float m) 255)
        (tensor.float m))))

(defun read-mnist-labels-tensor (n &key (loc +mnist-location+) (onehot T))
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

(defun read-mnist-data (&key (path +mnist-location+) (normalize T) (onehot T))
  #{:train-images (read-mnist-images-tensor "mnist-train-images.tensor"
                   :loc path :normalize normalize)
    :train-labels (read-mnist-labels-tensor "mnist-train-labels.tensor"
                   :loc path :onehot onehot)
    :test-images (read-mnist-images-tensor "mnist-test-images.tensor"
                  :loc path :normalize normalize)
    :test-labels (read-mnist-labels-tensor "mnist-test-labels.tensor"
                  :loc path :onehot onehot)})
