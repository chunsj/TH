(declaim (optimize (speed 3) (debug 0) (safety 0)))

(defpackage :th.db.celeba
  (:use #:common-lisp
        #:mu
        #:th)
  (:export #:read-celeba64-data))

(in-package :th.db.celeba)

(defparameter +celeba-location+ ($concat (namestring (user-homedir-pathname)) ".th/datasets/celeba"))

(defun read-celeba64-tensor (idx &key (loc +celeba-location+) (normalize T))
  (when (and (>= idx 1) (<= idx 20))
    (let ((f (file.disk (strcat loc (format nil "/tensors/celeba64-~2,'0D.tensor" idx)) "r"))
          (m (tensor.byte)))
      (setf ($fbinaryp f) t)
      ($fread m f)
      ($fclose f)
      (if normalize
          ($div! (tensor.float m) 255)
          (tensor.float m)))))

(defun read-celeba64-data (&key (indices '(1)) (loc +celeba-location+) (normalize T))
  (when indices
    (if (eq 1 ($count indices))
        (read-celeba64-tensor (car indices) :loc loc :normalize normalize)
        (apply #'$concat (append (mapcar (lambda (idx)
                                           (read-celeba64-tensor idx :loc loc :normalize normalize))
                                         indices)
                                 '(0))))))
