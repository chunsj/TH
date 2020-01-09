(declaim (optimize (speed 3) (debug 0) (safety 0)))

(defpackage :th.db.cats-and-dogs
  (:use #:common-lisp
        #:mu
        #:th)
  (:export #:read-cats-and-dogs-data))

(in-package :th.db.cats-and-dogs)

(defparameter +cnd-location+ ($concat (namestring (user-homedir-pathname))
                                      ".th/datasets/cats-and-dogs/tensors"))

(defun read-cats-and-dogs-64-train-tensor (idx &key (loc +cnd-location+) (normalize T))
  (when (and (>= idx 1) (<= idx 25))
    (let ((fc (file.disk (strcat loc (format nil "/cat64-train-~2,'0D.tensor" idx)) "r"))
          (fd (file.disk (strcat loc (format nil "/dog64-train-~2,'0D.tensor" idx)) "r"))
          (mc (tensor.byte))
          (md (tensor.byte)))
      (setf ($fbinaryp fc) t)
      (setf ($fbinaryp fd) t)
      ($fread mc fc)
      ($fread md fd)
      ($fclose fc)
      ($fclose fd)
      (let ((m (apply #'tensor.byte (append (list (* 2 ($size mc 0))) (cdr ($size mc)))))
            (n ($size mc 0)))
        (loop :for i :from 0 :below n
              :for idx =  (* i 2)
              :do (setf ($ m idx) ($ mc i)
                        ($ m (1+ idx)) ($ md i)))
        (if normalize
            ($div! (tensor.float m) 255)
            (tensor.float m))))))

(defun read-cats-and-dogs-data (&key (indices '(1)) (loc +cnd-location+) (normalize T))
  (when indices
    (if (eq 1 ($count indices))
        (read-cats-and-dogs-64-train-tensor (car indices) :loc loc :normalize normalize)
        (apply #'$concat (append (mapcar (lambda (idx)
                                           (read-cats-and-dogs-64-train-tensor idx
                                                                               :loc loc
                                                                               :normalize normalize))
                                         indices)
                                 '(0))))))
