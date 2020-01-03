(defpackage :celeba-data-work
  (:use #:common-lisp
        #:mu
        #:th
        #:th.image))

(in-package :celeba-data-work)

(defparameter +celeba-path+ ($concat (namestring (user-homedir-pathname))
                                     ".th/datasets/celeba/img_align_celeba/"))

;; about ~117MB size of 20 files will be emitted
(loop :for idx :from 1 :to 20
      :do (let* ((nd 10000)
                 (dataset (tensor.byte nd 3 64 64))
                 (rng (loop :for j :from 6 :below 70 :collect j))
                 (offset (* nd (1- idx))))
            (loop :for i :from (+ offset 1) :to (+ nd offset)
                  :for fname = (format nil "~6,'0D.jpg" i)
                  :for tx = ($index (tensor-from-jpeg-file ($concat +celeba-path+ fname)
                                                           :resize-dimension '(78 64)
                                                           :normalize nil)
                                    1
                                    rng)
                  :do (progn
                        (setf ($ dataset (- (1- i) offset)) tx)
                        (setf tx nil)
                        (gcf)))
            (let ((f (file.disk (format nil "celeba64-~2,'0D.tensor" idx) "w")))
              (setf ($fbinaryp f) t)
              ($fwrite dataset f)
              ($fclose f))
            (setf dataset nil)
            (gcf)))
