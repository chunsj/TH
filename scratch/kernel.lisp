(defpackage :kernels
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :kernels)

(defparameter *xs* ($reshape! (range -5 5 1.11111) 10 1))

(defun cov (xs1 xs2 &key (h 0.1))
  (when (eq ($size xs1 0) ($size xs2 0))
    (let* ((n0 ($size xs1 0))
           (diff ($- ($mm xs1 (ones 1 n0))
                     ($mm (ones n0 1) ($transpose xs2)))))
      ($exp ($* -0.5 ($square ($/ diff h)))))))

(prn (cov *xs* *xs*))
