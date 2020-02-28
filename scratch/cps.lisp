(defpackage :cps-study
  (:use #:common-lisp
        #:mu))

(in-package :cps-study)

(defun factorial-cps (n &optional (k #'values))
  (if (zerop n)
      (funcall k 1)
      (factorial-cps (- n 1)
                     (lambda (v) (funcall k (* v n))))))

(factorial-cps 10)

(defun fmapcar (fun lst &optional (k #'values))
  (if (not lst)
      (funcall k lst)
      (let ((r (funcall fun (car lst))))
        (fmapcar fun
                 (cdr lst)
                 (lambda (x) (funcall k (cons r x)))))))

(fmapcar #'factorial-cps '(0 1 2 3 4 5))
