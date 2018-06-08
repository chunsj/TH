(defpackage :la-test
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :la-test)

;; cholesky
(let* ((x (rnd 10 10))
       (a ($mm x ($transpose x))))
  (let* ((c ($potrf a))
         (b ($mm ($transpose c) c)))
    (print ($sum ($sub a b))))
  (let* ((u ($potrf a t))
         (b ($mm ($transpose u) u)))
    (print ($sum ($sub a b))))
  (let* ((l ($potrf a nil))
         (b ($mm l ($transpose l))))
    (print ($sum ($sub a b)))))
