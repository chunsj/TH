(defpackage :th.ad-test
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :th.ad-test)

(let* ((a (const (tensor '(5 5 5))))
       (c (var 5))
       (out ($broadcast c a))
       (gradient ($backprop out (tensor '(1 1 1)))))
  (unless ($eq ($data out) ($data a))
    (error "out should be the same one as a"))
  (unless (= 3 (-> gradient
                   ($children)
                   ($first)
                   ($gradient)))
    (error "should be 3")))

(let* ((a (const (tensor '(1 0 -3.21))))
       (b (const (tensor '(-1.3 2.8 -0.1)))))
  (unless (< (abs (- 11.3041 ($data ($dot a a)))) 0.0001)
    (error "invalid dot a and a"))
  (unless (< (abs (- -0.979 ($data ($dot a b)))) 0.01)
    (error "invalid dot a and b")))
