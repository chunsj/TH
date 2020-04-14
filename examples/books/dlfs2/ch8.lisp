(defpackage :dlfs2-ch8
  (:use #:common-lisp
        #:mu
        #:th
        #:th.ex.data
        #:th.layers
        #:th.text))

(in-package :dlfs2-ch8)

;; XXX test computation for attention weight
(time
 (let ((a (tensor '((1 2 3) (4 5 6) (4 5 6) (1 2 3))))
       (b (tensor '((3 2 1) (6 5 4) (6 5 4) (3 2 1)))))
   (prn ($* a b))
   (prn ($sum ($mul a b) 1))
   (let ((h1 ($sum ($mul a b) 1))
         (h2 ($* 2 ($sum ($mul a b) 1)))
         (h3 ($* 1 ($sum ($mul a b) 1))))
     (prn ($concat h1 h2 h3 1))
     (prn ($softmax ($concat h1 h2 h3 1)))
     (prn (reduce (lambda (a b) ($cat a b 1)) (list h1 h2 h3))))))

(let ((a (tensor '((1 2 3 4 5 6 7 8 9 0)
                   (0 1 2 3 4 5 6 7 8 9)
                   (9 0 1 2 3 4 5 6 7 8)
                   (8 9 0 1 2 3 4 5 6 7)
                   (7 8 9 0 1 2 3 4 5 6)))))
  (prn a)
  (prn ($ a 0))
  (prn ($ a '(0 1) '(0 10)))
  (prn ($ a '((0 1) (0 10))))
  (prn ($ a '(0 5) '(0 1)))
  (prn ($ a '((0 5) (0 1))))
  (prn ($mm ($ a '((0 5) (0 1))) (ones 1 3))))
