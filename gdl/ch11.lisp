(defpackage :gdl-ch11
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :gdl-ch11)

;; onehots
(defparameter *onehots* #{})

(setf ($ *onehots* "cat") (tensor '(1 0 0 0)))
(setf ($ *onehots* "the") (tensor '(0 1 0 0)))
(setf ($ *onehots* "dog") (tensor '(0 0 1 0)))
(setf ($ *onehots* "sat") (tensor '(0 0 0 1)))

(defun word2hot (w) ($ *onehots* w))

(let ((sentence '("the" "cat" "sat")))
  (print (reduce #'$+ (mapcar #'word2hot sentence))))

;; to implement efficient embedding layer, we need row/column selection
;; which is possible by using $index function
(let ((w (tensor '((1 2 3) (2 3 4) (3 4 5) (4 5 6) (5 6 7) (6 7 8) (7 8 9)))))
  (print ($index w 0 (tensor.long '(0 1 4))))
  (print ($sum ($index w 0 (tensor.long '(0 1 4))) 0))
  (print w))

;; compare multiplication and embedding layer shortcut, not that fast
(let ((x (tensor '((1 1 0 1))))
      (w (tensor '((1 2 3) (2 3 4) (3 4 5) (4 5 6)))))
  (print (time ($mm x w)))
  (print (time ($sum ($index w 0 '(0 1 3)) 0)))
  (print ($index ($nonzero x) 1 '(1)))
  (print (time ($sum ($index w 0 ($reshape ($index ($nonzero x) 1 '(1)) 3)) 0))))
