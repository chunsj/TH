(defpackage :dlfs-ch4
  (:use #:common-lisp
        #:mu
        #:th
        #:th.ex.data
        #:th.text))

(in-package :dlfs-ch4)

;; choice
(prn (loop :for i :from 0 :below 10 :collect (choice '(:a :b :c) '(0.1 0.2 0.7))))

(defun wimb (x w)
  (cond ((atom (car x)) ($wimb x w))
        ((listp (car x)) (apply #'$cat (append (loop :for idcs :in x :collect ($wimb idcs w))
                                               (list 0))))
        (T (error "cannot compute embedding for dim > 2"))))

(let* ((vs (loop :for i :from 0 :below 21 :collect i))
       (w (-> (tensor vs)
              ($reshape! 7 3))))
  (prn ($ w 2))
  (prn ($ w 5))
  (prn ($index w 0 '(1 0 3 0)))
  (prn ($wimb '(1 0 3 0) w))
  (prn (wimb '(1 0 3 0) w))
  (prn (wimb '((1 0 3 0) (1 0 3 0)) w)))

(prn ($cat (tensor '((1 2 3))) (tensor '((4 5 6))) 0))

(defun embed (x h w) ($reshape ($sum ($* ($index w 0 x) h) 1) ($size h 1)))

(let ((w (-> (tensor (loop :for i :from 0 :below 21 :collect i))
             ($reshape! 7 3)))
      (idx '(0 3 1))
      (h (tensor '((0 1 2)
                   (3 4 5)
                   (6 7 8)))))
  (prn ($index w 0 idx))
  (prn ($* ($index w 0 idx) h))
  (prn (embed idx h w)))

;; XXX to be implemented in th
($onehot tensor num-classes)

(let ((w (-> (tensor (loop :for i :from 0 :below 21 :collect i))
             ($reshape! 7 3)))
      (index '(0 3 1))
      (index2 '((0 3 1) (0 3 1))))
  (prn ($index w 0 index))
  (prn ($reshape (tensor.long index2) 6))
  (prn ($view ($index w 0 ($reshape (tensor.long index2) 6)) 2 3 3))
  (prn ($embedding (tensor.long index2) w))
  (prn ($embedding (tensor.long index2) ($parameter w))))

(gcf)
