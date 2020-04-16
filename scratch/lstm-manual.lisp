(defpackage :lstm-manual
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers))

(in-package :lstm-manual)

(let ((a (rndn 100 100))
      (b (rndn 100 100)))
  (gcf)
  (time
   (loop :repeat 10000
         :do (progn
               ($mm a b)
               ($mm a b)
               ($mm a b)
               ($mm a b)))))

(let ((a (rndn 100 100))
      (b (rndn 100 400)))
  (gcf)
  (time
   (loop :repeat 10000
         :do ($mm a b))))

(let ((a (rndn 100 100))
      (b (rndn 100 400))
      (s1 (tensor.long (loop :for i :from 0 :below 100 :collect i)))
      (s2 (tensor.long (loop :for i :from 0 :below 100 :collect (+ i 100))))
      (s3 (tensor.long (loop :for i :from 0 :below 100 :collect (+ i 200))))
      (s4 (tensor.long (loop :for i :from 0 :below 100 :collect (+ i 300)))))
  (gcf)
  (time
   (loop :repeat 10000
         :do (let ((c ($mm a b)))
               ($index c 1 s1)
               ($index c 1 s2)
               ($index c 1 s3)
               ($index c 1 s4)))))

(let ((a (rndn 100 100))
      (b (rndn 100 400)))
  (gcf)
  (time
   (loop :repeat 10000
         :do (let ((c ($mm a b)))
               ($narrow c 1 0 100)
               ($narrow c 1 100 100)
               ($narrow c 1 200 100)
               ($narrow c 1 300 100)))))
