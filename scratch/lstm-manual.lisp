(defpackage :lstm-manual
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers))

(in-package :lstm-manual)

;; XXX
;; to compare multiple mm vs one large mm

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
      (b (rndn 100 400)))
  (gcf)
  (time
   (loop :repeat 10000
         :do (let ((c ($mm a b)))
               ($narrow c 1 0 100)
               ($narrow c 1 100 100)
               ($narrow c 1 200 100)
               ($narrow c 1 300 100)))))
