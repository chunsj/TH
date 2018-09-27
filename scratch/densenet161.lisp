(defpackage :th.m.densenet161
  (:use #:common-lisp
        #:mu
        #:th
        #:th.image
        #:th.m.imagenet))

(in-package :th.m.densenet161)

;;(defparameter +model-location+ ($concat (namestring (user-homedir-pathname)) ".th.models"))
(defparameter +model-location+ ($concat (namestring (user-homedir-pathname)) "Desktop"))

(defun read-text-weight-file (wn &optional (readp t))
  (when readp
    (let ((f (file.disk (format nil "~A/densenet161/densenet161-~A.txt" +model-location+ wn) "r"))
          (tx (tensor)))
      ($fread tx f)
      ($fclose f)
      tx)))

(defun read-weight-file (wn &optional (readp t))
  (when readp
    (let ((f (file.disk (format nil "~A/densenet161/densenet161-~A.dat" +model-location+ wn) "r"))
          (tx (tensor)))
      (setf ($fbinaryp f) t)
      ($fread tx f)
      ($fclose f)
      tx)))

(defun kw (str) (values (intern (string-upcase str) "KEYWORD")))

(defun read-densenet161-text-weights ()
  (append (loop :for i :from 0 :to 481
                :for nm = (format nil "p~A" i)
                :for k = (kw nm)
                :append (list k (read-text-weight-file nm)))
          (loop :for i :from 1 :to 161
                :for nm = (format nil "v~A" i)
                :for k = (kw nm)
                :append (list k (read-text-weight-file nm)))
          (loop :for i :from 1 :to 161
                :for nm = (format nil "m~A" i)
                :for k = (kw nm)
                :append (list k (read-text-weight-file nm)))
          (list :fw1 (read-text-weight-file :f482)
                :fb1 (read-text-weight-file :f483))))

(defun w (w wn) (getf w wn))

(w (read-densenet161-text-weights) :p5)

(defun input-blk (x ws)
  (-> x
      ($conv2d (w ws :p0) nil 2 2 3 3)
      ($bn (w ws :p1) (w ws :p2) (w ws :m1) (w ws :v1))
      ($relu)
      ($dlmaxpool2d 3 3 2 2 1 1 1 1)))

(defun kwn (k i) (kw (format nil "~A~A" k i)))

;; p = p + 6, n = n + 2
(defun dense-layer (x ws p n)
  (let ((g1 (w ws (kwn :p p)))
        (b1 (w ws (kwn :p (+ p 1))))
        (m1 (w ws (kwn :m n)))
        (v1 (w ws (kwn :v n)))
        (k1 (w ws (kwn :p (+ p 2))))
        (g2 (w ws (kwn :p (+ p 3))))
        (b2 (w ws (kwn :p (+ p 4))))
        (m2 (w ws (kwn :m (+ n 1))))
        (v2 (w ws (kwn :v (+ n 1))))
        (k2 (w ws (kwn :p (+ p 5)))))
    ($cat x
          (-> x
              ($bn g1 b1 m1 v1)
              ($relu)
              ($conv2d k1 nil 1 1)
              ($bn g2 b2 m2 v2)
              ($relu)
              ($conv2d k2 nil 1 1 1 1))
          1)))

;; p = p + 3, n = n + 1
(defun transition (x ws p n)
  (let ((g1 (w ws (kwn :p p)))
        (b1 (w ws (kwn :p (+ p 1))))
        (m1 (w ws (kwn :m n)))
        (v1 (w ws (kwn :v n)))
        (k1 (w ws (kwn :p (+ p 2)))))
    (-> x
        ($bn g1 b1 m1 v1)
        ($relu)
        ($conv2d k1 nil 1 1)
        ($avgpool2d 2 2 2 2))))

;; out p = 39, n = 14
(defun dense-block1 (x ws)
  (-> x
      (dense-layer ws 3 2)
      (dense-layer ws 9 4)
      (dense-layer ws 15 6)
      (dense-layer ws 21 8)
      (dense-layer ws 27 10)
      (dense-layer ws 33 12)))

(loop :for i :from 42 :below 120 :by 6 :collect i)

;; out p = 114, n = 39
(defun dense-block2 (x ws)
  (-> x
      (dense-layer ws 42 15)
      (dense-layer ws 48 17)
      (dense-layer ws 54 19)
      (dense-layer ws 60 21)
      (dense-layer ws 66 23)
      (dense-layer ws 72 25)
      (dense-layer ws 78 27)
      (dense-layer ws 84 29)
      (dense-layer ws 90 31)
      (dense-layer ws 96 33)
      (dense-layer ws 102 35)
      (dense-layer ws 108 37)))

;; out p = 333, n = 112
(defun dense-block3 (x ws)
  (-> x
      (dense-layer ws 117 40)
      (dense-layer ws 123 42)
      (dense-layer ws 129 44)
      (dense-layer ws 135 46)
      (dense-layer ws 141 48)
      (dense-layer ws 147 50)
      (dense-layer ws 153 52)
      (dense-layer ws 159 54)
      (dense-layer ws 165 56)
      (dense-layer ws 171 58)
      (dense-layer ws 177 60)
      (dense-layer ws 183 62)
      (dense-layer ws 189 64)
      (dense-layer ws 195 66)
      (dense-layer ws 201 68)
      (dense-layer ws 207 70)
      (dense-layer ws 213 72)
      (dense-layer ws 219 74)
      (dense-layer ws 225 76)
      (dense-layer ws 231 78)
      (dense-layer ws 237 80)
      (dense-layer ws 243 82)
      (dense-layer ws 249 84)
      (dense-layer ws 255 86)
      (dense-layer ws 261 88)
      (dense-layer ws 267 90)
      (dense-layer ws 273 92)
      (dense-layer ws 279 94)
      (dense-layer ws 285 96)
      (dense-layer ws 291 98)
      (dense-layer ws 297 100)
      (dense-layer ws 303 102)
      (dense-layer ws 309 104)
      (dense-layer ws 315 106)
      (dense-layer ws 321 108)
      (dense-layer ws 327 110)))

(+ 117 (* 6 36))
;; 336 113 - after block3 transition

(+ 336 (* 6 24))
(+ 113 (* 2 24))

;; out p = 480, n = 161
(defun dense-block4 (x ws)
  (-> x
      (dense-layer ws 336 113)
      (dense-layer ws 342 115)
      (dense-layer ws 348 117)
      (dense-layer ws 354 119)
      (dense-layer ws 360 121)
      (dense-layer ws 366 123)
      (dense-layer ws 372 125)
      (dense-layer ws 378 127)
      (dense-layer ws 384 129)
      (dense-layer ws 390 131)
      (dense-layer ws 396 133)
      (dense-layer ws 402 135)
      (dense-layer ws 408 137)
      (dense-layer ws 414 139)
      (dense-layer ws 420 141)
      (dense-layer ws 426 143)
      (dense-layer ws 432 145)
      (dense-layer ws 438 147)
      (dense-layer ws 444 149)
      (dense-layer ws 450 151)
      (dense-layer ws 456 153)
      (dense-layer ws 462 155)
      (dense-layer ws 468 157)
      (dense-layer ws 474 159)))

(defparameter *weights* (read-densenet161-text-weights))

(let* ((rgb (tensor-from-png-file "data/cat.vgg16.png"))
       (x (imagenet-input rgb))
       (input (apply #'$reshape x (cons 1 ($size x))))
       (ws *weights*))
  (-> input
      (input-blk ws)
      (dense-block1 ws)
      (transition ws 39 14)
      (dense-block2 ws)
      (transition ws 114 39)
      (dense-block3 ws)
      (transition ws 333 112)
      (dense-block4 ws)
      ($bn (w ws :p480) (w ws :p481) (w ws :m161) (w ws :v161))
      ($avgpool2d 7 7 1 1)
      ($reshape 1 2208)
      ($affine (w ws :fw1) (w ws :fb1))
      ($softmax)
      ($max 1)
      (prn)))
