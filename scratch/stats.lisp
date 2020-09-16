(defpackage :stats
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :stats)

;; https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_PyMC2.ipynb

(defparameter *sms-data* (->> (slurp "./data/sms.txt")
                              (mapcar #'parse-float)
                              (mapcar #'round)))
(defparameter *N* ($count *sms-data*))

(defun mean (vs) (/ (reduce #'+ vs) ($count vs)))

(defparameter *alpha* (/ 1D0 (mean *sms-data*)))

(random/exponential 0.5)
(random/exponential (/ 1 10))

(let ((l 2))
  (mean (loop :repeat 1000 :collect (random/exponential (/ 1 l)))))

;; XXX following functions should have derivative function implementations
($lgamma (ones 10))
($gamma (ones 10))
($beta (ones 10) (ones 10))
($lbeta (ones 10) (ones 10))
($polygamma (ones 10) 0)
($erf (ones 10))
($erfc (ones 10))

($gamma (tensor '(1/3)))
($gamma (tensor '(4.8)))

($gamma (tensor '(6)))
($gamma (tensor '(5)))
($/ ($- ($gamma (tensor (list (+ 5 1E-5)))) ($gamma (tensor '(5)))) 1E-5)
($* ($gamma (tensor '(5))) ($polygamma (tensor '(5)) 0))

($/ ($/ ($- ($gamma (tensor (list (+ 5 1E-5)))) ($gamma (tensor '(5)))) 1E-5) 24)


($lgamma (tensor.double (list (+ 5 1E-6))))
($lgamma (tensor.double '(5)))
($- ($lgamma (tensor.double (list (+ 5 1E-6)))) ($lgamma (tensor.double '(5))))
($polygamma (tensor '(5)) 0)
($/ ($- ($lgamma (tensor (list (+ 5 1E-4)))) ($lgamma (tensor '(5)))) 1E-4)

($scalar ($erf (tensor.double '(-1))))

(let* ((x (tensor '((2 1) (1 2))))
       (u ($cholesky x)))
  ($sum ($abs ($- x ($@ ($transpose u) u)))))

;; XXX
;; trying to follow the design of webppl
;;

;; parameters could be set by the user, for example set mu as optimizable parameter but
;; not sigma. so following code should be modified

;; more generic $data equivalent method is required for this.
;; param -> $data, otherwise, just itself

(defgeneric $sample (distribution &optional n))
(defgeneric $score (distribution data))

(defclass distribution () ())

(defmethod $parameters ((d distribution)) '())

(defun pv (pv)
  (if ($parameterp pv)
      ($data pv)
      pv))

(defclass distribution/bernoulli (distribution)
  ((p :initform 0.5)))

(defun distribution/bernoulli (&optional (p 0.5D0))
  (let ((dist (make-instance 'distribution/bernoulli))
        (pin p))
    (with-slots (p) dist
      (setf p pin))
    dist))

(defmethod $parameters ((d distribution/bernoulli))
  (with-slots (p) d
    (if ($parameterp p)
        (list p)
        '())))

(defmethod $sample ((d distribution/bernoulli) &optional (n 1))
  (when (> n 0)
    (with-slots (p) d
      (cond ((eq n 1) (random/bernoulli (pv p)))
            (T ($bernoulli (tensor.byte n) (pv p)))))))

(defmethod $score ((d distribution/bernoulli) (data number))
  (with-slots (p) d
    (if (> data 0)
        ($log p)
        ($log ($sub 1 p)))))

(defmethod $score ((d distribution/bernoulli) (data list))
  (with-slots (p) d
    (let ((nd ($count data))
          (nt 0))
      (loop :for d :in data :do (when (> d 0) (incf nt)))
      ($add ($mul nt ($log p)) ($mul (- nd nt) ($log ($sub 1 p)))))))

(defmethod $score ((d distribution/bernoulli) (data th::tensor))
  (with-slots (p) d
    (let ((nd ($count data))
          (nt ($count ($nonzero data))))
      ($add ($mul nt ($log p)) ($mul (- nd nt) ($log ($sub 1 p)))))))

(defclass distribution/gaussian (distribution)
  ((mu :initform 0)
   (sigma :initform 1)))

(defun distribution/gaussian (&optional (mean 0) (stddev 1))
  (let ((dist (make-instance 'distribution/gaussian)))
    (with-slots (mu sigma) dist
      (setf mu mean
            sigma stddev))
    dist))

(defun distribution/normal (&optional (mean 0) (stddev 1))
  (distribution/gaussian mean stddev))

(defmethod $parameters ((d distribution/gaussian))
  (with-slots (mu sigma) d
    (let ((ps '()))
      (when ($parameterp sigma) (push sigma ps))
      (when ($parameterp mu) (push mu ps))
      ps)))

(defmethod $sample ((d distribution/gaussian) &optional (n 1))
  (when (> n 0)
    (with-slots (mu sigma) d
      (cond ((eq n 1) (random/normal (pv mu) (pv sigma)))
            (T ($normal (tensor n) (pv mu) (pv sigma)))))))

(defmethod $score ((d distribution/gaussian) (data number))
  (with-slots (mu sigma) d
    ($mul -1/2
          ($add ($log (* 2 pi))
                ($add ($mul 2 ($log sigma))
                      ($div ($square ($sub data mu))
                            ($square sigma)))))))

(defmethod $score ((d distribution/gaussian) (data list))
  (with-slots (mu sigma) d
    ($sum ($mul -1/2
                ($add ($log (* 2 pi))
                      ($add ($mul 2 ($log sigma))
                            ($div ($square ($sub (tensor data) mu))
                                  ($square sigma))))))))

(defmethod $score ((d distribution/gaussian) (data th::tensor))
  (with-slots (mu sigma) d
    ($sum ($mul -1/2
                ($add ($log (* 2 pi))
                      ($add ($mul 2 ($log sigma))
                            ($div ($square ($sub data mu))
                                  ($square sigma))))))))

(let ((d (distribution/bernoulli 0.8)))
  ($score d ($sample d 100)))

(let ((original (distribution/bernoulli 0.8)))
  (let ((data ($sample original 1000)))
    (let ((guess (distribution/bernoulli ($parameter 0.5))))
      (prn "LOSS[0]" ($neg ($score guess data)))
      ($amgd! ($parameters guess) 0.01)
      (loop :repeat 5000
            :for i :from 1
            :for l = ($neg ($score guess data))
            :do (progn
                  (when (zerop (rem i 100)) (prn "LOSS" i l))
                  ($amgd! ($parameters guess) 0.01)))
      (prn ($parameters guess)))))

(let ((d (distribution/normal)))
  ($score d ($sample d 10)))

(let ((original (distribution/normal 1 2)))
  (let ((data ($sample original 1000)))
    (let ((guess (distribution/gaussian ($parameter 0) ($parameter 1))))
      (prn "LOSS[0]:" ($neg ($score guess data)))
      ($amgd! ($parameters guess) 0.01)
      (loop :repeat 5000
            :for i :from 1
            :for l = ($neg ($score guess data))
            :do (progn
                  (when (zerop (rem i 100)) (prn "LOSS" i l))
                  ($amgd! ($parameters guess) 0.01)))
      (prn ($parameters guess))
      (prn ($mean data) ($sd data)))))
