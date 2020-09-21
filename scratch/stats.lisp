(defpackage :stats
  (:use #:common-lisp
        #:mu
        #:th
        #:th.distributions))

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

(time
 (let ((l 2))
   (mean (loop :repeat 1000 :collect (random/exponential (/ 1 l))))))

(time ($mean ($exponential (tensor 1000) (/ 1 2))))

;; XXX at least sampling method and pdf method should be provided.

;; XXX following functions should have derivative function implementations
($lgammaf (ones 10))
($gammaf (ones 10))
($betaf (ones 10) (ones 10))
($lbetaf (ones 10) (ones 10))
($polygamma (ones 10) 0)
($erf (ones 10))
($erfc (ones 10))

($gammaf (tensor '(1/3)))
($gammaf (tensor '(4.8)))

($gammaf (tensor '(6)))
($gammaf (tensor '(5)))
($/ ($- ($gammaf (tensor (list (+ 5 1E-5)))) ($gammaf (tensor '(5)))) 1E-5)
($* ($gammaf (tensor '(5))) ($polygamma (tensor '(5)) 0))

($/ ($/ ($- ($gammaf (tensor (list (+ 5 1E-5)))) ($gammaf (tensor '(5)))) 1E-5) 24)


($lgammaf (tensor.double (list (+ 5 1E-6))))
($lgammaf (tensor.double '(5)))
($- ($lgammaf (tensor.double (list (+ 5 1E-6)))) ($lgammaf (tensor.double '(5))))
($polygamma (tensor '(5)) 0)
($/ ($- ($lgammaf (tensor (list (+ 5 1E-4)))) ($lgammaf (tensor '(5)))) 1E-4)

($scalar ($erf (tensor.double '(-1))))

(let* ((x (tensor '((2 1) (1 2))))
       (u ($cholesky x)))
  ($sum ($abs ($- x ($@ ($transpose u) u)))))

;; XXX
;; trying to follow the design of webppl
;;
;; refer pyro code under the refs directory as well

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

($parameter)

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

(random/beta 1 2)
(random/gamma 1 1)

(random/binomial 10 0.5)
($mean ($binomial (tensor 10) 10 0.5))

($beta (tensor 10) 1 1)
($gamma (tensor 100) 1 1)

($uniform (tensor 10) 0 1)

;; hypergeometric
;; logistic distribution
(defun rlogis (location scale &optional (generator th::*generator*))
  (let ((u ($uniform generator 0 1)))
    (+ location (* scale (log (/ u (- 1 u)))))))
;; poisson
;; negative binomial
(defun rnbinom (size prob &optional (generator th::*generator*))
  (cond ((eq prob 1) 0)
        (T ($poisson generator ($gamma generator size (/ (- 1 prob) prob))))))
;; negative chisq
(defun rnchisq (df lam &optional (generator th::*generator*))
  (if (zerop lam)
      (cond ((zerop df) 0)
            (T ($gamma generator (/ df 2.0) 2.0)))
      (let ((r ($poisson generator (/ lam 2.0))))
        (when (> r 0)
          (setf f ($chisq generator (* 2.0 r))))
        (when (> df 0)
          (incf r ($gamma (/ df 2.0) 2.0)))
        r)))
;; t distribution
(defun tdist (df &optional (generator th::*generator*))
  (let ((n ($normal generator 0 1)))
    (/ n (sqrt ($chisq generator df)) df)))

(defun weibull (shape scale &optional (generator th::*generator*))
  (* scale (expt (- (log ($uniform generator 0 1))) (/ 1.0 shape))))

;; XXX different signature
($multinomial (tensor '(0.1 0.2 0.7)))
