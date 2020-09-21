(defpackage :sms-mcmc
  (:use #:common-lisp
        #:mu
        #:th
        #:th.distributions))

(in-package :sms-mcmc)

;; https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_PyMC2.ipynb

(defparameter *sms-data* (->> (slurp "./data/sms.txt")
                              (mapcar #'parse-float)
                              (mapcar #'round)))
(defparameter *N* ($count *sms-data*))

(defun mean (vs) (/ (reduce #'+ vs) ($count vs)))

(defparameter *alpha* (/ 1D0 (mean *sms-data*)))

(defun loglikelihood (l1 l2 tau)
  (if (and (> tau 0) (< tau *N*))
      (let ((tau (round tau)))
        (let ((d1 (subseq *sms-data* 0 tau))
              (d2 (subseq *sms-data* (1- tau))))
          ($+ ($score l1 d1) ($score l2 d2))))
      most-negative-single-float))

;; the solution from the original site
(let ((r1 18)
      (r2 23)
      (tau 45))
  (let ((l1 (distribution/poisson r1))
        (l2 (distribution/poisson r2)))
    (let ((d1 (subseq *sms-data* 0 tau))
          (d2 (subseq *sms-data* tau)))
      ($+ ($score l1 d1) ($score l2 d2)))))

;; MCMC
(time
 (let ((he (distribution/exponential *alpha*))
       (N 50000)
       (Nb 12500)
       (accepted '())
       (rejected '()))
   (let* ((r1 ($sample he))
          (r2 ($sample he))
          (tm (random *N*))
          (oldlk (loglikelihood (distribution/poisson r1)
                                (distribution/poisson r2)
                                tm)))
     (loop :repeat (+ N Nb)
           :for i :from 1
           :for nr1 = (abs (random/normal r1 1))
           :for nr2 = (abs (random/normal r2 1))
           :for ntm = (round (abs (random/normal tm 1)))
           :for newlk = (loglikelihood (distribution/poisson nr1)
                                       (distribution/poisson nr2)
                                       ntm)
           :do (let ((dl (- ($scalar newlk) ($scalar oldlk)))
                     (r (log (random 1D0))))
                 (if (< r dl)
                     (progn
                       (when (>= i Nb) (push (list nr1 nr2 ntm) accepted))
                       (setf oldlk newlk
                             r1 nr1
                             r2 nr2
                             tm ntm))
                     (when (>= i Nb) (push (list nr1 nr2 ntm) rejected)))))
     (let ((na ($count accepted))
           (nr ($count rejected)))
       (prn "ACCEPTED:" na "/" "REJECTED:" nr)
       (prn "R1:" (round (mean (mapcar #'$0 accepted))))
       (prn "R2:" (round (mean (mapcar #'$1 accepted))))
       (prn "TAU:" (round (mean (mapcar #'$2 accepted))))))))

;; XXX need discrete random to find tau
(let ((he (distribution/exponential *alpha*)))
  (let ((r1 ($parameter ($sample he)))
        (r2 ($parameter ($sample he)))
        (tau 45))
    (let ((l1 (distribution/poisson r1))
          (l2 (distribution/poisson r2)))
      (loop :repeat 5000
            :for iter :from 1
            :for loss = ($neg (loglikelihood l1 l2 tau))
            :do (progn
                  (when (zerop (rem iter 1000)) (prn iter loss))
                  ($amgd! (append ($parameters l1) ($parameters l2))
                          0.01)))
      (prn "R1:" (round ($scalar ($ l1 :l))))
      (prn "R2:" (round ($scalar ($ l2 :l)))))))
