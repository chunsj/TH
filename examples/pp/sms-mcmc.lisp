(defpackage :sms-mcmc
  (:use #:common-lisp
        #:mu
        #:th
        #:th.distributions))

(in-package :sms-mcmc)

;; https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_PyMC2.ipynb

;; XXX
;; following code does not work well
;; discrete parameter optimization is not supported yet.
;; XXX

(defparameter *sms-data* (->> (slurp "./data/sms.txt")
                              (mapcar #'parse-float)
                              (mapcar #'round)))
(defparameter *N* ($count *sms-data*))

(defun mean (vs) (/ (reduce #'+ vs) ($count vs)))

(defparameter *alpha* (/ 1D0 (mean *sms-data*)))

(defun loglikelihood (h l1 l2 dt tau)
  (if (and (> tau 0) (< tau *N*))
      (let ((tau (round tau)))
        (let ((d1 (subseq *sms-data* 0 tau))
              (d2 (subseq *sms-data* (1- tau))))
          ($+ ($ll l1 d1) ($ll l2 d2)
              ($ll h ($scalar ($ l1 :l)))
              ($ll h ($scalar ($ l2 :l)))
              ($ll dt tau))))
      most-negative-single-float))

;; XXX needs implementation for parameter data
($ll (distribution/exponential *alpha*) ($parameter 10))

;; the solution from the original site
(let ((r1 18)
      (r2 23)
      (tau 45)
      (he (distribution/exponential *alpha*))
      (dt (distribution/discrete ($softmax (tensor (loop :repeat *N* :collect 1))))))
  (let ((l1 (distribution/poisson r1))
        (l2 (distribution/poisson r2)))
    (prn (loglikelihood he l1 l2 dt tau))))

;; MCMC
(time
 (let ((he (distribution/exponential *alpha*))
       (dt (distribution/discrete ($softmax (tensor (loop :repeat *N* :collect 1)))))
       (N 100000)
       (Nb 20000)
       (accepted '())
       (rejected '()))
   (let* ((r1 ($sample he))
          (r2 ($sample he))
          (tm (random *N*))
          (oldlk (loglikelihood he
                                (distribution/poisson r1)
                                (distribution/poisson r2)
                                dt
                                tm)))
     (loop :repeat (+ N Nb)
           :for i :from 1
           :for nr1 = (random/normal r1 0.1)
           :for nr2 = (random/normal r2 0.1)
           :for ntm = (let* ((d 1)
                             (nn (1+ (* 2 d)))
                             (rn (random nn))
                             (dl (max 1 (- tm d)))
                             (dr (min (1- *N*) (+ tm d)))
                             (nt tm))
                        (setf nt (+ dl rn))
                        (when (> nt (1- *N*))
                          (setf nt (- dr rn)))
                        nt)
           :do (if (and (> nr1 0) (> nr2 0))
                   (let* ((newlk (loglikelihood he
                                                (distribution/poisson nr1)
                                                (distribution/poisson nr2)
                                                dt
                                                ntm))
                          (dl (- ($scalar newlk) ($scalar oldlk)))
                          (r (log (random 1D0))))
                     (if (< r dl)
                         (progn
                           (when (>= i Nb) (push (list nr1 nr2 ntm) accepted))
                           (setf oldlk newlk
                                 r1 nr1
                                 r2 nr2
                                 tm ntm))
                         (when (>= i Nb) (push (list nr1 nr2 ntm) rejected))))
                   (when (>= i Nb) (push (list nr1 nr2 ntm) rejected))))
     (when (> ($count accepted) 0)
       (let* ((na ($count accepted))
              (nr ($count rejected))
              (fr1 (round (mean (mapcar #'$0 accepted))))
              (sdfr1 ($sd (tensor (mapcar #'$0 accepted))))
              (fr2 (round (mean (mapcar #'$1 accepted))))
              (sdfr2 ($sd (tensor (mapcar #'$1 accepted))))
              (ftau (round (mean (mapcar #'$2 accepted))))
              (sdftau ($sd (tensor (mapcar #'$2 accepted)))))
         (prn "ACCEPTED:" na "/" "REJECTED:" nr)
         (prn "R1:" fr1 "/" sdfr1)
         (prn "R2:" fr2 "/" sdfr2)
         (prn "TAU:" ftau "/" sdftau)
         (prn "LL:" (loglikelihood he
                                   (distribution/poisson fr1)
                                   (distribution/poisson fr2)
                                   dt
                                   ftau)))))))

;; XXX need discrete random to find tau
(let ((he (distribution/exponential *alpha*)))
  (let ((r1 ($parameter ($sample he)))
        (r2 ($parameter ($sample he)))
        (ps ($parameter ($softmax (tensor (loop :repeat *N* :collect 1)))))
        (tau (random *N*)))
    (setf tau 45)
    (let ((l1 (distribution/poisson r1))
          (l2 (distribution/poisson r2))
          (dt (distribution/discrete ps)))
      (loop :repeat 10000
            :for iter :from 1
            :for loss = ($neg (loglikelihood he l1 l2 dt tau))
            :do (progn
                  (when (zerop (rem iter 1000)) (prn iter loss))
                  ($amgd! (append ($parameters l1) ($parameters l2) ($parameters dt))
                          0.01)))
      (prn "R1:" (round ($scalar ($ l1 :l))))
      (prn "R2:" (round ($scalar ($ l2 :l))))
      (prn ($argmax ($data ($ dt :ps)))))))
