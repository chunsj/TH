(defpackage :rv-play
  (:use #:common-lisp
        #:mu
        #:th
        #:th.distributions))

(in-package :rv-play)

(defvar *disasters* '(4 5 4 0 1 4 3 4 0 6 3 3 4 0 2 6
                      3 3 5 4 5 3 1 4 4 1 5 5 3 4 2 5
                      2 2 3 4 2 1 3 2 2 1 1 1 1 3 0 0
                      1 0 1 1 0 0 3 1 0 3 2 2 0 1 1 1
                      0 1 0 1 0 0 0 2 1 0 0 0 1 1 0 2
                      3 3 1 1 2 1 1 1 1 2 4 2 0 0 1 4
                      0 0 0 1 0 0 0 0 0 1 0 0 1 0 1))
(defvar *rate* (/ 1D0 ($mean *disasters*)))

(defun disaster-likelihood (switch-point early-mean late-mean)
  (let ((ls ($logp switch-point)))
    (when ls
      (let ((disasters-early (subseq *disasters* 0 ($data switch-point)))
            (disasters-late (subseq *disasters* ($data switch-point))))
        (let ((d1 (rv/poisson :rate early-mean :observation disasters-early))
              (d2 (rv/poisson :rate late-mean :observation disasters-late)))
          (let ((ld1 ($logp d1))
                (ld2 ($logp d2)))
            (when (and ls ld1 ld2)
              (+ ls ld1 ld2))))))))

;; MLE: 41, 3, 1
(let ((switch-point (rv/discrete-uniform :lower 1 :upper (- ($count *disasters*) 2)))
      (early-mean (rv/exponential :rate *rate*))
      (late-mean (rv/exponential :rate *rate*)))
  (let* ((traces (mh (list switch-point early-mean late-mean) #'disaster-likelihood
                     :iterations 10000
                     :thin 5
                     :verbose T)))
    (loop :for trc :in traces
          :do (prn (format nil "~A/~A ~8F ~8F ~8F" ($mcmc/count trc) ($count trc)
                           ($mcmc/mle trc) ($mcmc/mean trc) ($mcmc/sd trc))))))


;; FOR SMS example
;; https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/masterv/Chapter1_Introduction/Ch1_Introduction_PyMC2.ipynb
(defvar *sms* (->> (slurp "./data/sms.txt")
                   (mapcar #'parse-float)
                   (mapcar #'round)))
(defvar *srate* (/ 1D0 ($mean *sms*)))

(defun sms-likelihood (switch-point early-mean late-mean)
  (let ((ls ($logp switch-point)))
    (when ls
      (let ((disasters-early (subseq *sms* 0 ($data switch-point)))
            (disasters-late (subseq *sms* ($data switch-point))))
        (let ((d1 (rv/poisson :rate early-mean :observation disasters-early))
              (d2 (rv/poisson :rate late-mean :observation disasters-late)))
          (let ((ld1 ($logp d1))
                (ld2 ($logp d2)))
            (when (and ls ld1 ld2)
              (+ ls ld1 ld2))))))))

;; MLE: 45, 18, 23
(let ((switch-point (rv/discrete-uniform :lower 1 :upper (- ($count *sms*) 2)))
      (early-mean (rv/exponential :rate *srate*))
      (late-mean (rv/exponential :rate *srate*)))
  (let* ((traces (mh (list switch-point early-mean late-mean) #'sms-likelihood
                     :iterations 10000
                     :thin 5
                     :verbose T)))
    (loop :for trc :in traces
          :do (prn (format nil "~A/~A ~8F ~8F ~8F" ($mcmc/count trc) ($count trc)
                           ($mcmc/mle trc) ($mcmc/mean trc) ($mcmc/sd trc))))))
