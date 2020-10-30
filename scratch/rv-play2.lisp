(defpackage :rv-play2
  (:use #:common-lisp
        #:mu
        #:th
        #:th.distributions))

(in-package :rv-play2)

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

(defun disaster-gof (switch-point early-mean late-mean)
  (let ((ls ($logp switch-point)))
    (when ls
      (let ((disasters-early (subseq *disasters* 0 ($data switch-point)))
            (disasters-late (subseq *disasters* ($data switch-point))))
        (let ((d1 (rv/poisson :rate early-mean :observation disasters-early))
              (d2 (rv/poisson :rate late-mean :observation disasters-late)))
          (let ((ld1 ($logp d1))
                (ld2 ($logp d2)))
            (when (and ls ld1 ld2)
              (list (th.distributions::$gof d1)
                    (th.distributions::$gof d2)))))))))

;; MLE: 41, 3, 1
(let ((switch-point (rv/discrete-uniform :lower 1 :upper (- ($count *disasters*) 2)))
      (early-mean (rv/exponential :rate *rate*))
      (late-mean (rv/exponential :rate *rate*))
      (lfn #'disaster-likelihood)
      (gofn #'disaster-gof))
  (multiple-value-bind (traces deviance)
      (mh (list switch-point early-mean late-mean) lfn
          :iterations 10000
          :thin 5
          :verbose T)
    (loop :for trc :in traces
          :do (prn ($mcmc/summary trc)))
    (prn "AIC" ($mcmc/aic deviance 3))
    (prn "DIC" ($mcmc/dic traces deviance lfn))
    (prn "GOF" (let ((gofs (tensor ($mcmc/collect 1000 traces gofn))))
                 ($mean gofs 0)))))

(let ((acrs '(1.0d0 0.16686886853883606d0 0.05296233695178226d0 0.02513048977104168d0
              -0.026768236953095763d0 -0.009666847281365994d0 -0.01467972172316531d0
              0.022914797557304512d0 0.021270699559253998d0 -3.657343736521337d-4
              -0.02459088733309092d0 -0.007301008988125245d0 -0.005805687306320726d0
              0.031353524721021986d0 -0.022714435743048055d0 0.004213554410337965d0
              0.018790940857874604d0 -0.01914321221664786d0 -9.254498978603177d-4
              0.015455073341770462d0 -0.0034004425214771835d0)))
  (mplot:plot-boxes (loop :for i :from 0
                          :for v :in acrs
                          :collect (cons i v))))

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
      (late-mean (rv/exponential :rate *srate*))
      (lfn #'sms-likelihood))
  (multiple-value-bind (traces deviance)
      (mh (list switch-point early-mean late-mean) lfn
          :iterations 10000
          :thin 5
          :verbose T)
    (loop :for trc :in traces
          :do (prn ($mcmc/summary trc)))
    (prn "AIC" ($mcmc/aic deviance 3))
    (prn "DIC" ($mcmc/dic traces deviance lfn))))

(let ((switch-point (rv/discrete-uniform :lower 1 :upper (- ($count *sms*) 2)))
      (early-mean (rv/exponential :rate *srate*))
      (late-mean (rv/exponential :rate *srate*)))
  (let* ((traces (mh (list switch-point early-mean late-mean) #'sms-likelihood
                     :iterations 10000
                     :thin 5
                     :verbose T)))
    (loop :for trc :in traces
          :do (prn ($count trc) ($mcmc/mle trc) ($mcmc/mean trc) ($mcmc/sd trc)))))

(defun histogram (xs &key (nbins 10))
  (let ((xs (sort (copy-list xs) #'<))
        (steps nbins))
    (let* ((Xmin (apply #'min xs))
           (Xmax (apply #'max xs))
           (Xstep (/ (- Xmax Xmin) steps)))
      (loop :for i :from 0 :below steps
            :for minx = (+ Xmin (* i Xstep))
            :for maxx = (+ Xmin (* (1+ i) Xstep))
            :for nx = ($count (filter (lambda (x) (and (>= x minx)
                                                  (< x maxx)))
                                      xs))
            :collect (cons minx nx)))))

(-> (loop :for yr :from 1851
          :for o :in *disasters*
          :collect (cons yr o))
    (mplot:plot-boxes))
