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
(let ((switch-point (rv/discrete-uniform :lower 0 :upper (1- ($count *disasters*))))
      (early-mean (rv/exponential :rate *rate*))
      (late-mean (rv/exponential :rate *rate*)))
  (let* ((accepted (mh 10000 (list switch-point early-mean late-mean) #'disaster-likelihood
                       :verbose T))
         (na ($count accepted))
         (ns (round (* 0.2 na)))
         (selected (subseq accepted 0 ns)))
    (prn "SELECTED:" ns "/" na)
    (let ((ss (mapcar (lambda (ps) ($data ($0 ps))) selected))
          (es (mapcar (lambda (ps) ($data ($1 ps))) selected))
          (ls (mapcar (lambda (ps) ($data ($2 ps))) selected)))
      (prn "MEAN/SD[0]:" (round ($mean ss)) "/" (format nil "~8F" ($sd ss)))
      (prn "MEAN/SD[1]:" (round ($mean es)) "/" (format nil "~8F" ($sd es)))
      (prn "MEAN/SD[2]:" (round ($mean ls)) "/" (format nil "~8F" ($sd ls))))))

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
(let ((switch-point (rv/discrete-uniform :lower 0 :upper (1- ($count *sms*))))
      (early-mean (rv/exponential :rate *srate*))
      (late-mean (rv/exponential :rate *srate*)))
  (let* ((accepted (mh 10000 (list switch-point early-mean late-mean) #'sms-likelihood
                       :verbose T))
         (na ($count accepted))
         (ns (round (* 0.2 na)))
         (selected (subseq accepted 0 ns)))
    (prn "SELECTED:" ns "/" na)
    (let ((ss (mapcar (lambda (ps) ($data ($0 ps))) selected))
          (es (mapcar (lambda (ps) ($data ($1 ps))) selected))
          (ls (mapcar (lambda (ps) ($data ($2 ps))) selected)))
      (prn "MEAN/SD[0]:" (round ($mean ss)) "/" (format nil "~8F" ($sd ss)))
      (prn "MEAN/SD[1]:" (round ($mean es)) "/" (format nil "~8F" ($sd es)))
      (prn "MEAN/SD[2]:" (round ($mean ls)) "/" (format nil "~8F" ($sd ls))))))
