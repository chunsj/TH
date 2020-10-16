(defpackage :random-variable
  (:use #:common-lisp
        #:mu
        #:th
        #:th.distributions
        #:th.mcmc))

(in-package :random-variable)

(defparameter *sms-data* (->> (slurp "./data/sms.txt")
                              (mapcar #'parse-float)
                              (mapcar #'round)))
(defparameter *N* ($count *sms-data*))

(defun mean (vs) (/ (reduce #'+ vs) ($count vs)))
(defparameter *mean* (* 1D0 (mean *sms-data*)))
(defparameter *alpha* (/ 1D0 *mean*))

;; SMS example
;; r1/r2/tau = 18/23/45 from the book
;;

;; r1 -> p1
;; r2 -> p2
;; tau

;; how can i process proposal distribution in r1, r2 and tau?
;; how/where should their probability ratio be processed?
(defun sms-likelihood (r1 r2 tau)
  (let* ((idx (1- ($value tau)))
         (d1 (subseq *sms-data* 0 idx))
         (d2 (subseq *sms-data* idx)))
    (let ((p1 (rv/poisson :rate r1 :observation d1))
          (p2 (rv/poisson :rate r2 :observation d2)))
      (+ ($logp p1) ($logp p2) ($logp tau)))))

(defun proposal/gamma (v &optional (scale 1D0))
  (let ((nv ($sample/gamma 1 v scale)))
    (list :new nv
          :old v
          :lh (- ($ll/gamma v nv scale) ($ll/gamma nv v scale)))))

(defun proposal/dice (v &optional (n 1))
  (let ((dv (- ($sample/dice 1 (1+ (* 2 n))) (1+ n))))
    (list :new (+ v dv)
          :old v
          :lh 0D0)))

(defun proposal/gaussian (v &optional (scale 1D0))
  (let ((nv ($sample/gaussian 1 v scale)))
    (list :new nv
          :old v
          :lh 0D0)))

(let ((r1 (rv/exponential :rate *alpha*))
      (r2 (rv/exponential :rate *alpha*))
      (tau (rv/dice :n *N*))
      (flag T))
  (when flag
    (setf ($value r1) *mean*
          ($value r2) *mean*
          ($value tau) (/ *N* 2)))
  (let ((l0 (sms-likelihood r1 r2 tau))
        (accepted nil)
        (proposed nil))
    (loop :repeat 50000
          :do (let ((or1 ($value r1))
                    (or2 ($value r2))
                    (otau ($value tau)))
                (let ((pr1 (proposal/gaussian or1 0.1))
                      (pr2 (proposal/gaussian or2 0.1))
                      (ptau (proposal/gaussian otau 1)))
                  (push (round (getf ptau :new)) proposed)
                  (when (and (> (getf ptau :new) 0) (< (getf ptau :new) *N*)
                             (> (getf pr1 :new) 0) (> (getf pr2 :new) 0))
                    (let ((lh (+ (getf pr1 :lh) (getf pr2 :lh) (getf ptau :lh))))
                      (setf ($value r1) (getf pr1 :new)
                            ($value r2) (getf pr2 :new)
                            ($value tau) (round (getf ptau :new)))
                      (let ((l1 (sms-likelihood r1 r2 tau)))
                        (let ((lmh (+ (- l1 l0) lh))
                              (lu (log (random 1D0))))
                          (if (> lmh lu)
                              (progn
                                (push (list ($value r1) ($value r2) ($value tau))
                                      accepted)
                                (setf l0 l1))
                              (setf ($value r1) or1
                                    ($value r2) or2
                                    ($value tau) otau)))))))))
    (let* ((vs (subseq accepted 0 (min ($count accepted) 20000)))
           (mr1 ($mean (mapcar #'$0 vs)))
           (mr2 ($mean (mapcar #'$1 vs)))
           (mtau ($mean (mapcar #'$2 vs))))
      (list ($count accepted) mr1 mr2 mtau))))

($count (subseq *sms-data* 0 45))
($count (subseq *sms-data* (1- 45)))
(+ 45 29)
(identity *N*)

(let ((r1 (rv/exponential :rate *alpha*))
      (r2 (rv/exponential :rate *alpha*))
      (tau (rv/dice :n *N*))
      (flag T))
  (when flag
    (setf ($value r1) 18
          ($value r2) 22
          ($value tau) 45))
  (sms-likelihood r1 r2 tau))

;; use of sms-likelihood => -495.69xxx
(let ((r1 (rv/exponential :rate *alpha*))
      (r2 (rv/exponential :rate *alpha*))
      (tau (rv/dice :n *N*))
      (flag T))
  (when flag
    (setf ($value r1) 18
          ($value r2) 23
          ($value tau) 45))
  (sms-likelihood r1 r2 tau))

;; logp should be -495.69xxx
(let ((r1 (rv/exponential :rate *alpha*))
      (r2 (rv/exponential :rate *alpha*))
      (tau (rv/dice :n *N*))
      (flag T))
  (when flag
    (setf ($value r1) 18
          ($value r2) 23
          ($value tau) 45))
  (let ((d1 (subseq *sms-data* 0 ($value tau)))
        (d2 (subseq *sms-data* (1- ($value tau)))))
    (let ((p1 (rv/poisson :rate r1 :observation d1))
          (p2 (rv/poisson :rate r2 :observation d2)))
      (+ ($logp p1) ($logp p2) ($logp tau)))))

;; to validate computed logp value => -495.69xxx
(let ((r1 (rv/exponential :rate *alpha* :observation 18))
      (r2 (rv/exponential :rate *alpha* :observation 23))
      (tau (rv/dice :n *N* :observation 45)))
  (let ((lprior (+ ($logp r1) ($logp r2) ($logp tau))))
    (let ((d1 (subseq *sms-data* 0 45))
          (d2 (subseq *sms-data* (1- 45))))
      (let ((p1 (rv/poisson :rate 18 :observation d1))
            (p2 (rv/poisson :rate 23 :observation d2)))
        (let ((llike (+ ($logp p1) ($logp p2))))
          (+ lprior llike))))))

(let ((rv (rv/gaussian))
      (observed (tensor '(0 0 0))))
  (setf ($value rv) observed)
  (prn "VALUE:"($value rv))
  (prn "LOGP:" ($logp rv)))

(prn (rv/exponential))

(let* ((m (rv/exponential))
       (n (rv/normal :location m))
       (obs '(1 1 1)))
  (setf ($value m) 1)
  (prn "VALUE[M]:" ($value m))
  (setf ($value n) obs)
  (prn "VALUE[N]:" ($value n))
  (prn "LOGP:" ($logp n)))

(let ((m (rv/exponential :observation 1))
      (n (rv/normal :location 1))
      (obs '(1 1 1)))
  (setf ($value n) obs)
  (prn "VALUE[N]:" ($value n))
  (prn "LOGP:" ($logp n))
  (prn "VALUE[M]:" ($value m))
  (prn "LOGP-EXP:" ($logp m))
  (prn "SUM-LOGP:" ($add ($logp m) ($logp n))))

(let ((rv (make-instance 'rv/gaussian)))
  (list ($value rv) ($value rv))
  (setf ($value rv) (tensor '(0 0 0.1 -0.1)))
  ($value rv)
  (setf ($value rv) 0)
  ($logp rv))

($ll/normal 0 0 1)

;; XXX OLDER ONE

(defgeneric $val (rv))
(defgeneric $logp (rv))

(defclass random-variable ()
  ((g :initform nil)
   (v :initform nil)))

(defmethod print-object ((rv random-variable) stream)
  (format stream "~A" ($val rv)))

(defun rv (generator)
  (let ((n (make-instance 'random-variable)))
    (with-slots (g v) n
      (setf g generator)
      (setf v ($sample g 1)))
    n))

(defmethod $val ((rv random-variable))
  (with-slots (g v) rv
    (unless v
      (setf v ($sample g 1)))
    v))

(defmethod $reset! ((rv random-variable))
  (with-slots (g v) rv
    (setf v ($sample g 1))
    v))

(defmethod $logp ((rv random-variable))
  (let ((v ($val rv)))
    (with-slots (g) rv
      ($ll g v))))

(rv (distribution/normal))
($val (rv (distribution/normal)))
($reset! (rv (distribution/normal)))
($logp (rv (distribution/normal)))

(defparameter *population* ($sample (distribution/normal 2.5) 5000))
(defparameter *observation* (tensor (loop :repeat 1000
                                          :collect ($ *population* (random 5000)))))

($mean *population*)
($mean *observation*)

(defun potential (position)
  ($- ($ll (distribution/normal position) *observation*)))

(let ((samples (hmc 1000 0 #'potential :step-size 0.05)))
  ;; data mean vs computed parameter which is the mean of the normal distribution
  (list ($mean *population*) ($mean *observation*) ($mean samples)))

;; model
;;
;; parameters
;; distributions - parameters
;; observations - distributions - parameters
;; likelihood - observations - distributions - parameters

;; XXX
;; bare functional log-likelihood and other functions should be in
;; the th.distributions.
;; current class based implementations should depends on them.


(defclass mymodel ()
  ((mu :initform 0)
   (n :initform (distribution/normal 0))))

(defmethod $ll ((m mymodel) data)
  (with-slots (mu n) m
    (setf ($ n :mu) mu)
    ($ll n data)))

(defun $nll (m mu data)
  (let ((muin mu))
    (with-slots (mu) m
      (setf mu muin))
    ($- ($ll m data))))

(defparameter *model* (make-instance 'mymodel))

(defun potential (position) ($nll *model* position *observation*))

(let ((samples (hmc 1000 0 #'potential :step-size 0.05)))
  ;; data mean vs computed parameter which is the mean of the normal distribution
  (list ($mean *population*) ($mean *observation*) ($mean samples)))
