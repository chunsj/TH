(defpackage :mcmc-work
  (:use #:common-lisp
        #:mu
        #:th
        #:mr))

(in-package :mcmc-work)

(defun model1 (n) (tensor (rnorm n :mean 10 :sd 3)))

(defparameter *population* (model1 30000))
(defparameter *observation* (tensor (loop :repeat 10000
                                          :collect ($ *population* (random 30000)))))

(defparameter *mu-obs* ($mean *observation*))

(defun transition-model (x)
  (tensor (list ($0 x) (max 1E-7 (random/normal ($1 x) 0.5)))))

(defun prior (x) (if (<= ($1 x) 0) 1E-7 0.9999999))

(defun manual-log-like-normal (x data)
  ($sum ($- ($+ ($log ($* ($1 x) (sqrt (* 2 pi))))
                ($/ ($square ($- data ($0 x)))
                    ($* 2 ($square ($1 x))))))))

(defun acceptance (x xnew)
  (if (> xnew x)
      T
      (let ((accept (random/uniform 0 1)))
        (< ($log accept) ($- xnew x)))))

(defun metropolis-hastings (lfn prior transition pinit iterations data acceptance-rule)
  (let ((x pinit)
        (accepted '())
        (rejected '()))
    (loop :repeat iterations
          :for xnew = (funcall transition x)
          :for xlike = (funcall lfn x data)
          :for xnlike = (funcall lfn xnew data)
          :do (if (funcall acceptance-rule
                           ($+ xlike ($log (funcall prior x)))
                           ($+ xnlike ($log (funcall prior xnew))))
                  (progn
                    (setf x xnew)
                    (push xnew accepted))
                  (push xnew rejected)))
    (list :accepted (coerce (reverse accepted) 'vector)
          :rejected (coerce (reverse rejected) 'vector))))

(defparameter *simulation* (time
                            (metropolis-hastings #'manual-log-like-normal
                                                 #'prior
                                                 #'transition-model
                                                 (tensor (list *mu-obs* 0.1))
                                                 50000
                                                 *observation*
                                                 #'acceptance)))

(prn *simulation*)
(prn ($count (getf *simulation* :accepted)))
(prn ($count (getf *simulation* :rejected)))

(let* ((n ($count (getf *simulation* :accepted)))
       (m (round (/ n 4)))
       (avs (subseq (getf *simulation* :accepted) m))
       (vs (loop :for x :across avs :collect ($1 x)))
       (sum (reduce #'+ vs))
       (sd (* 1D0 (/ sum ($count vs)))))
  sd)
