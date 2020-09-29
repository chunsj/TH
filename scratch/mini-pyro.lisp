(defpackage :mini-pyro-check
  (:use #:common-lisp
        #:mu
        #:th
        #:th.distributions))

(in-package :mini-pyro-check)

;; https://willcrichton.net/notes/probabilistic-programming-under-the-hood/

($sample (distribution/bernoulli 0.5))
($sample (distribution/normal 0 1))

(-> (distribution/bernoulli 0.5)
    ($ll 0)
    ($exp))
(-> (distribution/normal 0 1)
    ($ll 0.35)
    ($exp))

(defvar *traces* '())
(defvar *tracep* nil)

(defun msample (name dist)
  (let ((v (or (getf (getf *traces* name) :val) ($sample dist))))
    (when *tracep*
      (let ((m (list :type :sample :ll ($ll dist v) :val v)))
        (setf (getf *traces* name) m)))
    v))

(defun mtrace (fn)
  (if *tracep*
      (progn (funcall fn)
             *traces*)
      (let ((*traces* '())
            (*tracep* T))
        (funcall fn)
        *traces*)))

(defun mcondition (fn conditions)
  (if *tracep*
      (progn
        (loop :for (name v) :on conditions :by #'cddr
              :for p = (getf *traces* name)
              :do (if p
                      (setf (getf p :val) v)
                      (progn
                        (setf (getf p :val) v)
                        (setf (getf *traces* name) p))))
        (mtrace fn))
      (let ((*traces* '())
            (*tracep* T))
        (loop :for (name v) :on conditions :by #'cddr
              :for p = (getf *traces* name)
              :do (if p
                      (setf (getf p :val) v)
                      (progn
                        (setf (getf p :val) v)
                        (setf (getf *traces* name) p))))
        (mtrace fn))))

(defun ll (traces)
  (let ((v 0D0))
    (loop :for (name p) :on traces :by #'cddr
          :for ll = (getf p :ll)
          :do (setf v ($add v ll)))
    v))

(defun sleepy-model ()
  "returns amount slept."
  (let ((feeling-lazy (->> (distribution/bernoulli 0.9)
                           (msample :feeling-lazy))))
    (if (> feeling-lazy 0)
        (let ((ignore-alarm (->> (distribution/bernoulli 0.8)
                                 (msample :ignore-alarm))))
          (->> (distribution/normal (+ 8 (* 2 ignore-alarm)) 1)
               (msample :amount-slept)))
        (->> (distribution/normal 6 1)
             (msample :amount-slept)))))

(sleepy-model)
($exp (ll (mtrace (lambda () (sleepy-model)))))
($exp (ll (mcondition (lambda () (sleepy-model)) '(:feeling-lazy 1
                                              :ignore-alarm 0
                                              :amount-slept 10))))

(let ((traces '()))
  (loop :repeat 10000
        :for tr = (mtrace (lambda () (sleepy-model)))
        :do (let ((vs (loop :for (n p) :on tr :by #'cddr
                            :appending (list n (getf p :val)))))
              (push vs traces)))
  (let ((feeling-lazies (mapcar (lambda (vs) (getf vs :feeling-lazy 0)) traces))
        (ignore-alarms (mapcar (lambda (vs) (getf vs :ignore-alarm 0)) traces))
        (amount-slepts (mapcar (lambda (vs) (getf vs :amount-slept 0)) traces)))
    (list ($mean feeling-lazies) ;; 0.9
          ($mean ignore-alarms) ;; 0.9 * 0.8
          ($mean amount-slepts)))) ;; 9~

(defun model (mu)
  (->> (distribution/normal mu 1)
       (msample :x)))

(let ((mu ($parameter 0)))
  (prn "LL:" (ll (mcondition (lambda () (model mu)) '(:x 5))))
  (prn "GRADIENT:" ($gradient mu)))

(let ((mu ($parameter 0))
      (mu-traces '()))
  (loop :repeat 1000
        :for iter :from 1
        :for loss = ($- (ll (mcondition (lambda () (model mu)) '(:x 5))))
        :do (let ((lv ($scalar loss)))
              (when (zerop (rem iter 100)) (prn iter "LOSS:" lv))
              ($amgd! mu 0.01)
              (push ($scalar mu) mu-traces)))
  mu-traces)

;; feeling-lazy, ignore-alarm, and amount-slept (final return) are required to be traced.
;; in pymc, these variables are traced.
;; in mini pymc as well, these are processed using msg.
