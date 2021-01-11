(defpackage :mixture
  (:use #:common-lisp
        #:mu
        #:th
        #:th.pp))

(in-package :mixture)

(defparameter *data* (->> (slurp "data/mixture.csv")
                          (mapcar (lambda (f) (parse-float f)))
                          (tensor)))

(defun posterior (p s0 s1 c0 c1)
  (let ((p-prior (score/uniform p 0.0 1.0))
        (s0-prior (score/uniform s0 0.0 100.0))
        (s1-prior (score/uniform s1 0.0 100.0))
        (c0-prior (score/gaussian c0 120.0 10.0))
        (c1-prior (score/gaussian c1 190.0 10.0)))
    (when (and p-prior s0-prior s1-prior c0-prior c1-prior)
      (let ((assignments (sample/categorical (tensor (list p (- 1 p))) ($count *data*))))
        (let ((sd (tensor (loop :for i :from 0 :below ($count assignments)
                                :collect (if (zerop ($ assignments i))
                                             s0
                                             s1))))
              (mean (tensor (loop :for i :from 0 :below ($count assignments)
                                  :collect (if (zerop ($ assignments i))
                                               c0
                                               c1)))))
          (let ((likelihood (score/normal *data* mean sd)))
            (when likelihood
              ($+ p-prior s0-prior s1-prior c0-prior c1-prior likelihood))))))))

(let ((traces (mcmc/mh '(0.5 50.0 50.0 120.0 190.0) #'posterior)))
  (loop :for trace :in traces
        :do (prn trace (trace/hpd trace))))
