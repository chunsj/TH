(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defmethod $copy! ((generator generator) from)
  (let ((gen (make-instance 'generator))
        (h (th-generator-copy ($handle generator)
                              ($handle from))))
    (setf ($handle gen) h)
    #+sbcl (sb-ext:finalize gen (lambda () (th-generator-free h)))
    gen))

(defmethod $seed ((generator generator)) (th-random-seed ($handle generator)))
(defmethod (setf $seed) (seed (generator generator))
  (th-random-manual-seed ($handle generator) (coerce seed 'integer))
  seed)

(defmethod $random ((generator generator))  (th-random-random ($handle generator)))

(defmethod $uniform ((generator generator) a b)
  (th-random-uniform ($handle generator)
                     (coerce a 'double-float)
                     (coerce b 'double-float)))

(defmethod $normal ((generator generator) mean stdev)
  (th-random-normal ($handle generator)
                    (coerce mean 'double-float)
                    (coerce stdev 'double-float)))

(defmethod $exponential ((generator generator) lam)
  (th-random-exponential ($handle generator)
                         (coerce lam 'double-float)))

(defmethod $cauchy ((generator generator) median sigma)
  (th-random-cauchy ($handle generator)
                    (coerce median 'double-float)
                    (coerce sigma 'double-float)))

(defmethod $lognormal ((generator generator) mean stdev)
  (th-random-log-normal ($handle generator)
                        (coerce mean 'double-float)
                        (coerce stdev 'double-float)))

(defmethod $geometric ((generator generator) p)
  (th-random-geometric ($handle generator)
                       (coerce p 'double-float)))

(defmethod $bernoulli ((generator generator) p)
  (th-random-bernoulli ($handle generator)
                       (coerce p 'double-float)))

(defmethod $binomial ((generator generator) n p)
  (th-random-binomial ($handle generator)
                      (coerce n 'integer)
                      (coerce p 'double-float)))

(defmethod $hypergeometric ((generator generator) nr nb k)
  (th-random-hypergeometric ($handle generator)
                            (coerce (round nr) 'integer)
                            (coerce (round nb) 'integer)
                            (coerce (round k) 'integer)))

(defmethod $poisson ((generator generator) mu)
  (th-random-poisson ($handle generator)
                     (coerce mu 'double-float)))

(defmethod $beta ((generator generator) a b)
  (th-random-beta ($handle generator) (coerce a 'double-float) (coerce b 'double-float)))

(defmethod $gamma ((generator generator) shape scale)
  (th-random-gamma2 ($handle generator) (coerce shape 'double-float) (coerce scale 'double-float)))

(defmethod $chisq ((generator generator) df)
  ($gamma generator (/ df 2.0) 2.0))

(defmethod $fdist ((generator generator) n1 n2)
  (/ (/ ($chisq generator n1) n1) (/ ($chisq generator n2) n2)))

(defun random/random () ($random *generator*))
(defun random/uniform (a b) ($uniform *generator* a b))
(defun random/normal (m s) ($normal *generator* m s))
(defun random/exponential (l) ($exponential *generator* l))
(defun random/cauchy (m s) ($cauchy *generator* m s))
(defun random/lognormal (m s) ($lognormal *generator* m s))
(defun random/geometric (p) ($geometric *generator* p))
(defun random/hypergeometric (nr nb k) ($hypergeometric *generator* nr nb k))
(defun random/poisson (mu) ($poisson *generator* mu))
(defun random/bernoulli (p) ($bernoulli *generator* p))
(defun random/binomial (n p) ($binomial *generator* n p))
(defun random/beta (a b) ($beta *generator* a b))
(defun random/gamma (shape scale) ($gamma *generator* shape scale))
(defun random/chisq (df) ($chisq *generator* df))
(defun random/fdist (n1 n2) ($fdist *generator* n1 n2))
