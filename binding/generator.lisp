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

(defmethod $beta ((generator generator) a b)
  (th-random-beta ($handle generator) (coerce a 'double-float) (coerce b 'double-float)))

(defmethod $gamma ((generator generator) shape scale))

(defun random/random () ($random *generator*))
(defun random/uniform (a b) ($uniform *generator* a b))
(defun random/normal (m s) ($normal *generator* m s))
(defun random/exponential (l) ($exponential *generator* l))
(defun random/cauchy (m s) ($cauchy *generator* m s))
(defun random/lognormal (m s) ($lognormal *generator* m s))
(defun random/geometric (p) ($geometric *generator* p))
(defun random/bernoulli (p) ($bernoulli *generator* p))
(defun random/beta (a b) ($beta *generator* a b))
(defun random/gamma (shape scale) ($gamma *generator* shape scale))
