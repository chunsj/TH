(in-package :th.distributions)

(defclass distribution/bernoulli (distribution)
  ((p :initform 0.5)))

(defun distribution/bernoulli (&optional (p 0.5D0))
  (let ((dist (make-instance 'distribution/bernoulli))
        (pin p))
    (with-slots (p) dist
      (setf p pin))
    dist))

(defmethod $parameters ((d distribution/bernoulli))
  (with-slots (p) d
    (if ($parameterp p)
        (list p)
        '())))

(defmethod $parameter-names ((d distribution/bernoulli))
  (list :p))

(defmethod $ ((d distribution/bernoulli) name &rest others-and-default)
  (declare (ignore others-and-default))
  (when (eq name :p)
    (with-slots (p) d
      p)))

(defmethod (setf $) (value (d distribution/bernoulli) name &rest others)
  (declare (ignore others))
  (when (eq name :p)
    (with-slots (p) d
      (setf p value)
      value)))

(defmethod $sample ((d distribution/bernoulli) &optional (n 1))
  (when (> n 0)
    (with-slots (p) d
      (cond ((eq n 1) (random/bernoulli (pv p)))
            (T ($bernoulli (tensor.byte n) (pv p)))))))

(defmethod $ll ((d distribution/bernoulli) (data number))
  (with-slots (p) d
    (if (> data 0)
        ($log p)
        ($log ($sub 1 p)))))

(defmethod $ll ((d distribution/bernoulli) (data list))
  (with-slots (p) d
    (let ((nd ($count data))
          (nt 0))
      (loop :for d :in data :do (when (> d 0) (incf nt)))
      ($add ($mul nt ($log p)) ($mul (- nd nt) ($log ($sub 1 p)))))))

(defmethod $ll ((d distribution/bernoulli) (data tensor))
  (with-slots (p) d
    (let ((nd ($count data))
          (nt ($count ($nonzero data))))
      ($add ($mul nt ($log p)) ($mul (- nd nt) ($log ($sub 1 p)))))))

(defclass distribution/binomial (distribution)
  ((n :initform 1)
   (p :initform 0.5)))

(defun distribution/binomial (&optional (n 1) (p 0.5D0))
  (let ((dist (make-instance 'distribution/binomial))
        (nin n)
        (pin p))
    (with-slots (n p) dist
      (setf n nin
            p pin))
    dist))

(defmethod $parameters ((d distribution/binomial))
  (with-slots (n p) d
    (let ((ps '()))
      (when ($parameterp p) (push p ps))
      (when ($parameterp n) (push n ps))
      ps)))

(defmethod $parameter-names ((d distribution/binomial))
  (list :n :p))

(defmethod $ ((d distribution/binomial) name &rest others-and-default)
  (declare (ignore others-and-default))
  (with-slots (n p) d
    (cond ((eq name :n) n)
          ((eq name :p) p))))

(defmethod (setf $) (value (d distribution/binomial) name &rest others)
  (declare (ignore others))
  (with-slots (n p) d
    (cond ((eq name :n) (setf n value))
          ((eq name :p) (Setf p value))))
  value)

(defmethod $sample ((d distribution/binomial) &optional (n 1))
  (let ((nin n))
    (when (> nin 0)
      (with-slots (n p) d
        (cond ((eq nin 1) (random/binomial ($scalar n) ($scalar p)))
              (T ($binomial (tensor.int nin) ($scalar n) ($scalar p))))))))

(defun logfac (n) ($lgammaf ($+ 1 n)))

(defun logbc (n x)
  (let ((m (if (< x (- n x)) x (- n x)))
        (o (if (< x (- n x)) (- n x) x))
        (lp 0))
    (loop :for i :from (+ o 1) :to n
          :do (incf lp (log i)))
    (- lp (logfac (coerce (round m) 'integer)))))

(defmethod $ll ((d distribution/binomial) (data number))
  (if (>= data 0)
      (with-slots (n p) d
        (if (and (>= data 0) (<= data ($scalar n)))
            ($+ (logbc ($scalar n) data)
                (if (zerop data)
                    0
                    ($mul data ($log p)))
                (if (zerop (- ($scalar n) data))
                    0
                    (if (zerop (- ($scalar p) 1D0))
                        0
                        ($mul (- ($scalar n) data) ($log ($sub 1 p))))))
            most-negative-single-float))
      most-negative-single-float))

(defmethod $ll ((d distribution/binomial) (data list))
  (let ((cnt ($count data))
        (npos ($count (filter (lambda (v) (>= v 0)) data))))
    (if (eq cnt npos)
        (with-slots (n p) d
          (let ((nlim ($count (filter (lambda (v) (<= v ($scalar n))) data))))
            (if (eq nlim cnt)
                (let ((dt (tensor data))
                      (lp ($log p))
                      (lmp ($log ($sub 1 p)))
                      (cs (tensor (mapcar (lambda (v) (logbc ($scalar n) v)) data))))
                  ($sum ($+ cs ($mul dt lp) ($mul ($sub ($scalar n) dt) lmp))))
                most-negative-single-float)))
        most-negative-single-float)))

(defmethod $ll ((d distribution/binomial) (data tensor))
  (let ((cnt ($count data))
        (npos ($sum ($ge data 0))))
    (if (eq cnt npos)
        (with-slots (n p) d
          (let ((nlim ($sum ($le data ($scalar n)))))
            (if (eq nlim cnt)
                (let ((lp ($log p))
                      (lmp ($log ($sub 1 p)))
                      (cs (tensor (mapcar (lambda (v) (logbc ($scalar n) v)) ($list data)))))
                  ($sum ($+ cs ($mul data lp) ($mul ($sub ($scalar n) data) lmp))))
                most-negative-single-float)))
        most-negative-single-float)))

(defun binomial-probability (n k p)
  "P(X=k) for X a binomial random variable with parameters n &
p. Binomial expectations for seeing k events in N trials, each having
probability p.  Use the Poisson approximation if N>100 and P<0.01."
  (if (and (> n 100) (< p 0.01))
      (let ((d (distribution/poisson (* n p))))
        ($exp ($ll d k)))
      (let ((p (coerce p 'double-float)))
        (* (choose n k)
           (expt p k)
           (expt (- 1 p) (- n k))))))

(defun binomial-cumulative-probability (n k p)
  "P(X<k) for X a binomial random variable with parameters n &
p. Bionomial expecations for fewer than k events in N trials, each
having probability p."
  (let ((sum-up-to-k-1 0d0))
    (dotimes (i k sum-up-to-k-1)
      (incf sum-up-to-k-1 (binomial-probability n i p)))))

(defmethod $cdf ((d distribution/binomial) (k number))
  (with-slots (n p) d
    (binomial-cumulative-probability ($scalar n) k ($scalar p))))

(defmethod $cdf ((d distribution/binomial) (ks list))
  (mapcar (lambda (k) ($cdf d k)) ks))

(defmethod $cdf ((d distribution/binomial) (ks tensor))
  (tensor (mapcar (lambda (k) ($cdf d k)) ($list ks))))

(defun ci/binomial (alpha n p &optional exactp)
  (if (and (> (* n p (- 1 p)) 10) (not exactp))
      (let ((difference (* (z (- 1 (/ alpha 2)))
                           (sqrt (/ (* p (- 1 p)) n)))))
        (values (- p difference) (+ p difference)))
      (values (find-critical-value
               (lambda (p1)
                 (let ((d (distribution/binomial n p1)))
                   ($cdf d (floor (* p n)))))
               (- 1 (/ alpha 2)))
              (find-critical-value
               (lambda (p2)
                 (let ((d (distribution/binomial n p2)))
                   ($cdf d (1+ (floor (* p n))))))
               (/ alpha 2)))))

(defclass distribution/discrete (distribution)
  ((ps :initform (tensor '(0.5 0.5)))))

(defun distribution/discrete (&optional (ps (tensor '(0.5 0.5))))
  (let ((dist (make-instance 'distribution/discrete))
        (psin ps))
    (with-slots (ps) dist
      (setf ps psin))
    dist))

(defmethod $parameters ((d distribution/discrete))
  (with-slots (ps) d
    (when ($parameterp ps)
      (list ps))))

(defmethod $parameter-names ((d distribution/discrete))
  (list :ps))

(defmethod $ ((d distribution/discrete) name &rest others-and-default)
  (declare (ignore others-and-default))
  (when (eq name :ps)
    (with-slots (ps) d
      ps)))

(defmethod (setf $) (value (d distribution/discrete) name &rest others)
  (declare (ignore others))
  (when (eq name :ps)
    (with-slots (ps) d
      (setf ps value)))
  value)

(defun sample-discrete (ps)
  (let ((sum ($scalar ($sum ps)))
        (rr (random 1D0))
        (n ($count ps))
        (accum 0))
    (loop :for i :from 0 :below n
          :for paccum = (let ((p ($ ps i)))
                          (incf accum p)
                          accum)
          :when (< (* rr sum) paccum)
            :return i)))

(defmethod $sample ((d distribution/discrete) &optional (n 1))
  (when (> n 0)
    (with-slots (ps) d
      (cond ((eq n 1) (sample-discrete ps))
            (T (tensor.int (loop :repeat n :collect (sample-discrete ps))))))))

(defmethod $ll ((d distribution/discrete) (data number))
  (with-slots (ps) d
    ($log ($div ($ ps data) ($sum ps)))))

(defmethod $ll ((d distribution/discrete) (data list))
  ($ll d (tensor.long data)))

(defmethod $ll ((d distribution/discrete) (data tensor))
  (with-slots (ps) d
    ($sum ($log ($div ($gather ps 0 (tensor.long data)) ($sum ps))))))

(defclass distribution/poisson (distribution)
  ((l :initform 1.0)))

(defun distribution/poisson (&optional (l 1D0))
  (let ((dist (make-instance 'distribution/poisson))
        (lin l))
    (with-slots (l) dist
      (setf l lin))
    dist))

(defmethod $parameters ((d distribution/poisson))
  (with-slots (l) d
    (when ($parameterp l)
      (list l))))

(defmethod $parameter-names ((d distribution/poisson))
  (list :l))

(defmethod $ ((d distribution/poisson) name &rest others-and-default)
  (declare (ignore others-and-default))
  (when (eq name :l)
    (with-slots (l) d
      l)))

(defmethod (setf $) (value (d distribution/poisson) name &rest others)
  (declare (ignore others))
  (when (eq name :l)
    (with-slots (l) d
      (setf l value)))
  value)

(defmethod $sample ((d distribution/poisson) &optional (n 1))
  (when (> n 0)
    (with-slots (l) d
      (cond ((eq n 1) (random/poisson ($scalar l)))
            (T ($poisson (tensor.int n) ($scalar l)))))))

(defmethod $ll ((d distribution/poisson) (data number))
  (with-slots (l) d
    ($sub ($sub ($mul data ($log l)) l) (logfac data))))

(defmethod $ll ((d distribution/poisson) (data list))
  ($ll d (tensor data)))

(defmethod $ll ((d distribution/poisson) (data tensor))
  (with-slots (l) d
    ($sum ($sub ($sub ($mul data ($log l)) l) (logfac data)))))

(defmethod $cdf ((d distribution/poisson) (k number))
  (if (< k 170)
      (let ((sum 0D0))
        (dotimes (x k sum)
          (incf sum ($exp ($ll d k)))))
      (let ((mu ($scalar ($ d :l))))
        (- 1D0 (gamma-incomplete (coerce k 'double-float) (coerce mu 'double-float))))))

(defmethod $cdf ((d distribution/poisson) (ks list))
  (mapcar (lambda (k) ($cdf d k)) ks))

(defmethod $cdf ((d distribution/poisson) (ks tensor))
  (tensor (mapcar (lambda (k) ($cdf d k)) ($list ks))))

(defun ci/poisson (alpha k)
  (values
   (find-critical-value
    (lambda (mu)
      (let ((d (distribution/poisson mu)))
        ($cdf d (1- k))))
    (- 1 (/ alpha 2)))
   (find-critical-value
    (lambda (mu)
      (let ((d (distribution/poisson mu)))
        ($cdf d k)))
    (/ alpha 2))))
