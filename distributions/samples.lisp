(in-package :th.distributions)

(defun $sample/beta (&optional (count 1) (a 1D0) (b 1D0))
  (if (eq count 1)
      (random/beta a b)
      ($beta (tensor count) a b)))

(defun $sample/exponential (&optional (count 1) (rate 1D0))
  (if (eq count 1)
      (random/exponential rate)
      ($exponential (tensor count) rate)))

(defun $sample/uniform (&optional (count 1) (l 0D0) (u 1D0))
  (if (eq count 1)
      (random/uniform l u)
      ($uniform (tensor count) l u)))

(defun $sample/gaussian (&optional (count 1) (location 0D0) (scale 1D0))
  (if (eq count 1)
      (random/normal location scale)
      ($normal (tensor count) location scale)))

(defun $sample/normal (&optional (count 1) (location 0D0) (scale 1D0))
  ($sample/gaussian count location scale))

(defun $sample/gamma (&optional (count 1) (shape 1D0) (scale 1D0))
  (if (eq count 1)
      (random/gamma shape scale)
      ($gamma (tensor count) shape scale)))

(defun $sample/t (&optional (count 1) (location 0D0) (scale 1D0) (dof 5))
  (let ((x ($sample/gaussian count 0 1))
        (y ($sample/gamma count (* 0.5 dof) 2)))
    ($add location ($mul scale ($div x ($sqrt ($div y dof)))))))

(defun $sample/chisq (&optional (count 1) (k 1D0))
  ($sample/gamma count (* 0.5 k) 2))

(defun $sample/dice (&optional (count 1) (n 6))
  (let ((n (round n)))
    (if (eq count 1)
        (1+ (random n))
        (tensor.int (loop :repeat count :collect (1+ (random n)))))))

(defun $sample/bernoulli (&optional (count 1) (p 0.5D0))
  (if (eq count 1)
      (random/bernoulli p)
      ($bernoulli (tensor.byte count) p)))

(defun $sample/binomial (&optional (count 1) (p 0.5D0) (n 1))
  (if (eq count 1)
      (random/binomial n p)
      ($binomial (tensor.int count) n p)))

(defun $sample/poisson (&optional (count 1) (rate 1D0))
  (if (eq count 1)
      (random/poisson rate)
      ($poisson (tensor.int count) rate)))
