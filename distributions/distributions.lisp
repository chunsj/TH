(defpackage :th.distributions
  (:use #:common-lisp
        #:mu
        #:th)
  (:export #:$sample
           #:$ll
           #:$cdf
           #:$parameter-names
           #:distribution/bernoulli
           #:distribution/binomial
           #:ci/binomial
           #:distribution/discrete
           #:distribution/poisson
           #:distribution/beta
           #:distribution/exponential
           #:distribution/gaussian
           #:distribution/normal
           #:distribution/gamma
           #:distribution/t
           #:distribution/uniform))

(in-package :th.distributions)

(defgeneric $sample (distribution &optional n) (:documentation "returns random sample."))
(defgeneric $ll (distribution data) (:documentation "returns log likelihood."))
(defgeneric $cdf (distribution v) (:documentation "return cummulative density of x < v."))
(defgeneric $parameter-names (distribution) (:documentation "returns a list of parameter names."))

(defclass distribution () ())

(defmethod $sample ((d distribution) &optional (n 1)) (declare (ignore n)) nil)
(defmethod $ll ((d distribution) data)
  (declare (ignore data))
  most-negative-single-float)
(defmethod $cdf ((d distribution) v)
  (declare (ignore v))
  0)
(defmethod $parameter-names ((d distribution)) '())
(defmethod $parameters ((d distribution)) '())

(defun pv (pv)
  (if ($parameterp pv)
      ($data pv)
      pv))

(defmacro underflow-goes-to-zero (&body body)
  "Protects against floating point underflow errors and sets the value to 0.0 instead."
  `(handler-case
       (progn ,@body)
     (floating-point-underflow (condition)
       (declare (ignore condition))
       (values 0.0d0))))

(defun safe-exp (x)
  "Eliminates floating point underflow for the exponential function.
Instead, it just returns 0.0d0"
  (setf x (coerce x 'double-float))
  (if (< x (log least-positive-double-float))
      0.0d0
      (exp x)))

(defun gamma-incomplete (a x)
  "Adopted from CLASP 1.4.3, http://eksl-www.cs.umass.edu/clasp.html"
  (declare (optimize (safety 3)))
  (setq a (coerce a 'double-float))
  (let ((gln (the double-float ($lgammaf a))))
    (when (= x 0.0)
      (return-from gamma-incomplete (values 0.0d0 gln)))
    (if (< x (+ a 1.0d0))
        ;; Use series representation.  The following is the code of what
        ;; Numerical Recipes in C calls ``GSER'
        (let* ((itmax 1000)
               (eps   3.0d-7)
               (ap    a)
               (sum   (/ 1d0 a))
               (del sum))
          (declare (type double-float ap sum del))
          (dotimes (i itmax)
            (incf ap 1.0d0)
            (setf del (* del (/ x ap)))
            (incf sum del)
            (if (< (abs del) (* eps (abs sum)))
                (let ((result (underflow-goes-to-zero
                                (* sum (safe-exp (- (* a (log x)) x gln))))))
                  (return-from gamma-incomplete (values result gln)))))
          (error "Series didn't converge:~%~
                  Either a=~s is too large, or ITMAX=~d is too small." a itmax))
        ;; Use the continued fraction representation.  The following is the
        ;; code of what Numerical Recipes in C calls ``GCF.'' Their code
        ;; computes the complement of the desired result, so we subtract from
        ;; 1.0 at the end.
        (let ((itmax 1000)
              (eps   3.0e-7)
              (gold 0d0) (g 0d0) (fac 1d0) (b1 1d0) (b0 0d0)
              (anf 0d0) (ana 0d0) (an 0d0) (a1 x) (a0 1d0))
          (declare (type double-float gold g fac b1 b0 anf ana an a1 a0))
          (dotimes (i itmax)
            (setf an  (coerce (1+ i) 'double-float)
                  ana (- an a)
                  a0  (* fac (+ a1 (* a0 ana)))
                  b0  (* fac (+ b1 (* b0 ana)))
                  anf (* fac an)
                  a1  (+ (* x a0) (* anf a1))
                  b1  (+ (* x b0) (* anf b1)))
            (unless (zerop a1)
              (setf fac (/ 1.0d0 a1)
                    g   (* b1 fac))
              (if (< (abs (/ (- g gold) g)) eps)
                  (let ((result (underflow-goes-to-zero
                                  (* (safe-exp (- (* a (log x)) x gln)) g))))
                    (return-from
                     gamma-incomplete (values (- 1.0 result) gln)))
                  (setf gold g))))
          (error "Continued Fraction didn't converge:~%~
                  Either a=~s is too large, or ITMAX=~d is too small." a
                  itmax)))))

(defun beta-incomplete (a b x)
  "Adopted from CLASP 1.4.3, http://eksl-www.cs.umass.edu/clasp.html"
  (flet ((betacf (a b x)
           ;; straight from Numerical Recipes in C, section 6.3
           (declare (type double-float a b x))
           (let ((ITMAX 1000)
                 (EPS   3.0d-7)
                 (qap 0d0) (qam 0d0) (qab 0d0) (em  0d0) (tem 0d0) (d 0d0)
                 (bz  0d0) (bm  1d0) (bp  0d0) (bpp 0d0)
                 (az  1d0) (am  1d0) (ap  0d0) (app 0d0) (aold 0d0))
             (declare (type double-float qap qam qab em tem d
                            bz bm bp bpp az am ap app aold))
             (setf qab (+ a b)
                   qap (+ a 1d0)
                   qam (- a 1d0)
                   bz  (- 1d0 (/ (* qab x) qap)))
             (dotimes (m ITMAX)
               (setf em   (coerce (1+ m) 'double-float)
                     tem  (+ em em)
                     d    (/ (* em (- b em) x)
                             (* (+ qam tem) (+ a tem)))
                     ap   (+ az (* d am))
                     bp   (+ bz (* d bm))
                     d    (/ (* (- (+ a em)) (+ qab em) x)
                             (* (+ qap tem) (+ a tem)))
                     app  (+ ap (* d az))
                     bpp  (+ bp (* d bz))
                     aold az
                     am   (/ ap bpp)
                     bm   (/ bp bpp)
                     az   (/ app bpp)
                     bz   1d0)
               (if (< (abs (- az aold)) (* EPS (abs az)))
                   (return-from betacf az)))
             (error "a=~s or b=~s too big, or ITMAX too small in BETACF"
                    a b))))
    (declare (notinline betacf))
    (setq a (coerce a 'double-float) b (coerce b 'double-float)
          x (coerce x 'double-float))
    (when (or (< x 0d0) (> x 1d0))
      (error "x must be between 0d0 and 1d0:  ~f" x))
    ;; bt is the factors in front of the continued fraction
    (let ((bt (if (or (= x 0d0) (= x 1d0))
                  0d0
                  (exp (+ ($lgammaf (+ a b))
                          (- ($lgammaf a))
                          (- ($lgammaf b))
                          (* a (log x))
                          (* b (log (- 1d0 x))))))))
      (if (< x (/ (+ a 1d0) (+ a b 2.0)))
          ;; use continued fraction directly
          (/ (* bt (betacf a b x)) a)
          ;; use continued fraction after making the symmetry transformation
          (- 1d0 (/ (* bt (betacf b a (- 1d0 x))) b))))))

(defun t-significance (t-statistic dof &key (tails :both))
  "Lookup table in Rosner; this is adopted from CLASP/Numeric
Recipes (CLASP 1.4.3), http://eksl-www.cs.umass.edu/clasp.html"
  (setf dof (float dof t-statistic))
  (let ((a (beta-incomplete (* 0.5 dof) 0.5 (/ dof (+ dof ($square t-statistic))))))
    ;; A is 2*Integral from (abs t-statistic) to Infinity of t-distribution
    (ecase tails
      (:both a)
      (:positive (if (plusp t-statistic)
                     (* .5 a)
                     (- 1.0 (* .5 a))))
      (:negative (if (plusp t-statistic)
                     (- 1.0 (* .5 a))
                     (* .5 a))))))

(defun f-significance
    (f-statistic numerator-dof denominator-dof &optional one-tailed-p)
  "Adopted from CLASP, but changed to handle F < 1 correctly in the
one-tailed case.  The `f-statistic' must be a positive number.  The
degrees of freedom arguments must be positive integers.  The
`one-tailed-p' argument is treated as a boolean.

This implementation follows Numerical Recipes in C, section 6.3 and
the `ftest' function in section 13.4."
  (setq f-statistic (float f-statistic))
  (let ((tail-area (beta-incomplete
                    (* 0.5d0 denominator-dof)
                    (* 0.5d0 numerator-dof)
                    (float (/ denominator-dof
                              (+ denominator-dof
                                 (* numerator-dof f-statistic))) 1d0))))
    (if one-tailed-p
        (if (< f-statistic 1) (- 1 tail-area) tail-area)
        (progn (setf tail-area (* 2.0 tail-area))
               (if (> tail-area 1.0)
                   (- 2.0 tail-area)
                   tail-area)))))

(defun find-critical-value
    (p-function p-value &optional (x-tolerance .00001) (y-tolerance .00001))
  "Adopted from CLASP 1.4.3, http://eksl-www.cs.umass.edu/clasp.html"
  (let* ((x-low 0d0)
         (fx-low 1d0)
         (x-high 1d0)
         (fx-high (coerce (funcall p-function x-high) 'double-float)))
    ;; double up
    (declare (type double-float x-low fx-low x-high fx-high))
    (do () (nil)
      ;; for general functions, we'd have to try the other way of bracketing,
      ;; and probably have another way to terminate if, say, y is not in the
      ;; range of f.
      (when (>= fx-low p-value fx-high)
	(return))
      (setf x-low x-high
            fx-low fx-high
            x-high (* 2.0 x-high)
            fx-high (funcall p-function x-high)))
    ;; binary search
    (do () (nil)
      (let* ((x-mid  (/ (+ x-low x-high) 2.0))
             (fx-mid (funcall p-function x-mid))
             (y-diff (abs (- fx-mid p-value)))
             (x-diff (- x-high x-low)))
	(when (or (< x-diff x-tolerance)
                  (< y-diff y-tolerance))
          (return-from find-critical-value x-mid))
	;; Because significance is monotonically decreasing with x, if the
	;; function is above the desired p-value...
	(if (< p-value fx-mid)
            ;; then the critical x is in the upper half
            (setf x-low x-mid
                  fx-low fx-mid)
            ;; otherwise, it's in the lower half
            (setf x-high x-mid
                  fx-high fx-mid))))))

(defun phi (x)
  "the CDF of standard normal distribution. Adopted from CLASP 1.4.3,
see copyright notice at http://eksl-www.cs.umass.edu/clasp.html"
  (* .5 (+ 1.0 ($erf (/ x (sqrt 2.0))))))

(defun z (percentile &key (epsilon 1d-15))
  "The inverse normal function, P(X<Zu) = u where X is distributed as
the standard normal. Uses binary search."
  (let ((target (coerce percentile 'double-float)))
    (do ((min -9d0 min)
         (max 9d0 max)
         (guess 0d0 (+ min (/ (- max min) 2d0))))
        ((< (- max min) epsilon) guess)
      (let ((result (coerce (phi guess) 'double-float)))
        (if (< result target)
            (setq min guess)
            (setq max guess))))))

(defun factorial (number)
  (if (not (and (integerp number) (>= number 0)))
      (error "factorial: ~a is not a positive integer" number)
      (labels ((fact (num) (if (= 0 num) 1 (* num (fact (1- num))))))
        (fact number))))

(defun choose (n k)
  "How may ways to take n things taken k at a time, when order doesn't
matter"
  (/ (factorial n) (* (factorial k) (factorial (- n k)))))
