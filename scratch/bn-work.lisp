(in-package :th)

(defun bnorm (x gamma beta rmean rvar smean ssd &optional (trainp t) (momentum 0.1) (eps 1E-5))
  (let ((out ($empty x))
        (input (if (eq 1 ($ndim x)) (apply #'$reshape x (cons 1 ($size x))) x)))
    (nn-batch-normalization-update-output input out gamma beta rmean rvar smean ssd
                                          (and trainp (> ($size input 0) 1)) momentum eps)
    out))

(defun bnbwd (x gradient dgamma dbeta gamma rmean rvar smean ssd
              &optional (trainp t) (eps 1E-5))
  (let ((input (if (eq 1 ($ndim x)) (apply #'$reshape x (cons 1 ($size x))) x))
        (dinput ($empty x)))
    ($resize! dinput gradient)
    (nn-batch-normalization-backward input gradient dinput dgamma dbeta gamma
                                     rmean rvar smean ssd
                                     (and trainp (> ($size input 0) 1))
                                     1 eps)
    dinput))

;; batch normalization test
(let* ((x (rndn 10 784))
       (nout 784)
       (gamma (rndn nout))
       (beta (rndn nout))
       (rmean (zeros nout))
       (rvar (ones nout))
       (smean (zeros nout))
       (ssd (ones nout))
       (gradient (rndn 10 784))
       (dgamma (zeros nout))
       (dbeta (zeros nout)))
  (print (bnorm x gamma beta rmean rvar smean ssd))
  (print smean)
  (print rmean)
  (print (bnbwd x gradient dgamma dbeta gamma rmean rvar smean ssd))
  (print dgamma)
  (print dbeta))

;; single row case
(let* ((x (rndn 784))
       (nout 784)
       (gamma (rndn nout))
       (beta (rndn nout))
       (rmean (zeros nout))
       (rvar (ones nout))
       (smean (zeros nout))
       (ssd (ones nout)))
  (print (bnorm x gamma beta rmean rvar smean ssd))
  (print smean)
  (print rmean))
