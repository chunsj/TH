(in-package :th.pp)

(defclass mcmc/trace ()
  ((collection :initform nil)
   (burn-ins :initform 0)
   (thin :initform 0)))

(defun mcmc/trace (&key (burn-in 0) (thin 0))
  (let ((ntr (make-instance 'mcmc/trace))
        (nb burn-in)
        (nt thin))
    (with-slots (burn-ins thin) ntr
      (setf burn-ins nb
            thin nt))
    ntr))
