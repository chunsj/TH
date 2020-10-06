(in-package :th.mcmc)

;; potential is negative-log-likelihood function
;; position is the parameter we want to compute

(defun dvdq (potential position)
  (let ((position ($parameter position)))
    (funcall potential position)
    ($gradient position)))

(defun leapfrog-integrator (position momentum potential path-length step-size)
  (let ((half-step-size (/ step-size 2)))
    (setf momentum ($- momentum ($* half-step-size (dvdq potential position))))
    (loop :repeat (1- (round (/ path-length step-size)))
          :do (progn
                (setf position ($+ position ($* step-size momentum)))
                (setf momentum ($- momentum ($* step-size (dvdq potential position))))))
    (setf position ($+ position ($* step-size momentum)))
    (setf momentum ($- momentum ($* half-step-size (dvdq potential position))))
    (list position ($neg momentum))))

(defun hmc (num-samples initial-position potential &key (path-length 1) (step-size 0.1))
  (let* ((momentum-distribution (distribution/normal))
         (momentums ($sample momentum-distribution num-samples))
         (position initial-position)
         (samples '()))
    (loop :for i :from 0 :below num-samples
          :for momentum = ($ momentums i)
          :for irs = (leapfrog-integrator position momentum potential path-length step-size)
          :do (let* ((position-new ($0 irs))
                     (momentum-new ($1 irs))
                     (lp (- (funcall potential position)
                            ($ll momentum-distribution momentum)))
                     (lp-new (- (funcall potential position-new)
                                ($ll momentum-distribution momentum-new)))
                     (lu (log (random 1D0))))
                (if (< lu (- lp lp-new))
                    (progn
                      (push position-new samples)
                      (setf position position-new))
                    (push (if (car samples)
                              (car samples)
                              initial-position)
                          samples))))
    (reverse samples)))
