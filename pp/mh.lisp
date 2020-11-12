(in-package :th.pp)

;; XXX
;; here comes the metropolis-hastings algorithm
;;
;; write down the exact process of this sampling function,
;; class-responnsibilities-collaborators like analysis could be helpful.
;; also, check hamiltonian monte carlo function.

;; XXX this is a mere copy of rv.lisp from distributions package, this does not work.
(defun mh (parameters likelihoodfn &key (iterations 50000) (tune-steps 100)
                                     (burn-ins 1000) (thin 1)
                                     verbose)
  (let ((old-likelihood (apply likelihoodfn parameters))
        (old-parameters (mapcar #'$clone parameters))
        (proposals (mapcar (lambda (p)
                             (if ($continuousp p)
                                 (proposal/gaussian)
                                 (proposal/discrete-gaussian)))
                           parameters))
        (traces (mapcar (lambda (p) (mcmc/trace ($clone p) :burn-ins burn-ins :thin thin)) parameters))
        (deviance nil)
        (max-likelihood most-negative-single-float))
    (when old-likelihood
      (setf deviance (mcmc/trace (rv/variable (* -2 old-likelihood))
                                 :burn-ins burn-ins :thin thin))
      (loop :repeat (+ iterations burn-ins)
            :for iter :from 1
            :do (let ((can-tune (and (> iter 1) (<= iter burn-ins) (zerop (rem iter tune-steps)))))
                  (when can-tune
                    (loop :for proposal :in proposals
                          :do ($tune! proposal)))
                  (loop :for proposal :in proposals
                        :for parameter :in old-parameters
                        :for trace :in traces
                        :for proposed = ($propose proposal parameter)
                        :for new-parameter = (car proposed)
                        :for log-hastings-ratio = (cdr proposed)
                        :for parameter-index :from 0
                        :for ps = (loop :for p :in old-parameters
                                        :for i :from 0
                                        :collect (if (eq i parameter-index)
                                                     new-parameter
                                                     p))
                        :for new-likelihood = (apply likelihoodfn ps)
                        :do (let ((acceptable (and log-hastings-ratio
                                                   new-likelihood
                                                   (> (+ (- new-likelihood old-likelihood)
                                                         log-hastings-ratio)
                                                      (log (random 1D0))))))
                              (when acceptable
                                ($mcmc/push! ($clone new-parameter) trace)
                                (setf ($ old-parameters parameter-index) new-parameter
                                      old-likelihood new-likelihood)
                                ($accepted! proposal)
                                (when (> new-likelihood max-likelihood)
                                  (set-mlev! trace ($clone new-parameter))
                                  (setf max-likelihood new-likelihood)))
                              (unless acceptable
                                ($mcmc/push! ($clone parameter) trace)
                                ($rejected! proposal))))
                  ($mcmc/push! (rv/variable (* -2 old-likelihood)) deviance))))
    (when verbose
      (prn "* MLE PARAMETERS")
      (prn (format nil "~{~A ~}" (mapcar #'$mcmc/mle traces))))
    (values traces deviance)))
