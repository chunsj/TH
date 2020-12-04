(in-package :th.pp)

(defun hmc/momentum (rv m sd)
  (let ((v ($data rv)))
    (cond (($tensorp v) ($resize! (sample/normal m sd ($count v)) v))
          (T (sample/gaussian m sd)))))

(defun update-parameters! (parameters momentums step-size)
  (loop :for p :in parameters
        :for m :in momentums
        :do (when (r/continuousp p)
              ($incf ($data p) ($* step-size m)))))

(defun update-momentums! (momentums gradients step-size)
  (loop :for g :in gradients
        :for i :from 0
        :do ($incf ($ momentums i) ($* step-size g))))

(defun leapfrog (candidates momentums posterior path-length step-size)
  (let ((half-step-size (/ step-size 2))
        (cs (mapcar #'$clone candidates))
        (ps (->> candidates
                 (mapcar (lambda (c)
                           (if (r/continuousp c)
                               ($parameter ($data c))
                               ($data c))))))
        (nr (max 0 (1- (round (/ path-length step-size))))))
    (labels ((dvdq (cs)
               (loop :for c :in cs
                     :for p :in ps
                     :do (when (r/continuousp c)
                           ($cg! p)
                           (setf ($data p) ($data c))))
               (funcall posterior ps)
               (->> ps
                    (mapcar (lambda (p)
                              (if ($parameterp p)
                                  ($gradient p)
                                  ($zero p)))))))
      (loop :repeat nr
            :do (progn
                  (update-momentums! momentums (dvdq cs) half-step-size)
                  (update-parameters! cs momentums step-size)
                  (update-momentums! momentums (dvdq cs) half-step-size)))
      (loop :for m :in momentums :do ($neg! m))
      (list (funcall posterior (mapcar #'$data cs)) cs momentums))))

(defun hmc/accepted (l k nl nk)
  (when (and l k nl nk)
    (< (log (random 1.0)) (- (- nl nk) (- l k)))))

(defclass hmc/step-sizer ()
  ((mu :initform nil)
   (target-ratio :initform nil)
   (gamma :initform nil)
   (l :initform nil)
   (kappa :initform nil)
   (errsum :initform 0)
   (lavgstep :initform 0)))

(defun hmc/step-sizer (step-size0 &key (tr 0.65) (g 0.05) (l0 10D0) (k 0.75))
  (let ((n (make-instance 'hmc/step-sizer)))
    (with-slots (mu target-ratio gamma l kappa errsum lavgstep) n
      (setf mu (log (* 10 step-size0))
            target-ratio tr
            gamma g
            l l0
            kappa k
            errsum 0
            lavgstep 0))
    n))

(defun hmc/update-step-sizer! (sizer paccept)
  (with-slots (mu target-ratio gamma l kappa errsum lavgstep) sizer
    (let ((logstep nil)
          (eta nil)
          (min-step 0.01))
      (incf errsum (- target-ratio paccept))
      (setf logstep (- mu (/ errsum (* (sqrt l) gamma))))
      (setf eta (expt l (- kappa)))
      (setf lavgstep (+ (* eta logstep) (* (- 1 eta) lavgstep)))
      (incf l)
      (list (max min-step (exp logstep)) (max min-step (exp lavgstep))))))

(defun hmc/kinetic (ms)
  (loop :for momentum :in ms
        :summing ($* 0.5 ($dot momentum momentum))))

(defun mcmc/hmc (parameters posterior-function
                 &key (iterations 8000) (tune-steps 100) (burn-in 1000) (thin 1))
  (labels ((posterior (vs) (apply posterior-function vs))
           (vals (parameters) (mapcar #'$data parameters)))
    (let ((nsize (+ burn-in iterations))
          (l (posterior (vals parameters)))
          (np ($count parameters))
          (m 0)
          (sd 1)
          (path-length 1)
          (step-size 0.1))
      (when l
        (let ((proposals (mapcar #'r/proposal parameters))
              (cs (mapcar #'$clone parameters))
              (traces (mcmc/traces np :burn-in burn-in :thin thin))
              (maxprob l)
              (naccepted 0)
              (nrejected 0)
              (step step-size)
              (fstep step-size)
              (sizer (hmc/step-sizer step-size))
              (tuning-done nil)
              (failed nil))
          (prn (format nil "[MCMC/HMC: TUNING..."))
          (loop :for trace :in traces
                :for candidate :in cs
                :do (trace/map! trace ($data candidate)))
          (loop :repeat nsize
                :for iter :from 1
                :for burning = (<= iter burn-in)
                :for tuneable = (zerop (rem iter tune-steps))
                :for ms = (mapcar (lambda (c) (hmc/momentum c m sd)) cs)
                :for k = (hmc/kinetic ms)
                :for (nl ncs nms) = (leapfrog cs ms #'posterior path-length step)
                :for nk = (hmc/kinetic nms)
                :while (not failed)
                :do (let ((accept (hmc/accepted l k nl nk))
                          (tune (and (> iter 1) burning tuneable)))
                      (when tune
                        (loop :for proposal :in proposals :do (proposal/tune! proposal)))
                      (when burning
                        (let ((r (* 1D0 (/ naccepted (+ 1E-7 naccepted nrejected)))))
                          (let ((stune (hmc/update-step-sizer! sizer r)))
                            (setf fstep (car stune))
                            (setf step (car stune)))))
                      (unless burning
                        (unless tuning-done
                          (prns (format nil " DONE. SAMPLING..."))
                          (setf tuning-done T))
                        (unless (= step fstep)
                          (setf step fstep)))
                      (when tuning-done
                        (unless burning
                          (when (zerop naccepted)
                            (setf failed T))))
                      (when accept
                        (incf naccepted)
                        (loop :for tr :in traces
                              :for c :in ncs
                              :do (when (r/continuousp c)
                                    (trace/push! ($data c) tr)))
                        (setf cs ncs)
                        (setf l nl)
                        (when (> l maxprob)
                          (setf maxprob l)
                          (loop :for trace :in traces
                                :for candidate :in cs
                                :do (when (r/continuousp candidate)
                                      (trace/map! trace ($data candidate))))))
                      (unless accept
                        (incf nrejected)
                        (loop :for tr :in traces
                              :for c :in cs
                              :do (when (r/continuousp c)
                                    (trace/push! ($data c) tr))))
                      (loop :for proposal :in proposals
                            :for candidate :in cs
                            :for trace :in traces
                            :do (when (r/discretep candidate)
                                  (let* ((lhr (r/propose! candidate proposal))
                                         (nprob (posterior (vals cs)))
                                         (accepted (mh/accepted l nprob lhr)))
                                    (r/accept! candidate proposal accepted)
                                    (trace/push! ($data candidate) trace)
                                    (when accepted
                                      (setf l nprob)
                                      (when (> nprob maxprob)
                                        (setf maxprob nprob)
                                        (trace/map! trace ($data candidate)))))))))
          (if (zerop naccepted)
              (prns (format nil " FAILED]~%"))
              (prns (format nil " FINISHED]~%")))
          traces)))))
