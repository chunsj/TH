(in-package :th.pp)

(defun hmc/momentum (rv m sd)
  (let ((v ($data rv)))
    (cond (($tensorp v) ($resize! (sample/normal m sd ($count v)) v))
          (T (sample/gaussian m sd)))))

(defun hmc/update-parameters! (parameters momentums step-size)
  (loop :for p :in parameters
        :for m :in momentums
        :do (when (r/continuousp p)
              ($incf ($data p) ($* step-size m)))))

(defun hmc/update-momentums! (momentums gradients step-size)
  (loop :for g :in gradients
        :for i :from 0
        :do ($incf ($ momentums i) ($* step-size g))))

(defun hmc/leapfrog (candidates momentums posterior steps step-size)
  (let ((half-step-size (/ step-size 2))
        (cs (mapcar #'$clone candidates))
        (ps (->> candidates
                 (mapcar (lambda (c)
                           (if (r/continuousp c)
                               ($parameter ($data c))
                               ($data c)))))))
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
      (loop :repeat steps
            :do (progn
                  (hmc/update-momentums! momentums (dvdq cs) half-step-size)
                  (hmc/update-parameters! cs momentums step-size)
                  (hmc/update-momentums! momentums (dvdq cs) half-step-size)))
      (list cs momentums))))

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

(defun find-reasonable-start (parameters posterior-function)
  (labels ((posterior (vs) (apply posterior-function vs)))
    (let ((cs (mapcar (lambda (p)
                        (if (r/continuousp p)
                            ($parameter ($data p))
                            ($data p)))
                      parameters))
          (lr 0.1)
          (max-iter 200)
          (max-reset 10)
          (iter 0)
          (reset 0))
      (loop :while (and (< iter max-iter) (< reset max-reset))
            :for logp = (posterior cs)
            :do (if logp
                    (progn
                      ($neg logp)
                      ($amgd! cs lr)
                      (incf iter))
                    (progn
                      ($cg! cs)
                      (setf iter 0)
                      (incf reset)
                      (setf lr (* 0.5 lr)))))
      (when (>= reset (1- max-reset))
        (setf cs (mapcar (lambda (p)
                           (if (r/continuousp p)
                               ($parameter ($data p))
                               ($data p)))
                         parameters)))
      (mapcar (lambda (c p)
                (let ((np ($clone p)))
                  (if (r/continuousp np)
                      (setf ($data np) ($data c))
                      np)
                  np))
              cs parameters))))

(defun find-reasonable-epsilon (parameters posterior-function)
  (let ((eps 1)
        (log0.5 (log 0.5))
        (maxn-eps 10)
        (cs (find-reasonable-start ($clone parameters) posterior-function))
        (lf nil))
    (labels ((posterior (vs) (apply posterior-function vs))
             (vals (parameters) (mapcar #'$data parameters)))
      (let ((log-posterior0 (posterior (vals cs)))
            (log-posterior nil)
            (ms (mapcar (lambda (c) (hmc/momentum c 0 1)) cs)))
        (loop :for i :from 0
              :while (and (or (null lf) (null log-posterior)) (< i maxn-eps))
              :do (progn
                    (setf lf (hmc/leapfrog cs ms #'posterior 1 eps))
                    (unless lf (setf eps (* 0.5 eps)))
                    (when lf
                      (setf log-posterior (posterior (vals ($0 lf))))
                      (unless log-posterior (setf eps (* 0.5 eps))))))
        (when lf
          (let ((logp0 (- log-posterior0 (hmc/kinetic ms)))
                (logp (- (posterior (vals ($0 lf)))
                         (hmc/kinetic ($1 lf))))
                (alpha nil))
            (setf alpha (if (> (- logp logp0) log0.5) 1 -1))
            (loop :while (> (- logp logp0) (* log0.5 (- alpha)))
                  :do (progn
                        (setf eps (* eps (expt 2 alpha)))
                        (setf lf (hmc/leapfrog cs ms #'posterior 1 eps))
                        (unless lf (prn "null lf"))
                        (setf logp (- (posterior (vals ($0 lf))) (hmc/kinetic ($1 lf))))))
            (list eps cs (> (- logp logp0) (* log0.5 (- alpha))))))))))

(defun mcmc/hmc (parameters posterior-function
                 &key (iterations 2000) (tune-steps 100) (burn-in 1000) (thin 1))
  (labels ((posterior (vs) (apply posterior-function vs))
           (vals (parameters) (mapcar #'$data parameters)))
    (let ((nsize (+ burn-in iterations))
          (l (posterior (vals parameters)))
          (np ($count parameters))
          (m 0)
          (sd 1)
          (steps 10)
          (step-size-info (find-reasonable-epsilon parameters posterior-function)))
      (when l
        (let ((proposals (mapcar #'r/proposal parameters))
              (cs ($1 step-size-info))
              (traces (mcmc/traces np :burn-in burn-in :thin thin))
              (maxprob l)
              (naccepted 0)
              (nrejected 0)
              (step ($0 step-size-info))
              (fstep ($0 step-size-info))
              (sizer (hmc/step-sizer ($0 step-size-info)))
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
                :for (ncs nms) = (hmc/leapfrog cs ms #'posterior steps step)
                :for nl = (posterior (vals ncs))
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
