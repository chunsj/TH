(in-package :th.pp)

(defun hmc/momentum (rv m sd)
  (let ((v ($data rv)))
    (cond (($tensorp v) ($resize! (sample/normal m sd ($count v)) v))
          (T (sample/gaussian m sd)))))

(defun hmc/momentum-score (ms m sd)
  (loop :for momentum :in ms
        :summing (score/gaussian momentum m sd)))

(defun hmc/dvdq (potential parameters)
  (let ((ps (->> parameters
                 (mapcar (lambda (p)
                           (if (r/continuousp p)
                               ($parameter ($data p))
                               ($data p)))))))
    (funcall potential ps)
    (->> ps
         (mapcar (lambda (p)
                   (if ($parameterp p)
                       ($gradient p)
                       ($zero p)))))))

(defun update-parameters! (parameters momentums step-size)
  (loop :for p :in parameters
        :for m :in momentums
        :do (when (r/continuousp p)
              ($incf ($data p) ($* step-size m)))))

(defun update-momentums! (momentums gradients step-size)
  (loop :for g :in gradients
        :for i :from 0
        :do ($decf ($ momentums i) ($* step-size g))))

(defun leapfrog (candidates momentums potential path-length step-size)
  (let ((half-step-size (/ step-size 2))
        (cs (mapcar #'$clone candidates))
        (nr (max 0 (1- (round (/ path-length step-size))))))
    (update-momentums! momentums (hmc/dvdq potential cs) half-step-size)
    (loop :repeat nr
          :do (progn
                (update-parameters! cs momentums step-size)
                (update-momentums! momentums (hmc/dvdq potential cs) step-size)))
    (update-parameters! cs momentums step-size)
    (update-momentums! momentums (hmc/dvdq potential cs) half-step-size)
    (loop :for m :in momentums :do ($neg! m))
    (list (funcall potential (mapcar #'$data cs)) cs momentums)))

(defun hmc/accepted (h sm nh nsm)
  (when (and h sm nh nsm)
    (< (log (random 1D0)) (- (- h sm) (- nh nsm)))))

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
          (min-step 0.02))
      (incf errsum (- target-ratio paccept))
      (setf logstep (- mu (/ errsum (* (sqrt l) gamma))))
      (setf eta (expt l (- kappa)))
      (setf lavgstep (+ (* eta logstep) (* (- 1 eta) lavgstep)))
      (incf l)
      (list (max min-step (exp logstep)) (max min-step (exp lavgstep))))))

(defun mcmc/hmc (parameters posterior-function
                 &key (iterations 2000) (tune-steps 100) (burn-in 1000) (thin 1))
  (labels ((potential (vs)
             (let ((p (apply posterior-function vs)))
               (when p ($neg p))))
           (posterior (vs) (apply posterior-function vs))
           (vals (parameters) (mapcar #'$data parameters)))
    (let ((nsize (+ burn-in iterations))
          (h (potential (vals parameters)))
          (np ($count parameters))
          (m 0)
          (sd 1)
          (path-length 1D0)
          (step-size 0.1D0))
      (when h
        (let ((proposals (mapcar #'r/proposal parameters))
              (cs (mapcar #'$clone parameters))
              (traces (mcmc/traces np :burn-in burn-in :thin thin))
              (maxprob ($neg h))
              (naccepted 0)
              (nrejected 0)
              (step step-size)
              (fstep step-size)
              (sizer (hmc/step-sizer step-size))
              (tuning-done nil))
          (prn (format nil "[MCMC/HMC: TUNING..."))
          (loop :for trace :in traces
                :for candidate :in cs
                :do (trace/map! trace ($data candidate)))
          (loop :repeat nsize
                :for iter :from 1
                :for burning = (<= iter burn-in)
                :for tuneable = (zerop (rem iter tune-steps))
                :for ms = (->> cs
                               (mapcar (lambda (c) (hmc/momentum c m sd))))
                :for sm = (hmc/momentum-score ms m sd)
                :for (nh ncs nms) = (leapfrog cs ms #'potential path-length step)
                :for nsm = (hmc/momentum-score nms m sd)
                :do (let ((accept (hmc/accepted h sm nh nsm))
                          (tune (and (> iter 1) burning tuneable)))
                      (when tune
                        (loop :for proposal :in proposals :do (proposal/tune! proposal)))
                      (when burning
                        (let ((r (* 1D0 (/ naccepted (+ 1 naccepted nrejected)))))
                          (unless (and (> r 0.6) (< r 0.7))
                            (let ((stune (hmc/update-step-sizer! sizer r)))
                              (setf fstep (car stune))
                              (setf step (car stune))))))
                      (unless burning
                        (unless tuning-done
                          (prns (format nil " DONE. SAMPLING..."))
                          (setf tuning-done T))
                        (unless (= step fstep)
                          (setf step fstep)))
                      (when accept
                        (incf naccepted)
                        (loop :for tr :in traces
                              :for c :in ncs
                              :do (when (r/continuousp c)
                                    (trace/push! ($data c) tr)))
                        (setf cs ncs)
                        (setf h nh)
                        (when (> ($neg h) maxprob)
                          (setf maxprob ($neg h))
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
                                         (accepted (mh/accepted ($neg h) nprob lhr)))
                                    (r/accept! candidate proposal accepted)
                                    (trace/push! ($data candidate) trace)
                                    (when accepted
                                      (setf h ($neg nprob))
                                      (when (> nprob maxprob)
                                        (setf maxprob nprob)
                                        (trace/map! trace ($data candidate)))))))))
          (prns (format nil " FINISHED]~%"))
          traces)))))
