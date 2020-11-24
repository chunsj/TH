(in-package :th.pp)

(defun r/momentum (rv m sd)
  (let ((v (r/value rv)))
    (cond (($tensorp v) ($resize! (sample/normal m sd ($count v)) v))
          (T (sample/gaussian m sd)))))

(defun momentum-score (ms m sd)
  (loop :for momentum :in ms
        :summing (score/gaussian momentum m sd)))

(defun dvdq (potential parameters)
  (let ((ps (->> parameters
                 (mapcar (lambda (p)
                           (if (r/continuousp p)
                               ($parameter (r/value p))
                               (r/value p)))))))
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
              ($add! (r/value p) ($* step-size m)))))

(defun update-momentums! (momentums gradients step-size)
  (loop :for g :in gradients
        :for m :in momentums
        :do ($sub! m ($* step-size g))))

(defun leapfrog (candidates momentums potential path-length step-size)
  (let ((half-step-size (/ step-size 2))
        (cs (mapcar #'$clone candidates)))
    (update-momentums! momentums (dvdq potential cs) half-step-size)
    (loop :repeat (1- (round (/ path-length step-size)))
          :do (progn
                (update-parameters! cs momentums step-size)
                (update-momentums! momentums (dvdq potential cs) step-size)))
    (update-parameters! cs momentums step-size)
    (update-momentums! momentums (dvdq potential cs) half-step-size)
    (loop :for m :in momentums :do ($neg! m))
    (list (funcall potential (mapcar #'r/value cs)) cs momentums)))

(defun hmc/accepted (h sm nh nsm)
  (when (and h sm nh nsm)
    (< (log (random 1D0)) (- (- h sm) (- nh nsm)))))

(defun mcmc/hmc (parameters posterior-function
                 &key (iterations 50000) (burn-in 1000) (thin 1) (path-length 1) (step-size 0.1))
  (labels ((potential (vs) ($neg (funcall posterior-function vs)))
           (vals (parameters) (mapcar #'r/value parameters)))
    (let ((nsize (+ burn-in iterations))
          (h (potential (vals parameters)))
          (np ($count parameters))
          (m 0)
          (sd 1))
      (when h
        (let ((cs (mapcar #'$clone parameters))
              (traces (mcmc/traces np :burn-in burn-in :thin thin)))
          (loop :repeat nsize
                :for ms = (->> cs
                               (mapcar (lambda (c) (r/momentum c m sd))))
                :for sm = (momentum-score ms m sd)
                :for (nh ncs nms) = (leapfrog cs ms #'potential path-length step-size)
                :for nsm = (momentum-score nms m sd)
                :do (let ((accept (hmc/accepted h sm nh nsm)))
                      (when accept
                        (loop :for tr :in traces
                              :for c :in ncs
                              :do (trace/push! (r/value c) tr))
                        (setf cs ncs)
                        (setf h nh))
                      (unless accept
                        (loop :for tr :in traces
                              :for c :in cs
                              :do (trace/push! (r/value c) tr)))))
          traces)))))
