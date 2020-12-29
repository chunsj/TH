(in-package :th.pp)

(defgeneric proposal/scale! (proposal s))
(defgeneric proposal/tune! (proposal))
(defgeneric proposal/accepted! (proposal acceptedp))
(defgeneric proposal/propose (proposal value))

(defgeneric r/proposal (rv))
(defgeneric r/propose! (rv proposal))
(defgeneric r/accept! (rv proposal acceptedp))

(defclass mcmc/proposal ()
  ((accepted :initform 0)
   (rejected :initform 0)
   (factor :initform 1D0)
   (pvalue :initform nil :accessor $data)))

(defmethod proposal/scale! ((proposal mcmc/proposal) s)
  (with-slots (factor) proposal
    (setf factor s)))

(defmethod proposal/tune! ((proposal mcmc/proposal))
  (with-slots (accepted rejected factor) proposal
    (let ((total (+ accepted rejected)))
      (when (> total 0)
        (let ((r (/ accepted total)))
          (cond ((< r 0.001) (setf factor (* factor 0.1)))
                ((< r 0.05) (setf factor (* factor 0.5)))
                ((< r 0.2) (setf factor (* factor 0.9)))
                ((> r 0.95) (setf factor (* factor 10.0)))
                ((> r 0.75) (setf factor (* factor 2.0)))
                ((> r 0.5) (setf factor (* factor 1.1))))
          (setf accepted 0
                rejected 0))))))

(defmethod proposal/accepted! ((proposal mcmc/proposal) acceptedp)
  (with-slots (accepted rejected) proposal
    (if acceptedp
        (incf accepted)
        (incf rejected))
    proposal))

(defmethod r/propose! ((rv r/variable) (proposal mcmc/proposal))
  (let ((proposed (proposal/propose proposal ($data rv))))
    (setf ($data proposal) ($data rv))
    (setf ($data rv) (car proposed))
    (cdr proposed)))

(defmethod r/accept! ((rv r/variable) (proposal mcmc/proposal) acceptedp)
  (proposal/accepted! proposal acceptedp)
  (unless acceptedp
    (setf ($data rv) ($data proposal)
          ($data proposal) nil))
  rv)

(defclass proposal/gaussian (mcmc/proposal)
  ((scale :initform 1D0)))

(defun proposal/gaussian (&optional (scale 1D0))
  (let ((n (make-instance 'proposal/gaussian))
        (s scale))
    (with-slots (scale) n
      (setf scale s))
    n))

(defmethod proposal/propose ((proposal proposal/gaussian) value)
  (with-slots (scale factor) proposal
    (cons (sample/gaussian value (max 1E-7 (* factor scale))) 0D0)))

(defclass proposal/discrete-gaussian (mcmc/proposal)
  ((scale :initform 2D0)))

(defun proposal/discrete-gaussian (&optional (scale 2D0))
  (let ((n (make-instance 'proposal/discrete-gaussian))
        (s scale))
    (with-slots (scale) n
      (setf scale s))
    n))

(defmethod proposal/propose ((proposal proposal/discrete-gaussian) value)
  (with-slots (scale factor) proposal
    (cons (round (sample/gaussian value (max 1 (* factor scale)))) 0D0)))

(defmethod r/proposal ((rv r/discrete)) (proposal/discrete-gaussian))
(defmethod r/proposal ((rv r/continuous)) (proposal/gaussian))

(defclass rstat ()
  ((n :initform 0.0)
   (pm :initform 0.0)
   (nm :initform 0.0)
   (ps :initform 0.0)
   (ns :initform 0.0)))

(defun rstat () (make-instance 'rstat))

(defun rstat/push! (rstat x)
  (with-slots (n pm nm ps ns) rstat
    (incf n)
    (if (= n 1)
        (setf pm x
              nm x
              ps 0.0)
        (setf nm (+ pm (/ (- x pm) n))
              ns (+ ps (* (- x pm) (- x nm)))))
    (setf pm nm
          ps ns)))

(defun rstat/mean (rstat)
  (with-slots (n nm) rstat
    (if (> n 0)
        nm
        0.0)))

(defun rstat/variance (rstat)
  (with-slots (n ns) rstat
    (if (> n 1)
        (/ ns n)
        0.0)))

(defun mh/accepted (prob nprob log-hastings-ratio)
  (when (and prob nprob log-hastings-ratio)
    (let ((alpha (+ (- nprob prob) log-hastings-ratio)))
      (> alpha (log (random 1D0))))))

(defun mcmc/mh (parameters posterior-function
                &key (iterations 50000) (tune-steps 1000) (burn-in 10000) (thin 1))
  (labels ((posterior (vs) (apply posterior-function vs))
           (vals (parameters) (mapcar #'$data parameters)))
    (let ((prob (posterior (vals parameters)))
          (np ($count parameters)))
      (when prob
        (let ((proposals (mapcar #'r/proposal parameters))
              (traces (r/traces np :n iterations :burn-in burn-in :thin thin))
              (candidates (mapcar #'$clone parameters))
              (nsize (+ iterations burn-in))
              (pstep (round (/ iterations 20)))
              (maxprob prob)
              (naccepted 0)
              (tuning-done-reported nil))
          (prn (format nil "[MCMC/MH: TUNING"))
          (loop :for trace :in traces
                :for candidate :in candidates
                :do (setf ($data trace) ($clone ($data candidate))))
          (loop :repeat nsize
                :for iter :from 1
                :for burning = (<= iter burn-in)
                :for tuneable = (zerop (rem iter tune-steps))
                :do (let ((tune (and (> iter 1) burning tuneable)))
                      (when tune
                        (prns ".")
                        (loop :for proposal :in proposals :do (proposal/tune! proposal)))
                      (unless burning
                        (unless tuning-done-reported
                          (prns (format nil " DONE. SAMPLING"))
                          (setf tuning-done-reported T)))
                      (when (and (not burning) (zerop (rem (- iter burn-in) pstep)))
                        (prns "."))
                      (loop :for proposal :in proposals
                            :for candidate :in candidates
                            :for trace :in traces
                            :for lhr = (r/propose! candidate proposal)
                            :for nprob = (posterior (vals candidates))
                            :do (let ((accepted (mh/accepted prob nprob lhr)))
                                  (r/accept! candidate proposal accepted)
                                  (setf ($ trace (1- iter)) ($clone ($data candidate)))
                                  (when accepted
                                    (incf naccepted)
                                    (setf prob nprob)
                                    (when (> prob maxprob)
                                      (setf maxprob prob)
                                      (setf ($data trace) ($clone ($data candidate)))))))))
          (if (zerop naccepted)
              (prns (format nil " FAILED]~%"))
              (prns (format nil " DONE]~%")))
          traces)))))

(defun mcmc/mh-1 (parameters posterior-function
                  &key (iterations 50000) (burn-in 10000) (thin 1))
  (labels ((posterior (vs) (apply posterior-function vs))
           (vals (parameters) (mapcar #'$data parameters)))
    (let ((prob (posterior (vals parameters)))
          (np ($count parameters)))
      (when prob
        (let ((proposals (mapcar #'r/proposal parameters))
              (traces (r/traces np :n iterations :burn-in burn-in :thin thin))
              (candidates (mapcar #'$clone parameters))
              (rstats (loop :for p :in parameters :collect (rstat)))
              (nsize (+ iterations burn-in))
              (pstep (round (/ iterations 20)))
              (maxprob prob)
              (naccepted 0))
          (prn (format nil "[MCMC/MH: SAMPLING"))
          (loop :for trace :in traces
                :for candidate :in candidates
                :do (setf ($data trace) ($clone ($data candidate))))
          (loop :repeat nsize
                :for iter :from 1
                :for burning = (<= iter burn-in)
                :for tuneable = (>= iter 11)
                :do (progn
                      (when (and (not burning) (zerop (rem (- iter burn-in) pstep)))
                        (prns "."))
                      (loop :for proposal :in proposals
                            :for candidate :in candidates
                            :for trace :in traces
                            :for rs :in rstats
                            :for lhr = (r/propose! candidate proposal)
                            :for nprob = (posterior (vals candidates))
                            :do (let ((accepted (mh/accepted prob nprob lhr)))
                                  (r/accept! candidate proposal accepted)
                                  (setf ($ trace (1- iter)) ($clone ($data candidate)))
                                  (rstat/push! rs ($data candidate))
                                  (when tuneable
                                    (let ((g (* (* 2.4 2.4) (+ (rstat/variance rs) 0.05))))
                                      (proposal/scale! proposal (max 1.0 g))))
                                  (when accepted
                                    (incf naccepted)
                                    (setf prob nprob)
                                    (when (> prob maxprob)
                                      (setf maxprob prob)
                                      (setf ($data trace) ($clone ($data candidate)))))))))
          (if (zerop naccepted)
              (prns (format nil " FAILED]~%"))
              (prns (format nil " DONE]~%")))
          traces)))))

(defun mcmc/mh0 (parameters posterior-function
                 &key (iterations 50000) (scale 1) (burn-in 10000) (thin 1))
  (labels ((posterior (vs) (apply posterior-function vs))
           (vals (parameters) (mapcar #'$data parameters)))
    (let ((prob (posterior (vals parameters)))
          (np ($count parameters)))
      (when prob
        (let ((proposals (mapcar #'r/proposal parameters))
              (traces (r/traces np :n iterations :burn-in burn-in :thin thin))
              (candidates (mapcar #'$clone parameters))
              (nsize (+ iterations burn-in))
              (pstep (round (/ iterations 20)))
              (maxprob prob)
              (naccepted 0))
          (prn (format nil "[MCMC/MH0: SAMPLING"))
          (loop :for trace :in traces
                :for candidate :in candidates
                :do (setf ($data trace) ($clone ($data candidate))))
          (loop :for proposal :in proposals
                :for i :from 0
                :do (proposal/scale! proposal (if (numberp scale) scale ($ scale i))))
          (loop :repeat nsize
                :for iter :from 1
                :do (let ((burning (<= iter burn-in)))
                      (when (and (not burning) (zerop (rem (- iter burn-in) pstep)))
                        (prns "."))
                      (loop :for proposal :in proposals
                            :for candidate :in candidates
                            :for trace :in traces
                            :for lhr = (r/propose! candidate proposal)
                            :for nprob = (posterior (vals candidates))
                            :do (let ((accepted (mh/accepted prob nprob lhr)))
                                  (r/accept! candidate proposal accepted)
                                  (setf ($ trace (1- iter)) ($clone ($data candidate)))
                                  (when accepted
                                    (incf naccepted)
                                    (setf prob nprob)
                                    (when (> prob maxprob)
                                      (setf maxprob prob)
                                      (setf ($data trace) ($clone ($data candidate)))))))))
          (if (zerop naccepted)
              (prns (format nil " FAILED]~%"))
              (prns (format nil " DONE]~%")))
          traces)))))

(defun mcmc/mh1 (parameters posterior-function
                 &key (iterations 50000) (scale 1) (burn-in 10000) (thin 1))
  (labels ((posterior (vs) (apply posterior-function vs))
           (vals (parameters) (mapcar #'$data parameters)))
    (let ((prob (posterior (vals parameters)))
          (np ($count parameters)))
      (when prob
        (let ((proposals (mapcar #'r/proposal parameters))
              (traces (r/traces np :n iterations :burn-in burn-in :thin thin))
              (candidates (mapcar #'$clone parameters))
              (nsize (+ iterations burn-in))
              (pstep (round (/ iterations 20)))
              (maxprob prob)
              (naccepted 0))
          (prn (format nil "[MCMC/MH0: SAMPLING"))
          (loop :for trace :in traces
                :for candidate :in candidates
                :do (setf ($data trace) ($clone ($data candidate))))
          (loop :for proposal :in proposals
                :for i :from 0
                :do (proposal/scale! proposal (if (numberp scale) scale ($ scale i))))
          (loop :repeat nsize
                :for iter :from 1
                :do (let ((burning (<= iter burn-in))
                          (lhrs nil)
                          (lhr nil)
                          (nprob nil))
                      (when (and (not burning) (zerop (rem (- iter burn-in) pstep)))
                        (prns "."))
                      (setf lhrs (loop :for proposal :in proposals
                                       :for candidate :in candidates
                                       :collect (r/propose! candidate proposal)))
                      (setf lhr (reduce (lambda (s p) (when (and s p) (+ s p))) lhrs))
                      (setf nprob (posterior (vals candidates)))
                      (let ((accepted (and lhr nprob (mh/accepted prob nprob lhr))))
                        (loop :for proposal :in proposals
                              :for candidate :in candidates
                              :for trace :in traces
                              :do (progn
                                    (r/accept! candidate proposal accepted)
                                    (setf ($ trace (1- iter)) ($clone ($data candidate)))))
                        (when accepted
                          (incf naccepted)
                          (setf prob nprob)
                          (when (> prob maxprob)
                            (setf maxprob prob)
                            (loop :for candidate :in candidates
                                  :for trace :in traces
                                  :do (setf ($data trace) ($clone ($data candidate)))))))))
          (if (zerop naccepted)
              (prns (format nil " FAILED]~%"))
              (prns (format nil " DONE]~%")))
          traces)))))

(defun mcmc/mh2 (parameters posterior-function
                 &key (iterations 50000) (tune-steps 1000) (burn-in 10000) (thin 1))
  (labels ((posterior (vs) (apply posterior-function vs))
           (vals (parameters) (mapcar #'$data parameters)))
    (let ((prob (posterior (vals parameters)))
          (np ($count parameters)))
      (when prob
        (let ((proposals (mapcar #'r/proposal parameters))
              (traces (r/traces np :n iterations :burn-in burn-in :thin thin))
              (candidates (mapcar #'$clone parameters))
              (nsize (+ iterations burn-in))
              (pstep (round (/ iterations 20)))
              (maxprob prob)
              (naccepted 0)
              (tuning-done-reported nil))
          (prn (format nil "[MCMC/MH: TUNING"))
          (loop :for trace :in traces
                :for candidate :in candidates
                :do (setf ($data trace) ($clone ($data candidate))))
          (loop :repeat nsize
                :for iter :from 1
                :for burning = (<= iter burn-in)
                :for tuneable = (zerop (rem iter tune-steps))
                :do (let ((tune (and (> iter 1) burning tuneable))
                          (lhrs nil)
                          (lhr nil)
                          (nprob nil))
                      (when tune
                        (prns ".")
                        (loop :for proposal :in proposals :do (proposal/tune! proposal)))
                      (unless burning
                        (unless tuning-done-reported
                          (prns (format nil " DONE. SAMPLING"))
                          (setf tuning-done-reported T)))
                      (when (and (not burning) (zerop (rem (- iter burn-in) pstep)))
                        (prns "."))
                      (setf lhrs (loop :for proposal :in proposals
                                       :for candidate :in candidates
                                       :collect (r/propose! candidate proposal)))
                      (setf lhr (reduce (lambda (s p) (when (and s p) (+ s p))) lhrs))
                      (setf nprob (posterior (vals candidates)))
                      (let ((accepted (and lhr nprob (mh/accepted prob nprob lhr))))
                        (loop :for proposal :in proposals
                              :for candidate :in candidates
                              :for trace :in traces
                              :do (progn
                                    (r/accept! candidate proposal accepted)
                                    (setf ($ trace (1- iter)) ($clone ($data candidate)))))
                        (when accepted
                          (incf naccepted)
                          (setf prob nprob)
                          (when (> prob maxprob)
                            (setf maxprob prob)
                            (loop :for candidate :in candidates
                                  :for trace :in traces
                                  :do (setf ($data trace) ($clone ($data candidate)))))))))
          (if (zerop naccepted)
              (prns (format nil " FAILED]~%"))
              (prns (format nil " DONE]~%")))
          traces)))))

(defun mcmc/mh3 (parameters posterior-function
                 &key (iterations 50000) (tune-steps 1000) (burn-in 10000) (thin 1))
  (labels ((posterior (vs) (apply posterior-function vs))
           (vals (parameters) (mapcar #'$data parameters)))
    (let ((prob (posterior (vals parameters)))
          (np ($count parameters)))
      (when prob
        (let ((proposals (mapcar #'r/proposal parameters))
              (traces (r/traces np :n iterations :burn-in burn-in :thin thin))
              (candidates (mapcar #'$clone parameters))
              (rstats (loop :for p :in parameters :collect (rstat)))
              (nsize (+ iterations burn-in))
              (pstep (round (/ iterations 20.0)))
              (maxprob prob)
              (naccepted 0))
          (prn (format nil "[MCMC/MH: SAMPLING"))
          (loop :for trace :in traces
                :for candidate :in candidates
                :do (setf ($data trace) ($clone ($data candidate))))
          (loop :for proposal :in proposals
                :do (proposal/scale! proposal (* 5.0 5.0)))
          (loop :repeat nsize
                :for iter :from 1
                :for burning = (<= iter burn-in)
                :for tuneable = (and (not burning) (zerop (/ iter tune-steps)))
                :do (let ((lhrs nil)
                          (lhr nil)
                          (nprob nil))
                      (when (and (not burning) (zerop (rem (- iter burn-in) pstep)))
                        (prns "."))
                      (setf lhrs (loop :for proposal :in proposals
                                       :for candidate :in candidates
                                       :collect (r/propose! candidate proposal)))
                      (setf lhr (reduce (lambda (s p) (when (and s p) (+ s p))) lhrs))
                      (setf nprob (posterior (vals candidates)))
                      (let ((accepted (and lhr nprob (mh/accepted prob nprob lhr))))
                        (loop :for proposal :in proposals
                              :for candidate :in candidates
                              :for trace :in traces
                              :for rstat :in rstats
                              :do (progn
                                    (r/accept! candidate proposal accepted)
                                    (setf ($ trace (1- iter)) ($clone ($data candidate)))
                                    (rstat/push! rstat ($data candidate))
                                    (when (and tuneable (>= iter 11))
                                      (let ((g (* (* 2.4 2.4) (+ (rstat/variance rstat) 0.05))))
                                        (proposal/scale! proposal (max 1.0 g))))))
                        (when accepted
                          (incf naccepted)
                          (setf prob nprob)
                          (when (> prob maxprob)
                            (setf maxprob prob)
                            (loop :for candidate :in candidates
                                  :for trace :in traces
                                  :do (setf ($data trace) ($clone ($data candidate)))))))))
          (if (zerop naccepted)
              (prns (format nil " FAILED]~%"))
              (prns (format nil " DONE]~%")))
          traces)))))

(defun mcmc/mh4 (parameters posterior-function
                 &key (iterations 50000) (tune-steps 1000) (burn-in 10000) (thin 1))
  (labels ((posterior (vs) (apply posterior-function vs))
           (vals (parameters) (mapcar #'$data parameters)))
    (let ((prob (posterior (vals parameters)))
          (np ($count parameters)))
      (when prob
        (let ((proposals (mapcar #'r/proposal parameters))
              (traces (r/traces np :n iterations :burn-in burn-in :thin thin))
              (candidates (mapcar #'$clone parameters))
              (rstats (loop :for p :in parameters :collect (rstat)))
              (nsize (+ iterations burn-in))
              (pstep (round (/ iterations 20.0)))
              (maxprob prob)
              (naccepted 0)
              (tuning-done-reported nil))
          (prn (format nil "[MCMC/MH: TUNING"))
          (loop :for trace :in traces
                :for candidate :in candidates
                :do (setf ($data trace) ($clone ($data candidate))))
          (loop :repeat nsize
                :for iter :from 1
                :for burning = (<= iter burn-in)
                :for tuneable = (zerop (rem iter tune-steps))
                :do (let ((tune (and (> iter 1) burning tuneable))
                          (lhrs nil)
                          (lhr nil)
                          (nprob nil))
                      (when tune
                        (prns ".")
                        (loop :for proposal :in proposals :do (proposal/tune! proposal)))
                      (unless burning
                        (unless tuning-done-reported
                          (prns (format nil " DONE. SAMPLING"))
                          (setf tuning-done-reported T)))
                      (when (and (not burning) (zerop (rem (- iter burn-in) pstep)))
                        (prns "."))
                      (setf lhrs (loop :for proposal :in proposals
                                       :for candidate :in candidates
                                       :collect (r/propose! candidate proposal)))
                      (setf lhr (reduce (lambda (s p) (when (and s p) (+ s p))) lhrs))
                      (setf nprob (posterior (vals candidates)))
                      (let ((accepted (and lhr nprob (mh/accepted prob nprob lhr))))
                        (loop :for proposal :in proposals
                              :for candidate :in candidates
                              :for trace :in traces
                              :for rstat :in rstats
                              :do (progn
                                    (r/accept! candidate proposal accepted)
                                    (setf ($ trace (1- iter)) ($clone ($data candidate)))
                                    (when (not burning)
                                      (rstat/push! rstat ($data candidate))
                                      (when (>= iter (+ burn-in 11))
                                        (let ((g (* (* 2.4 2.4) (+ (rstat/variance rstat) 0.05))))
                                          (proposal/scale! proposal g))))))
                        (when accepted
                          (incf naccepted)
                          (setf prob nprob)
                          (when (> prob maxprob)
                            (setf maxprob prob)
                            (loop :for candidate :in candidates
                                  :for trace :in traces
                                  :do (setf ($data trace) ($clone ($data candidate)))))))))
          (if (zerop naccepted)
              (prns (format nil " FAILED]~%"))
              (prns (format nil " DONE]~%")))
          traces)))))
