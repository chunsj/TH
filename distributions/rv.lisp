(in-package th.distributions)

(defgeneric $logp (rv))
(defgeneric $observation (rv))
(defgeneric $continuousp (rv))
(defgeneric $sample! (rv))
(defgeneric $supportp (rv v))
(defgeneric $supportedp (rv))

(defgeneric $propose (proposal rv))
(defgeneric $accepted! (proposal))
(defgeneric $rejected! (proposal))
(defgeneric $tune! (proposal))

(defmethod $logp ((rv T)) 0)
(defmethod $sample! ((rv T)) rv)
(defmethod $supportp ((rv T) v) T)
(defmethod $supportedp ((rv T)) T)

(defmethod $logp ((rvs list))
  (let ((rvs0 (remove-duplicates rvs)))
    (unless (some (lambda (r) (not ($supportedp r))) rvs0)
      (loop :for rv :in rvs0 :summing ($logp rv)))))

(defclass rv/variable ()
  ((value :initform nil)
   (observedp :initform nil)))

(defun rv/variable (v &key observedp)
  (let ((n (make-instance 'rv/variable))
        (obp observedp))
    (with-slots (value observedp) n
      (setf value v
            observedp obp))
    n))

(defmethod $data ((rv rv/variable))
  (with-slots (value) rv
    value))

(defmethod (setf $data) (v (rv rv/variable))
  (with-slots (value) rv
    (setf value v)))

(defmethod $clone ((rv rv/variable))
  (let ((n (make-instance (class-of rv))))
    (with-slots (value observedp) rv
      (let ((v value)
            (o observedp))
        (with-slots (value observedp) n
          (setf value v
                observedp o))))
    n))

(defmethod $logp ((rv rv/variable)) nil)
(defmethod $sample! ((rv rv/variable)) nil)

(defmethod $continuousp ((rv rv/variable)) T)

(defmethod $observation ((rv rv/variable))
  (with-slots (value observedp) rv
    (when observedp
      value)))

(defmethod (setf $observation) (observation (rv rv/variable))
  (when observation
    (with-slots (value observedp) rv
      (setf value observation
            observedp T))
    observation))

(defmethod print-object ((rv rv/variable) stream)
  (with-slots (observedp value) rv
    (if ($continuousp rv)
        (format stream "~8F~A" value (if (not observedp) "?" ""))
        (format stream "~8D~A" value (if (not observedp) "?" "")))))

(defmethod $supportedp ((rv rv/variable))
  (with-slots (value) rv
    ($supportp rv value)))

(defclass rv/discrete-uniform (rv/variable)
  ((lower :initform 0)
   (upper :initform 9)))

(defun rv/discrete-uniform (&key (lower 0) (upper 9) observation)
  (let ((l lower)
        (u upper)
        (n (make-instance 'rv/discrete-uniform)))
    (setf ($observation n) observation)
    (with-slots (lower upper) n
      (setf lower l
            upper u)
      ($sample! n))
    n))

(defmethod $continuousp ((rv rv/discrete-uniform)) nil)

(defmethod $clone ((rv rv/discrete-uniform))
  (let ((n (call-next-method rv)))
    (with-slots (lower upper) rv
      (let ((l ($clone lower))
            (u ($clone upper)))
        (with-slots (lower upper) n
          (setf lower l
                upper u))))
    n))

(defun supportp-discrete-uniform (lower upper v)
  (cond ((listp v)
         (cond ((and (listp lower) (listp upper)
                     (= ($count v) ($count lower) ($count upper)))
                (not (some (lambda (val l u) (or (< val ($data l)) (> val ($data u))))
                           v lower upper)))
               (T (not (some (lambda (val) (or (< val ($data lower)) (> val ($data upper))))
                             v)))))
        (T (and (>= v ($data lower)) (<= v ($data upper))))))

(defmethod $supportp ((rv rv/discrete-uniform) v)
  (with-slots (lower upper) rv
    (supportp-discrete-uniform lower upper v)))

(defun sample-discrete-uniform (lower upper)
  (+ ($data lower) (1- ($sample/dice 1 (1+ (- ($data upper) ($data lower)))))))

(defmethod $sample! ((rv rv/discrete-uniform))
  (with-slots (value observedp lower upper) rv
    (unless observedp
      (cond ((and (listp lower) (listp upper) (= ($count lower) ($count upper)))
             (setf value (mapcar (lambda (l u) (sample-discrete-uniform l u)) lower upper)))
            (T (setf value (sample-discrete-uniform lower upper))))
      value)))

(defun logp-discrete-uniform (value lower upper)
  (when (supportp-discrete-uniform lower upper value)
    (cond ((and (listp value) (listp lower) (listp upper)
                (= ($count value) ($count lower) ($count upper)))
           (loop :for v :in value
                 :for l :in lower
                 :for u :in upper
                 :summing ($ll/uniform v ($data l) ($data u))))
          (T ($ll/uniform value ($data lower) ($data upper))))))

(defmethod $logp ((rv rv/discrete-uniform))
  (with-slots (value lower upper) rv
    (let ((ll (logp-discrete-uniform value lower upper))
          (llower ($logp lower))
          (lupper ($logp upper)))
      (when (and ll llower lupper)
        (+ ll llower lupper)))))

(defclass rv/exponential (rv/variable)
  ((rate :initform 1D0)))

(defun rv/exponential (&key (rate 1D0) observation)
  (let ((r rate)
        (n (make-instance 'rv/exponential)))
    (setf ($observation n) observation)
    (with-slots (rate value) n
      (setf rate r)
      ($sample! n))
    n))

(defmethod $clone ((rv rv/exponential))
  (let ((n (call-next-method rv)))
    (with-slots (rate) rv
      (let ((r ($clone rate)))
        (with-slots (rate) n
          (setf rate r))))
    n))

(defun supportp-exponential (v)
  (cond ((listp v) (not (some (lambda (val) (< val 0)) v)))
        (T (> v 0))))

(defmethod $supportp ((rv rv/exponential) v) (supportp-exponential v))

(defmethod $sample! ((rv rv/exponential))
  (with-slots (value observedp rate) rv
    (unless observedp
      (cond ((listp rate) (setf value (mapcar (lambda (r) ($sample/exponential 1 r)) rate)))
            (T (setf value ($sample/exponential 1 rate))))
      value)))

(defun logp-exponential (value rate)
  (when (supportp-exponential value)
    (cond ((and (listp value) (listp rate) (= ($count value) ($count rate)))
           (loop :for v :in value
                 :for r :in rate
                 :summing ($ll/exponential v ($data r))))
          (T ($ll/exponential value ($data rate))))))

(defmethod $logp ((rv rv/exponential))
  (with-slots (value rate) rv
    (let ((ll (logp-exponential value rate))
          (lrate ($logp rate)))
      (when (and ll lrate)
        (+ ll lrate)))))

(defclass rv/poisson (rv/variable)
  ((rate :initform 1D0)))

(defun rv/poisson (&key (rate 1D0) observation)
  (let ((r rate)
        (n (make-instance 'rv/poisson)))
    (setf ($observation n) observation)
    (with-slots (rate value) n
      (setf rate r)
      ($sample! n))
    n))

(defmethod $clone ((rv rv/poisson))
  (let ((n (call-next-method rv)))
    (with-slots (rate) rv
      (let ((r ($clone rate)))
        (with-slots (rate) n
          (setf rate r))))
    n))

(defmethod $sample! ((rv rv/poisson))
  (with-slots (value observedp rate) rv
    (unless observedp
      (cond ((listp rate) (setf value (mapcar (lambda (r) ($sample/poisson 1 r)) rate)))
            (T (setf value ($sample/poisson 1 rate))))
      value)))

(defun supportp-poisson (v)
  (cond ((listp v) (not (some (lambda (val) (< val 0)) v)))
        (T (> v 0))))

(defmethod $supportp ((rv rv/poisson) v) (supportp-poisson v))

(defun logp-poisson (value rate)
  (when (supportp-poisson value)
    (cond ((and (listp value) (listp rate) (= ($count value) ($count rate)))
           (loop :for v :in value
                 :for r :in rate
                 :summing ($ll/poisson v ($data r))))
          (T ($ll/poisson value ($data rate))))))

(defmethod $logp ((rv rv/poisson))
  (with-slots (value rate) rv
    (let ((ll (logp-poisson value rate))
          (lrate ($logp rate)))
      (when (and ll lrate)
        (+ ll lrate)))))

(defclass proposal ()
  ((accepted :initform 0)
   (rejected :initform 0)
   (factor :initform 1D0)))

(defmethod $tune! ((proposal proposal))
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

(defmethod $accepted! ((p proposal))
  (with-slots (accepted) p
    (incf accepted)))

(defmethod $rejected! ((p proposal))
  (with-slots (rejected) p
    (incf rejected)))

(defclass proposal/uniform (proposal)
  ((scale :initform 1D0)))

(defun proposal/uniform (&optional (scale 1D0))
  (let ((n (make-instance 'proposal/uniform))
        (s scale))
    (with-slots (scale) n
      (setf scale s))
    n))

(defmethod $propose ((proposal proposal/uniform) (rv rv/variable))
  (let ((n ($clone rv)))
    (with-slots (scale factor) proposal
      (with-slots (value) n
        (let ((new-value ($sample/uniform 1
                                          (+ value (* -1 factor scale))
                                          (+ value (* factor scale)))))
          (setf value new-value))))
    (cons n 0D0)))

(defclass proposal/gaussian (proposal)
  ((scale :initform 1D0)))

(defun proposal/gaussian (&optional (scale 1D0))
  (let ((n (make-instance 'proposal/gaussian))
        (s scale))
    (with-slots (scale) n
      (setf scale s))
    n))

(defmethod $propose ((proposal proposal/gaussian) (rv rv/variable))
  (let ((n ($clone rv)))
    (with-slots (scale factor) proposal
      (with-slots (value) n
        (let ((new-value ($sample/gaussian 1 value (* factor scale))))
          (setf value new-value))))
    (cons n 0D0)))

(defclass proposal/discrete-gaussian (proposal/gaussian)
  ())

(defun proposal/discrete-gaussian (&optional (scale 1D0))
  (let ((n (make-instance 'proposal/discrete-gaussian))
        (s scale))
    (with-slots (scale factor) n
      (setf scale s
            factor 1D0))
    n))

(defmethod $propose ((proposal proposal/discrete-gaussian) (rv rv/variable))
  (let ((n ($clone rv)))
    (with-slots (scale factor) proposal
      (with-slots (value) n
        (let ((new-value (round ($sample/gaussian 1 value (* factor scale)))))
          (setf value new-value))))
    (cons n 0D0)))

(defun autocov (series &optional (lag 1) divnk)
  (let ((n ($count series))
        (zbar ($mean series))
        (lseries (nthcdr lag series)))
    (/ (loop :for i :from 0 :below (- n lag)
             :for v :in series
             :for lv :in lseries
             :for vz = (- v zbar)
             :for lz = (- lv zbar)
             :summing (* vz lz))
       (if divnk (- n lag) n))))

(defun autocorr (series &optional (lag 1))
  (/ (autocov series lag) (autocov series 0)))

(defun min-interval (vs alpha)
  (let* ((mn nil)
         (mx nil)
         (n ($count vs))
         (start 0)
         (end (round (* n (- 1 alpha))))
         (min-width most-positive-single-float))
    (loop :while (< end n)
          :for hi = ($ vs end)
          :for lo = ($ vs start)
          :for width = (- hi lo)
          :do (progn
                (when (< width min-width)
                  (setf min-width width
                        mn lo
                        mx hi))
                (incf start)
                (incf end)))
    (cons mn mx)))

(defun interval-zscores (vs a b &optional (intervals 20))
  (let* ((end (1- ($count vs)))
         (hend (/ end 2))
         (sindices (loop :for i :from 0 :below (round hend) :by (round (/ hend intervals))
                         :collect i)))
    (loop :for start :in sindices
          :for slice-a = (subseq vs start (+ start (round (* a (- end start)))))
          :for slice-b = (subseq vs (round (- end (* b (- end start)))))
          :for zn = (- ($mean slice-a) ($mean slice-b))
          :for zd = (+ ($square ($sd slice-a)) ($square ($sd slice-b)))
          :collect (cons start (/ zn zd)))))

(defgeneric $mcmc/trace (trace))
(defgeneric $mcmc/push! (data trace))
(defgeneric $mcmc/mle (trace))
(defgeneric $mcmc/mean (trace))
(defgeneric $mcmc/sd (trace))
(defgeneric $mcmc/count (trace))
(defgeneric $mcmc/error (trace))
(defgeneric $mcmc/autocorrelation (trace &key maxlag))
(defgeneric $mcmc/quantiles (trace))
(defgeneric $mcmc/hpd (trace alpha))
(defgeneric $mcmc/geweke (trace &key first last intervals))
(defgeneric $mcmc/summary (trace))
(defgeneric $mcmc/aic (deviance k))
(defgeneric $mcmc/dic (traces deviance likelihoodfn))

(defclass mcmc/trace ()
  ((traces :initform nil)
   (burn-ins :initform 0)
   (thin :initform 0)
   (mlev :initform nil)))

(defmethod print-object ((trc mcmc/trace) stream)
  (with-slots (traces burn-ins thin mlev) trc
    (cond ((null mlev) (format stream "<MCMC/TRACE UNINITIALIZED>"))
          (T (format stream "<MCMC/TRACE [~A/~A] ~A>" burn-ins thin mlev)))))

(defun mcmc/trace (rv &key (burn-ins 0) (thin 1))
  (let ((n (make-instance 'mcmc/trace))
        (nb burn-ins)
        (th thin))
    (with-slots (burn-ins thin mlev) n
      (setf burn-ins nb
            thin th
            mlev rv))
    n))

(defmethod $mcmc/trace ((trace mcmc/trace))
  (with-slots (traces burn-ins thin) trace
    (let ((rtraces (nthcdr burn-ins (reverse traces))))
      (loop :for lst :on rtraces :by (lambda (l) (nthcdr thin l))
            :collect (car lst)))))

(defmethod $mcmc/push! (data (trace mcmc/trace))
  (with-slots (traces) trace
    (push data traces))
  data)

(defmethod $count ((trace mcmc/trace))
  (with-slots (traces) trace
    ($count traces)))

(defun set-mlev! (trace v)
  (with-slots (mlev) trace
    (setf mlev v)))

(defmethod $mcmc/mle ((trace mcmc/trace))
  (with-slots (mlev) trace
    mlev))

(defmethod $mcmc/mean ((trace mcmc/trace))
  (let ((trc ($mcmc/trace trace)))
    ($mean (mapcar #'$data trc))))

(defmethod $mcmc/sd ((trace mcmc/trace))
  (let ((trc ($mcmc/trace trace)))
    ($sd (mapcar #'$data trc))))

(defmethod $mcmc/error ((trace mcmc/trace))
  (let ((n ($mcmc/count trace))
        (sd ($mcmc/sd trace)))
    (when (>= n 1) (/ sd (sqrt n)))))

(defmethod $mcmc/count ((trace mcmc/trace))
  ($count ($mcmc/trace trace)))

(defmethod $mcmc/autocorrelation ((trace mcmc/trace) &key (maxlag 100))
  (let ((trcvs (mapcar #'$data ($mcmc/trace trace))))
    (loop :for k :from 0 :below (1+ maxlag)
          :collect (autocorr trcvs k))))

(defmethod $mcmc/quantiles ((trace mcmc/trace))
  (let* ((trcvs (mapcar #'$data ($mcmc/trace trace)))
         (n ($count trcvs))
         (qlist '(2.5 25 50 75 97.5)))
    (when (> n 10)
      (let ((vs (sort trcvs #'<)))
        (loop :for q :in qlist
              :for ridx = (round (* n (/ q 100)))
              :collect (let ((i ridx))
                         (when (< i 0) (setf i 0))
                         (when (> i (1- n)) n)
                         (cons q ($ vs i))))))))

(defmethod $mcmc/hpd ((trace mcmc/trace) alpha)
  (let ((vs (sort (mapcar #'$data ($mcmc/trace trace)) #'<)))
    (when (> ($count vs) 10)
      (min-interval vs alpha))))

(defmethod $mcmc/geweke ((trace mcmc/trace) &key (first 0.1) (last 0.5) (intervals 20))
  (when (< (+ first last))
    (let ((vs (mapcar #'$data ($mcmc/trace trace))))
      (when (> ($count vs) intervals)
        (interval-zscores vs first last intervals)))))

(defmethod $mcmc/summary ((trace mcmc/trace))
  (let ((quantiles ($mcmc/quantiles trace))
        (n ($mcmc/count trace))
        (sd ($mcmc/sd trace))
        (x ($data ($mcmc/mle trace)))
        (m ($mcmc/mean trace))
        (err ($mcmc/error trace))
        (hpd ($mcmc/hpd trace 0.05))
        (acr ($mcmc/autocorrelation trace))
        (geweke (let ((gvs (mapcar #'cdr ($mcmc/geweke trace))))
                  (cons (apply #'min gvs) (apply #'max gvs)))))
    (list :count n
          :mle x
          :mean m
          :sd sd
          :error err
          :hpd-95 hpd
          :quantiles quantiles
          :autocurrelation (subseq acr 0 (min 21 ($count acr)))
          :geweke geweke)))

(defmethod $mcmc/aic ((deviance mcmc/trace) k)
  (let ((quantiles (loop :for qi :in ($mcmc/quantiles deviance)
                         :for q = (car qi)
                         :for v = (+ (* 2 k) (cdr qi))
                         :collect (cons q v)))
        (n ($mcmc/count deviance))
        (sd ($mcmc/sd deviance))
        (m ($mcmc/mean deviance))
        (err ($mcmc/error deviance))
        (hpd (let ((mi ($mcmc/hpd deviance 0.05)))
               (cons (+ (* 2 k) (car mi))
                     (+ (* 2 k) (cdr mi))))))
    (list :count n
          :mean m
          :sd sd
          :error err
          :hpd-95 hpd
          :quantiles quantiles)))

(defmethod $mcmc/dic ((traces list) (deviance mcmc/trace) likelihoodfn)
  (let ((parameters (mapcar (lambda (trc) ($clone (car ($mcmc/trace trc)))) traces))
        (means (mapcar (lambda (trc) ($mcmc/mean trc)) traces)))
    (loop :for p :in parameters
          :for m :in means
          :do (setf ($data p) (if ($continuousp p) m (round m))))
    (let ((mean-deviance ($mcmc/mean deviance))
          (ll (apply likelihoodfn parameters)))
      (when ll
        (+ (* 2 mean-deviance)
           (* 2 ll))))))

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
