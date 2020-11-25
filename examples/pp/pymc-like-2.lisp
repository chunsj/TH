(defpackage :pymc-like-2
  (:use #:common-lisp
        #:mu
        #:th
        #:th.pp))

(in-package :pymc-like-2)

(defgeneric $logp (rv))
(defgeneric $observation (rv))
(defgeneric $continuousp (rv))

(defgeneric $propose (proposal rv))
(defgeneric $accepted! (proposal))
(defgeneric $rejected! (proposal))
(defgeneric $tune! (proposal))

(defmethod $logp ((rv T)) 0)

(defclass rv/variable ()
  ((value :initform nil)
   (observedp :initform nil)))

(defmethod $data ((rv rv/variable))
  (with-slots (value) rv
    value))

(defmethod $clone ((rv rv/variable))
  (let ((n (make-instance (class-of rv))))
    (with-slots (value observedp) rv
      (let ((v value)
            (o observedp))
        (with-slots (value observedp) n
          (setf value v
                observedp o))))
    n))

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

(defclass rv/discrete-uniform (rv/variable)
  ((lower :initform 0)
   (upper :initform 9)))

(defun rv/discrete-uniform (&key (lower 0) (upper 9) observation)
  (let ((l lower)
        (u upper)
        (n (make-instance 'rv/discrete-uniform)))
    (setf ($observation n) observation)
    (with-slots (lower upper value) n
      (setf lower l
            upper u)
      (unless value
        (setf value (th.pp:sample/discrete-uniform lower upper))))
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

(defmethod $logp ((rv rv/discrete-uniform))
  (with-slots (value lower upper) rv
    (let ((ll (score/uniform value ($data lower) ($data upper)))
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
      (unless value
        (setf value (sample/exponential rate))))
    n))

(defmethod $clone ((rv rv/exponential))
  (let ((n (call-next-method rv)))
    (with-slots (rate) rv
      (let ((r ($clone rate)))
        (with-slots (rate) n
          (setf rate r))))
    n))

(defmethod $logp ((rv rv/exponential))
  (with-slots (value rate) rv
    (let ((ll (score/exponential value ($data rate)))
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
      (unless value
        (setf value (sample/poisson ($data rate)))))
    n))

(defmethod $clone ((rv rv/poisson))
  (let ((n (call-next-method rv)))
    (with-slots (rate) rv
      (let ((r ($clone rate)))
        (with-slots (rate) n
          (setf rate r))))
    n))

(defmethod $logp ((rv rv/poisson))
  (with-slots (value rate) rv
    (let ((ll (score/poisson value ($data rate)))
          (lrate ($logp rate)))
      (when (and ll lrate)
        (+ ll lrate)))))

(defclass proposal ()
  ((accepted :initform 0)
   (rejected :initform 0)))

(defmethod $tune! ((p proposal)))

(defmethod $accepted! ((p proposal))
  (with-slots (accepted) p
    (incf accepted)))

(defmethod $rejected! ((p proposal))
  (with-slots (rejected) p
    (incf rejected)))

(defclass proposal/gaussian (proposal)
  ((scale :initform 1D0)
   (factor :initform 0.1D0)))

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
        (let ((new-value (sample/gaussian value (* factor scale))))
          (setf value new-value))))
    (cons n 0D0)))

(defmethod $tune! ((proposal proposal/gaussian))
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
        (let ((new-value (round (sample/gaussian value (* factor scale)))))
          (setf value new-value))))
    (cons n 0D0)))

(defun mh (nsamples parameters likelihoodfn &key (max-iterations 100000) (tune-steps 1000)
                                              verbose)
  (let ((old-likelihood (apply likelihoodfn parameters))
        (old-parameters (mapcar #'$clone parameters))
        (proposals (mapcar (lambda (p)
                             (if ($continuousp p)
                                 (proposal/gaussian)
                                 (proposal/discrete-gaussian)))
                           parameters))
        (accepted nil)
        (naccepted 0)
        (max-likelihood most-negative-single-float)
        (mle-parameters parameters)
        (done nil))
    (when verbose
      (prn (format nil "*    START MLE: ~10,4F" old-likelihood))
      (prn (format nil "            ~{~A ~}" old-parameters)))
    (when old-likelihood
      (loop :while (not done)
            :repeat (max nsamples max-iterations)
            :for iter :from 1
            :for proposed = (mapcar (lambda (proposal parameter)
                                      ($propose proposal parameter))
                                    proposals
                                    old-parameters)
            :for new-parameters = (mapcar #'car proposed)
            :for log-hastings-ratio = (reduce (lambda (s lhr)
                                                (when lhr (+ s lhr)))
                                              (mapcar #'cdr proposed))
            :for new-likelihood = (apply likelihoodfn new-parameters)
            :do (let ((valid-result (and log-hastings-ratio new-likelihood)))
                  (if valid-result
                      (let ((lmh (+ (- new-likelihood old-likelihood) log-hastings-ratio))
                            (lu (log (random 1D0))))
                        (when (> lmh lu)
                          (push (mapcar #'$clone new-parameters) accepted)
                          (incf naccepted)
                          (setf old-parameters new-parameters
                                old-likelihood new-likelihood)
                          (when (> new-likelihood max-likelihood)
                            (setf max-likelihood new-likelihood
                                  mle-parameters new-parameters)
                            (when verbose
                              (prn (format nil "* ~8,D MLE: ~10,4F" naccepted max-likelihood))
                              (prn (format nil "            ~{~A ~}" new-parameters))))
                          (loop :for proposal :in proposals
                                :do ($accepted! proposal))
                          (when (>= naccepted nsamples) (setf done T))))
                      (loop :for proposal :in proposals
                            :do ($rejected! proposal)))
                  (when (zerop (rem iter tune-steps))
                    (loop :for proposal :in proposals
                          :do ($tune! proposal))))))
    (when verbose
      (prn "* MLE PARAMETERS")
      (prn (format nil "~{~A ~}" mle-parameters)))
    accepted))

(defvar *disasters* '(4 5 4 0 1 4 3 4 0 6 3 3 4 0 2 6
                      3 3 5 4 5 3 1 4 4 1 5 5 3 4 2 5
                      2 2 3 4 2 1 3 2 2 1 1 1 1 3 0 0
                      1 0 1 1 0 0 3 1 0 3 2 2 0 1 1 1
                      0 1 0 1 0 0 0 2 1 0 0 0 1 1 0 2
                      3 3 1 1 2 1 1 1 1 2 4 2 0 0 1 4
                      0 0 0 1 0 0 0 0 0 1 0 0 1 0 1))
(defvar *rate* (/ 1D0 ($mean *disasters*)))

(defun disaster-likelihood (switch-point early-mean late-mean)
  (let ((ls ($logp switch-point)))
    (when ls
      (let ((disasters-early (tensor (subseq *disasters* 0 ($data switch-point))))
            (disasters-late (tensor (subseq *disasters* ($data switch-point)))))
        (let ((d1 (rv/poisson :rate early-mean :observation disasters-early))
              (d2 (rv/poisson :rate late-mean :observation disasters-late)))
          (let ((ld1 ($logp d1))
                (ld2 ($logp d2)))
            (when (and ls ld1 ld2)
              (+ ls ld1 ld2))))))))

;; MLE: 41, 3, 1
(time
 (let ((switch-point (rv/discrete-uniform :lower 0 :upper (1- ($count *disasters*))))
       (early-mean (rv/exponential :rate *rate*))
       (late-mean (rv/exponential :rate *rate*)))
   (let* ((accepted (mh 10000 (list switch-point early-mean late-mean) #'disaster-likelihood
                        :verbose T))
          (na ($count accepted))
          (ns (round (* 0.2 na)))
          (selected (subseq accepted 0 ns)))
     (prn "SELECTED:" ns "/" na)
     (let ((ss (mapcar (lambda (ps) ($data ($0 ps))) selected))
           (es (mapcar (lambda (ps) ($data ($1 ps))) selected))
           (ls (mapcar (lambda (ps) ($data ($2 ps))) selected)))
       (prn "MEAN/SD[0]:" (round ($mean ss)) "/" (format nil "~8F" ($sd ss)))
       (prn "MEAN/SD[1]:" (round ($mean es)) "/" (format nil "~8F" ($sd es)))
       (prn "MEAN/SD[2]:" (round ($mean ls)) "/" (format nil "~8F" ($sd ls)))))))

;; FOR SMS example
;; https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/masterv/Chapter1_Introduction/Ch1_Introduction_PyMC2.ipynb
(defvar *sms* (->> (slurp "./data/sms.txt")
                   (mapcar #'parse-float)
                   (mapcar #'round)))
(defvar *srate* (/ 1D0 ($mean *sms*)))

(defun sms-likelihood (switch-point early-mean late-mean)
  (let ((ls ($logp switch-point)))
    (when ls
      (let ((disasters-early (tensor (subseq *sms* 0 ($data switch-point))))
            (disasters-late (tensor (subseq *sms* ($data switch-point)))))
        (let ((d1 (rv/poisson :rate early-mean :observation disasters-early))
              (d2 (rv/poisson :rate late-mean :observation disasters-late)))
          (let ((ld1 ($logp d1))
                (ld2 ($logp d2)))
            (when (and ls ld1 ld2)
              (+ ls ld1 ld2))))))))

;; MLE: 45, 18, 23
(time
 (let ((switch-point (rv/discrete-uniform :lower 0 :upper (1- ($count *sms*))))
       (early-mean (rv/exponential :rate *srate*))
       (late-mean (rv/exponential :rate *srate*)))
   (let* ((accepted (mh 10000 (list switch-point early-mean late-mean) #'sms-likelihood
                        :verbose T))
          (na ($count accepted))
          (ns (round (* 0.2 na)))
          (selected (subseq accepted 0 ns)))
     (prn "SELECTED:" ns "/" na)
     (let ((ss (mapcar (lambda (ps) ($data ($0 ps))) selected))
           (es (mapcar (lambda (ps) ($data ($1 ps))) selected))
           (ls (mapcar (lambda (ps) ($data ($2 ps))) selected)))
       (prn "MEAN/SD[0]:" (round ($mean ss)) "/" (format nil "~8F" ($sd ss)))
       (prn "MEAN/SD[1]:" (round ($mean es)) "/" (format nil "~8F" ($sd es)))
       (prn "MEAN/SD[2]:" (round ($mean ls)) "/" (format nil "~8F" ($sd ls)))))))
