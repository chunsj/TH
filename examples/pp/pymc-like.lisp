(defpackage :pymc-like
  (:use #:common-lisp
        #:mu
        #:th
        #:th.pp))

(in-package :pymc-like)

(defvar *disasters* '(4 5 4 0 1 4 3 4 0 6 3 3 4 0 2 6
                      3 3 5 4 5 3 1 4 4 1 5 5 3 4 2 5
                      2 2 3 4 2 1 3 2 2 1 1 1 1 3 0 0
                      1 0 1 1 0 0 3 1 0 3 2 2 0 1 1 1
                      0 1 0 1 0 0 0 2 1 0 0 0 1 1 0 2
                      3 3 1 1 2 1 1 1 1 2 4 2 0 0 1 4
                      0 0 0 1 0 0 0 0 0 1 0 0 1 0 1))
(defvar *rate* (/ 1D0 ($mean *disasters*)))

(defgeneric $logp (rv))
(defgeneric $loghr (rv))
(defgeneric $loghr! (rv))
(defgeneric $propose (rv))

(defmethod $logp ((rv T)) 0)
(defmethod $loghr ((rv T)) 0)

(defclass rv/var ()
  ((value :initform nil)
   (observedp :initform nil)
   (ps :initform 0.1D0)
   (loghr :initform 0)))

(defmethod $data ((rv rv/var))
  (with-slots (value) rv
    value))

(defmethod $clone ((rv rv/var))
  (let ((n (make-instance (class-of rv))))
    (with-slots (value observedp ps) rv
      (let ((v value)
            (o observedp)
            (p ps))
        (with-slots (value observedp ps) n
          (setf value v
                observedp o
                ps p))))
    n))

(defmethod $propose ((rv rv/var))
  (let ((n ($clone rv)))
    (with-slots (value ps loghr) n
      (let ((new-value (sample/gaussian value ps)))
        (setf value new-value
              loghr 0)))
    n))

(defmethod $loghr ((rv rv/var))
  (with-slots (loghr) rv
    loghr))

(defmethod $loghr! ((rv rv/var))
  (with-slots (loghr) rv
    (setf loghr 0))
  rv)

(defun set-observation (rv observation)
  (when observation
    (with-slots (value observedp) rv
      (setf value observation
            observedp T))))

(defmethod print-object ((rv rv/var) stream)
  (with-slots (observedp value) rv
    (format stream "~A~A" value (if (not observedp) "?" ""))))

(defclass rv/discrete-uniform (rv/var)
  ((lower :initform 0)
   (upper :initform 9)))

(defun rv/discrete-uniform (&key (lower 0) (upper 9) (ps 2) observation)
  (let ((l lower)
        (u upper)
        (s ps)
        (n (make-instance 'rv/discrete-uniform)))
    (with-slots (lower upper value ps) n
      (set-observation n observation)
      (setf lower l
            upper u
            ps s)
      (unless value
        (setf value (sample/discrete-uniform lower upper))))
    n))

(defmethod $clone ((rv rv/discrete-uniform))
  (let ((n (call-next-method rv)))
    (with-slots (lower upper) rv
      (let ((l ($clone lower))
            (u ($clone upper)))
        (with-slots (lower upper) n
          (setf lower l
                upper u))))
    n))

(defmethod $propose ((rv rv/discrete-uniform))
  (let ((n ($clone rv)))
    (with-slots (value ps loghr) n
      (let ((new-value (round (sample/gaussian value ps))))
        (setf value new-value
              loghr 0)))
    n))

(defmethod $logp ((rv rv/discrete-uniform))
  (with-slots (value lower upper) rv
    (let ((ll (score/discrete-uniform value ($data lower) ($data upper)))
          (llower ($logp lower))
          (lupper ($logp upper)))
      (when (and ll llower lupper)
        (+ ll llower lupper)))))

(defclass rv/exponential (rv/var)
  ((rate :initform 1D0)))

(defun rv/exponential (&key (rate 1D0) observation)
  (let ((r rate)
        (n (make-instance 'rv/exponential)))
    (with-slots (rate value) n
      (set-observation n observation)
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

(defclass rv/poisson (rv/var)
  ((rate :initform 1D0)))

(defun rv/poisson (&key (rate 1D0) observation)
  (let ((r rate)
        (n (make-instance 'rv/poisson)))
    (with-slots (rate value) n
      (set-observation n observation)
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

(defun likelihood (switch-point early-mean late-mean)
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

(defun log-hastings-ratio (parameters)
  (let ((lhs (mapcar #'$loghr parameters)))
    (reduce (lambda (s l) (when (and s l) (+ s l))) lhs)))

(defun mh (nsamples parameters likelihoodfn &key (max-iterations 50000))
  (let ((old-likelihood (apply likelihoodfn parameters))
        (old-parameters (mapcar #'$clone parameters))
        (accepted nil)
        (naccepted 0)
        (max-likelihood most-negative-single-float)
        (done nil))
    (prn (format nil "*    START MLE: ~10,4F" old-likelihood))
    (prn (format nil "            ~{~A ~}" old-parameters))
    (when old-likelihood
      (loop :while (not done)
            :repeat (max nsamples max-iterations)
            :for iter :from 1
            :for new-parameters = (mapcar #'$propose old-parameters)
            :for log-hastings-ratio = (log-hastings-ratio new-parameters)
            :for new-likelihood = (apply likelihoodfn new-parameters)
            :do (when (and log-hastings-ratio new-likelihood)
                  (let ((lmh (+ (- new-likelihood old-likelihood) log-hastings-ratio))
                        (lu (log (random 1D0))))
                    (when (> lmh lu)
                      (push (mapcar #'$clone new-parameters) accepted)
                      (incf naccepted)
                      (setf old-parameters (mapcar #'$loghr! new-parameters)
                            old-likelihood new-likelihood)
                      (when (> new-likelihood max-likelihood)
                        (setf max-likelihood new-likelihood)
                        (prn (format nil "* ~8,D MLE: ~10,4F" naccepted max-likelihood))
                        (prn (format nil "            ~{~A ~}" new-parameters)))))
                  (when (>= naccepted nsamples)
                    (setf done T)))))
    accepted))

;; MLE: 41, 3, 1
(time
 (let ((switch-point (rv/discrete-uniform :lower 0 :upper (1- ($count *disasters*))))
       (early-mean (rv/exponential :rate *rate*))
       (late-mean (rv/exponential :rate *rate*)))
   (let* ((accepted (mh 100000 (list switch-point early-mean late-mean) #'likelihood))
          (na ($count accepted))
          (ns (round (* 0.2 na)))
          (selected (subseq accepted 0 ns)))
     (prn "SELECTED:" ns "/" na)
     (let ((ss (mapcar (lambda (ps) ($data ($0 ps))) selected))
           (es (mapcar (lambda (ps) ($data ($1 ps))) selected))
           (ls (mapcar (lambda (ps) ($data ($2 ps))) selected)))
       (prn "P0:" (round ($mean ss)))
       (prn "P1:" (round ($mean es)))
       (prn "P2:" (round ($mean ls)))))))

;; FOR SMS example
;; https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/masterv/Chapter1_Introduction/Ch1_Introduction_PyMC2.ipynb
(setf *disasters* (->> (slurp "./data/sms.txt")
                       (mapcar #'parse-float)
                       (mapcar #'round)))
(setf *rate* (/ 1D0 ($mean *disasters*)))

;; MLE: 45, 18, 23
(time
 (let ((switch-point (rv/discrete-uniform :lower 0 :upper (1- ($count *disasters*)) :ps 1))
       (early-mean (rv/exponential :rate *rate*))
       (late-mean (rv/exponential :rate *rate*)))
   (let* ((accepted (mh 100000 (list switch-point early-mean late-mean) #'likelihood))
          (na ($count accepted))
          (ns (round (* 0.2 na)))
          (selected (subseq accepted 0 ns)))
     (prn "SELECTED:" ns "/" na)
     (let ((ss (mapcar (lambda (ps) ($data ($0 ps))) selected))
           (es (mapcar (lambda (ps) ($data ($1 ps))) selected))
           (ls (mapcar (lambda (ps) ($data ($2 ps))) selected)))
       (prn "P0:" (round ($mean ss)))
       (prn "P1:" (round ($mean es)))
       (prn "P2:" (round ($mean ls)))))))