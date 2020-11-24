(in-package :th.pp)

(defclass mcmc/trace ()
  ((collection :initform nil)
   (mval :initform nil :reader trace/map)
   (burn-ins :initform 0)
   (thin :initform 0)))

(defmethod $count ((trace mcmc/trace))
  (with-slots (collection) trace
    ($count collection)))

(defun mcmc-trace (&key (burn-in 0) (thin 0))
  (let ((ntr (make-instance 'mcmc/trace))
        (nb burn-in)
        (nt thin))
    (with-slots (burn-ins thin) ntr
      (setf burn-ins nb
            thin nt))
    ntr))

(defun mcmc/traces (n &key (burn-in 0) (thin 0))
  (loop :repeat n :collect (mcmc-trace :burn-in burn-in :thin thin)))

(defun trace/push! (val trace)
  (with-slots (collection) trace
    (push ($clone val) collection))
  val)

(defun trace/map! (trace v)
  (with-slots (mval) trace
    (setf mval ($clone v)))
  v)

(defun trace/values (trace)
  (with-slots (collection burn-ins thin) trace
    (let ((rtraces (nthcdr burn-ins (reverse collection))))
      (loop :for lst :on rtraces :by (lambda (l) (nthcdr thin l))
            :collect (car lst)))))

(defun trace/mean (trace) ($mean (trace/values trace)))

(defun trace/sd (trace) ($sd (trace/values trace)))

(defun trace/count (trace) ($count (trace/values trace)))

(defun trace/error (trace)
  (let ((trc (trace/values trace)))
    (let ((n ($count trc))
          (sd ($sd trc)))
      (when (>= n 1) (/ sd (sqrt n))))))

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

(defun trace/autocorrelation (trace &key (maxlag 100))
  (let ((trcvs (trace/values trace)))
    (loop :for k :from 0 :below (1+ maxlag)
          :collect (autocorr trcvs k))))

(defun trace/quantiles (trace)
  (let* ((trcvs (trace/values trace))
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

(defun trace/hpd (trace alpha)
  (let ((vs (sort (copy-list (trace/values trace)) #'<)))
    (when (> ($count vs) 10)
      (min-interval vs alpha))))

(defun trace/geweke (trace &key (first 0.1) (last 0.5) (intervals 20))
  (when (< (+ first last))
    (let ((vs (trace/values trace)))
      (when (> ($count vs) intervals)
        (interval-zscores vs first last intervals)))))

(defun trace/summary (trace)
  (let ((quantiles (trace/quantiles trace))
        (n (trace/count trace))
        (sd (trace/sd trace))
        (m (trace/mean trace))
        (err (trace/error trace))
        (hpd (trace/hpd trace 0.05))
        (acr ($mean (subseq (trace/autocorrelation trace) 1)))
        (geweke (let ((gvs (mapcar #'cdr (trace/geweke trace))))
                  (cons (apply #'min gvs) (apply #'max gvs)))))
    (list :count n
          :mean m
          :sd sd
          :error err
          :hpd-95 hpd
          :quantiles quantiles
          :acmean acr
          :gwkrng geweke)))

(defun trace/sample (traces &key (n 1) transform)
  (let ((trcs (loop :for trace :in traces :collect (trace/values trace)))
        (ntrcs (loop :for trace :in traces :collect (trace/count trace))))
    (loop :repeat n
          :for parameters = (loop :for trc :in trcs
                                  :for ntrc :in ntrcs
                                  :for idx = (random ntrc)
                                  :collect ($ trc idx))
          :collect (apply (or transform #'identity) parameters))))
