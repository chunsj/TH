(in-package :th.pp)

(defclass r/trace (r/variable)
  ((collection :initform nil :reader trace/collection)
   (burn-ins :initform 0)
   (thin :initform 0)
   (vals :initform nil :accessor trace/values)
   (mean :initform nil)
   (variance :initform nil)))

(defun r/trace (v &key (n 1) (burn-in 0) (thin 0))
  ;; XXX use the shape of v.
  (let ((tr (make-instance 'r/trace))
        (nb burn-in)
        (nt thin))
    (with-slots (value collection burn-ins thin vals) tr
      (setf collection (zeros (+ n nb))
            value v
            burn-ins nb
            thin nt
            vals (zeros (floor (/ n nt)))))
    tr))

(defun r/traces (vs &key (n 1) (burn-in 0) (thin 0))
  (loop :for v :in vs
        :collect (r/trace v :n n :burn-in burn-in :thin thin)))

(defmethod $count ((trace r/trace))
  (with-slots (collection) trace
    ($count collection)))

(defmethod $ ((trace r/trace) index &rest others-and-default)
  (declare (ignore others-and-default))
  (with-slots (collection) trace
    ($ collection index)))

(defmethod (setf $) (value (trace r/trace) index &rest others)
  (declare (ignore others))
  (with-slots (collection burn-ins vals thin) trace
    (setf ($ collection index) value)
    (when (>= index burn-ins)
      (let ((i (- index burn-ins)))
        (when (zerop (rem i thin))
          (setf ($ vals i) value))))
    value))

(defun trace/mean (trace)
  (with-slots (mean) trace
    (unless mean
      (setf mean ($mean (trace/values trace))))
    mean))

(defun trace/variance (trace)
  (with-slots (variance) trace
    (unless variance
      (setf variance ($var (trace/values trace))))
    variance))

(defun trace/sd (trace)
  (with-slots (variance) trace
    (unless variance
      (setf variance ($var (trace/values trace))))
    ($sqrt variance)))

(defun trace/error (trace)
  (let ((n ($count (trace/values trace)))
        (sd (trace/sd trace)))
    (when (>= n 1) (/ sd (sqrt n)))))

(defun trace/acr (trace &key (maxlag 100))
  (let ((vals (trace/values trace)))
    (loop :for k :from 0 :to (min maxlag (1- ($count vals)))
          :collect ($acr vals k))))

(defun trace/quantiles (trace)
  (let* ((trcvs (trace/values trace))
         (n ($count trcvs))
         (qlist '(2.5 25 50 75 97.5)))
    (when (> n 10)
      (let ((vs (car ($sort trcvs))))
        (loop :for q :in qlist
              :for ridx = (round (* n (/ q 100)))
              :collect (let ((i ridx))
                         (when (< i 0) (setf i 0))
                         (when (> i (1- n)) n)
                         (cons q ($ vs i))))))))

(defun trace/hpd (trace &optional (alpha 0.05))
  (labels ((min-interval (vs alpha)
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
               (cons mn mx))))
    (let ((vs (car ($sort (trace/values trace)))))
      (when (> ($count vs) 10)
        (min-interval vs alpha)))))

(defun trace/geweke (trace &key (first 0.1) (last 0.5) (intervals 20))
  (when (< (+ first last))
    (labels ((interval-zscores (vs a b &optional (intervals 20))
               (let* ((end (1- ($count vs)))
                      (hend (/ end 2))
                      (sindices (loop :for i :from 0 :below (round hend)
                                      :by (round (/ hend intervals))
                                      :collect i)))
                 (loop :for start :in sindices
                       :for asize = (round (* a (- end start)))
                       :for slice-a = ($subview vs start asize)
                       :for bstart = (round (- end (* b (- end start))))
                       :for slice-b = ($subview vs bstart (- ($count vs) bstart))
                       :for zn = (- ($mean slice-a) ($mean slice-b))
                       :for zd = (+ ($square ($sd slice-a)) ($square ($sd slice-b)))
                       :collect (cons start (/ zn zd))))))
      (let ((vs (trace/values trace)))
        (when (> ($count vs) intervals)
          (interval-zscores vs first last intervals))))))

(defun trace/summary (trace)
  (let ((quantiles (trace/quantiles trace))
        (n ($count (trace/values trace)))
        (sd (trace/sd trace))
        (m (trace/mean trace))
        (err (trace/error trace))
        (hpd (trace/hpd trace 0.05))
        (acr ($mean (subseq (trace/acr trace) 1)))
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

(defun traces/sample (traces &key (n 1) transform)
  (let ((trcs (loop :for trace :in traces :collect (trace/values trace))))
    (loop :repeat n
          :for parameters = (loop :for trc :in trcs
                                  :for ntrc = ($count trc)
                                  :for idx = (random ntrc)
                                  :collect ($ trc idx))
          :collect (apply (or transform #'identity) parameters))))
