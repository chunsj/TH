(defpackage :rv-play
  (:use #:common-lisp
        #:mu
        #:th
        #:th.distributions))

(in-package :th.distributions)

(defclass rv/poissons (rv/variable)
  ((rates :initform '(1D0))))

(defun rv/poissons (&key (rates '(1D0)) observation)
  (let ((r rates)
        (n (make-instance 'rv/poissons)))
    (setf ($observation n) observation)
    (with-slots (rates value) n
      (setf rates r)
      (unless value
        (setf value (mapcar (lambda (r) ($sample/poisson 1 ($data r))) rates))))
    n))

(defmethod $clone ((rv rv/poissons))
  (let ((n (call-next-method rv)))
    (with-slots (rates) rv
      (let ((rs (mapcar (lambda (r) ($clone r)) rates)))
        (with-slots (rates) n
          (setf rates rs))))
    n))

(defmethod $logp ((rv rv/poissons))
  (with-slots (value rates) rv
    (let ((lrates (reduce (lambda (s v) (when (and s v) (+ s v)))
                          (mapcar #'$logp (remove-duplicates rates))))
          (lls (reduce (lambda (s v) (when (and s v) (+ s v)))
                       (mapcar (lambda (v r) ($ll/poisson v ($data r))) value rates))))
      (when (and lls lrates)
        (+ lls lrates)))))

(in-package :rv-play)

(defvar *disasters* '(4 5 4 0 1 4 3 4 0 6 3 3 4 0 2 6
                      3 3 5 4 5 3 1 4 4 1 5 5 3 4 2 5
                      2 2 3 4 2 1 3 2 2 1 1 1 1 3 0 0
                      1 0 1 1 0 0 3 1 0 3 2 2 0 1 1 1
                      0 1 0 1 0 0 0 2 1 0 0 0 1 1 0 2
                      3 3 1 1 2 1 1 1 1 2 4 2 0 0 1 4
                      0 0 0 1 0 0 0 0 0 1 0 0 1 0 1))
(defvar *rate* (/ 1D0 ($mean *disasters*)))

(let ((switch-point (rv/discrete-uniform :lower 0 :upper (1- ($count *disasters*))))
      (early-mean (rv/exponential :rate *rate*))
      (late-mean (rv/exponential :rate *rate*)))
  (setf ($data switch-point) 41
        ($data early-mean) 3
        ($data late-mean) 1)
  (list (disaster-likelihood switch-point early-mean late-mean)
        (disaster-likelihood2 switch-point early-mean late-mean)))

(let* ((lsw ($ll/uniform 41 0 (- ($count *disasters*) 1)))
       (lem ($ll/exponential 3 *rate*))
       (llm ($ll/exponential 1 *rate*))
       (lDe (reduce #'+ (mapcar (lambda (dv) ($ll/poisson dv 3)) (subseq *disasters* 0 41))))
       (lDl (reduce #'+ (mapcar (lambda (dv) ($ll/poisson dv 1)) (subseq *disasters* 41)))))
  (+ lsw lem llm lDe lDl))

(defun disaster-likelihood2 (switch-point early-mean late-mean)
  (let ((ls ($logp switch-point)))
    (when ls
      (let* ((rates (loop :for i :from 0 :below ($count *disasters*)
                          :collect (if (< i ($data switch-point))
                                       early-mean
                                       late-mean)))
             (D (th.distributions::rv/poissons :rates rates :observation *disasters*))
             (lD ($logp D)))
        (when (and ls lD) (+ ls lD))))))

(defun disaster-likelihood (switch-point early-mean late-mean)
  (let ((ls ($logp switch-point)))
    (when ls
      (let ((disasters-early (subseq *disasters* 0 ($data switch-point)))
            (disasters-late (subseq *disasters* ($data switch-point))))
        (let ((d1 (rv/poisson :rate early-mean :observation disasters-early))
              (d2 (rv/poisson :rate late-mean :observation disasters-late)))
          (let ((ld1 ($logp d1))
                (ld2 ($logp d2)))
            (when (and ls ld1 ld2)
              (+ ls ld1 ld2))))))))

;; MLE: 41, 3, 1
(let ((switch-point (rv/discrete-uniform :lower 0 :upper (1- ($count *disasters*))))
      (early-mean (rv/exponential :rate *rate*))
      (late-mean (rv/exponential :rate *rate*)))
  (setf ($data switch-point) 41
        ($data early-mean) 3
        ($data late-mean) 1)
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
      (prn "MEAN/SD[2]:" (round ($mean ls)) "/" (format nil "~8F" ($sd ls))))))

;; FOR SMS example
;; https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/masterv/Chapter1_Introduction/Ch1_Introduction_PyMC2.ipynb
(defvar *sms* (->> (slurp "./data/sms.txt")
                   (mapcar #'parse-float)
                   (mapcar #'round)))
(defvar *srate* (/ 1D0 ($mean *sms*)))

(defun sms-likelihood (switch-point early-mean late-mean)
  (let ((ls ($logp switch-point)))
    (when ls
      (let ((disasters-early (subseq *sms* 0 ($data switch-point)))
            (disasters-late (subseq *sms* ($data switch-point))))
        (let ((d1 (rv/poisson :rate early-mean :observation disasters-early))
              (d2 (rv/poisson :rate late-mean :observation disasters-late)))
          (let ((ld1 ($logp d1))
                (ld2 ($logp d2)))
            (when (and ls ld1 ld2)
              (+ ls ld1 ld2))))))))

;; MLE: 45, 18, 23
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
      (prn "MEAN/SD[2]:" (round ($mean ls)) "/" (format nil "~8F" ($sd ls))))))

(defun histogram (xs &key (nbins 10))
  (let ((xs (sort (copy-list xs) #'<))
        (steps nbins))
    (let* ((Xmin (apply #'min xs))
           (Xmax (apply #'max xs))
           (Xstep (/ (- Xmax Xmin) steps)))
      (loop :for i :from 0 :below steps
            :for minx = (+ Xmin (* i Xstep))
            :for maxx = (+ Xmin (* (1+ i) Xstep))
            :for nx = ($count (filter (lambda (x) (and (>= x minx)
                                                  (< x maxx)))
                                      xs))
            :collect (cons minx nx)))))

(-> (loop :for yr :from 1851
          :for o :in *disasters*
          :collect (cons yr o))
    (mplot:plot-boxes))

;; XXX
;; 0. check the results of simulation
;; 1. scale parameter
;; 2. add previous sample
;; 3. compare accepted and rejected
