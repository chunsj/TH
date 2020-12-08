(in-package :th)

(defvar *mvnc* (log (* 2 pi)))

(defgeneric score/mvn (data mean tril))
(defgeneric sample/mvn (mean tril &optional n))

(defun mvn/det (tril) ($square ($prd ($diag tril))))
(defun mvn/inverse (tril)
  (let ((inv ($inverse tril)))
    ($mm ($transpose inv) inv)))

(defun of-mvn-p (det) (of-plusp det))

(defun mvn/sub (data mean)
  (let ((nd ($ndim data))
        (d0 ($size data 0))
        (nm ($count mean)))
    (cond ((= nd 1) ($reshape ($sub data mean) 1 d0))
          ((= nd 2) ($sub data ($mm (ones d0 1) ($reshape mean 1 nm)))))))

(defun mvn/f (z m) ($sum ($mul z ($transpose ($mm m ($transpose z)))) 1))

(defun log-mvn (data mean tril)
  (let ((det (mvn/det tril)))
    (when (of-mvn-p det)
      (let ((m (mvn/inverse tril))
            (z (mvn/sub data mean))
            (k ($count mean))
            (tm (- (log det))))
        (let ((f (mvn/f z m)))
          (list ($sum ($mul 0.5 ($sub ($sub tm f) (* k *mvnc*))))
                z
                m))))))

(defun dlog-mvn/dmean (z m)
  ($sum ($mm z m) 0))

(defun dlog-mvn/dtril (z m) ($mul! ($sub! ($mm ($transpose z) z) ($mul m ($size z 0))) 0.5))

(defun dlog-mvn/ddata (z m) ($neg (dlog-mvn/dmean z m)))

;; needs cholesky decomposed lower triangular matrix of covariance matrix.
(defmethod score/mvn ((data tensor) (mean tensor) (tril tensor)) (car (log-mvn data mean tril)))

(defmethod score/mvn ((data tensor) (mean node) (tril tensor))
  (let ((res (log-mvn data ($data mean) tril)))
    (when res
      (node (car res)
            :name :mvn
            :link (link (to mean ($mul! (apply #'dlog-mvn/dmean (cdr res)) gv)))))))

(defmethod score/mvn ((data tensor) (mean tensor) (tril node))
  (let ((res (log-mvn data mean ($data tril))))
    (when res
      (node (car res)
            :name :mvn
            :link (link (to tril ($mul! (apply #'dlog-mvn/dtril (cdr res)) gv)))))))

(defmethod score/mvn ((data tensor) (mean node) (tril node))
  (let ((res (log-mvn data ($data mean) ($data tril))))
    (when res
      (node (car res)
            :name :mvn
            :link (link
                    (to mean ($mul! (apply #'dlog-mvn/dmean (cdr res)) gv))
                    (to tril ($mul! (apply #'dlog-mvn/dtril (cdr res)) gv)))))))

(defmethod score/mvn ((data node) (mean tensor) (tril tensor))
  (let ((res (log-mvn ($data data) mean tril)))
    (when res
      (node (car res)
            :name :mvn
            :link (link (to data ($mul! (apply #'dlog-mvn/ddata (cdr res)) gv)))))))

(defmethod score/mvn ((data node) (mean node) (tril tensor))
  (let ((res (log-mvn ($data data) ($data mean) tril)))
    (when res
      (node (car res)
            :name :mvn
            :link (link
                    (to data ($mul! (apply #'dlog-mvn/ddata (cdr res)) gv))
                    (to mean ($mul! (apply #'dlog-mvn/dmean (cdr res)) gv)))))))

(defmethod score/mvn ((data node) (mean tensor) (tril node))
  (let ((res (log-mvn data mean ($data tril))))
    (when res
      (node (car res)
            :name :mvn
            :link (link
                    (to data ($mul! (apply #'dlog-mvn/ddata (cdr res)) gv))
                    (to tril ($mul! (apply #'dlog-mvn/dtril (cdr res)) gv)))))))

(defmethod score/mvn ((data node) (mean node) (tril node))
  (let ((res (log-mvn data ($data mean) ($data tril))))
    (when res
      (node (car res)
            :name :mvn
            :link (link
                    (to data ($mul! (apply #'dlog-mvn/ddata (cdr res)) gv))
                    (to mean ($mul! (apply #'dlog-mvn/dmean (cdr res)) gv))
                    (to tril ($mul! (apply #'dlog-mvn/dtril (cdr res)) gv)))))))

(defmethod sample/mvn ((mean tensor) (tril tensor) &optional (n 1))
  (let ((rn ($reshape! ($normal! (tensor (* n ($size mean 0))) 0 1) n ($size mean 0))))
    (cond ((= n 1) ($reshape! ($add ($mm rn tril) mean) ($size mean 0)))
          ((> n 1) ($add ($mm rn tril) ($mm (ones n 1) ($reshape mean 1 ($size mean 0))))))))

(defmethod sample/mvn ((mean node) (tril tensor) &optional (n 1))
  (let ((mean ($data mean)))
    (let ((rn ($reshape! ($normal! (tensor (* n ($size mean 0))) 0 1) n ($size mean 0))))
      (cond ((= n 1) ($reshape! ($add ($mm rn tril) mean) ($size mean 0)))
            ((> n 1) ($add ($mm rn tril) ($mm (ones n 1) ($reshape mean 1 ($size mean 0)))))))))

(defmethod sample/mvn ((mean tensor) (tril node) &optional (n 1))
  (let ((tril ($data tril)))
    (let ((rn ($reshape! ($normal! (tensor (* n ($size mean 0))) 0 1) n ($size mean 0))))
      (cond ((= n 1) ($reshape! ($add ($mm rn tril) mean) ($size mean 0)))
            ((> n 1) ($add ($mm rn tril) ($mm (ones n 1) ($reshape mean 1 ($size mean 0)))))))))

(defmethod sample/mvn ((mean node) (tril tensor) &optional (n 1))
  (let ((mean ($data mean))
        (tril ($data tril)))
    (let ((rn ($reshape! ($normal! (tensor (* n ($size mean 0))) 0 1) n ($size mean 0))))
      (cond ((= n 1) ($reshape! ($add ($mm rn tril) mean) ($size mean 0)))
            ((> n 1) ($add ($mm rn tril) ($mm (ones n 1) ($reshape mean 1 ($size mean 0)))))))))
