(in-package :th.pp)

(defgeneric score/poisson (data rate))
(defgeneric sample/poisson (rate &optional n))

(defun of-poisson-p (data rate) (and (of-plusp rate) (of-ge data 0)))

(defmethod score/poisson ((data number) (rate number))
  (when (of-poisson-p data rate)
    (let ((lfac ($lgammaf (+ 1 data))))
      (- (* data (log rate)) (+ rate lfac)))))

(defmethod score/poisson ((data number) (rate node))
  (when (of-poisson-p data ($data rate))
    (let ((lfac ($lgammaf (+ 1 data))))
      ($sub ($mul data ($log rate)) ($add rate lfac)))))

(defmethod score/poisson ((data tensor) (rate number))
  (when (of-poisson-p data rate)
    (let ((lfac ($lgammaf ($add 1 data))))
      ($sum ($sub ($mul data (log rate)) ($add rate lfac))))))

(defmethod score/poisson ((data tensor) (rate node))
  (when (of-poisson-p data ($data rate))
    (let ((lfac ($lgammaf ($add 1 data))))
      ($sum ($sub ($mul data ($log rate)) ($add rate lfac))))))

(defmethod score/poisson ((data node) (rate number))
  (when (of-poisson-p ($data data) rate)
    (let ((lfac ($lgammaf ($add 1 data))))
      ($sum ($sub ($mul data (log rate)) ($add rate lfac))))))

(defmethod score/poisson ((data node) (rate node))
  (when (of-poisson-p ($data data) ($data rate))
    (let ((lfac ($lgammaf ($add 1 data))))
      ($sum ($sub ($mul data ($log rate)) ($add rate lfac))))))

(defmethod sample/poisson ((rate number) &optional (n 1))
  (cond ((= n 1) (random/poisson rate))
        ((> n 1) ($poisson! (tensor n) rate))))

(defmethod sample/poisson ((rate node) &optional (n 1))
  (cond ((= n 1) (random/poisson ($data rate)))
        ((> n 1) ($poisson! (tensor n) ($data rate)))))
