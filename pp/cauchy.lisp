(in-package :th.pp)

(defgeneric ll/cauchy (data location scale))
(defgeneric sample/cauchy (location scale &optional n))

(defun of-cauchy-p (scale) (of-plusp scale))

(defmethod ll/cauchy ((data number) (location number) (scale number))
  (when (of-cauchy-p scale)
    (let ((lpi (log pi))
          (ls (log scale))
          (z (/ (- data location) scale)))
      (- lpi (+ ls (log (+ 1 ($square z))))))))

(defmethod ll/cauchy ((data number) (location node) (scale number))
  (when (of-cauchy-p scale)
    (let ((lpi (log pi))
          (ls (log scale))
          (z ($div ($sub data location) scale)))
      ($sub lpi ($add ls ($log1p ($square z)))))))

(defmethod ll/cauchy ((data number) (location number) (scale node))
  (when (of-cauchy-p ($data scale))
    (let ((lpi (log pi))
          (ls ($log scale))
          (z ($div (- data location) scale)))
      ($sub lpi ($add ls ($log1p ($square z)))))))

(defmethod ll/cauchy ((data number) (location node) (scale node))
  (when (of-cauchy-p ($data scale))
    (let ((lpi (log pi))
          (ls ($log scale))
          (z ($div ($sub data location) scale)))
      ($sub lpi ($add ls ($log1p ($square z)))))))

(defmethod ll/cauchy ((data tensor) (location number) (scale number))
  (when (of-cauchy-p scale)
    (let ((lpi (log pi))
          (ls (log scale))
          (z ($div ($sub data location) scale)))
      ($sum ($sub lpi ($add ls ($log1p ($square z))))))))

(defmethod ll/cauchy ((data tensor) (location node) (scale number))
  (when (of-cauchy-p scale)
    (let ((lpi (log pi))
          (ls (log scale))
          (z ($div ($sub data location) scale)))
      ($sum ($sub lpi ($add ls ($log1p ($square z))))))))

(defmethod ll/cauchy ((data tensor) (location number) (scale node))
  (when (of-cauchy-p ($data scale))
    (let ((lpi (log pi))
          (ls (log scale))
          (z ($div ($sub data location) scale)))
      ($sum ($sub lpi ($add ls ($log1p ($square z))))))))

(defmethod ll/cauchy ((data tensor) (location node) (scale node))
  (when (of-cauchy-p ($data scale))
    (let ((lpi (log pi))
          (ls (log scale))
          (z ($div ($sub data location) scale)))
      ($sum ($sub lpi ($add ls ($log1p ($square z))))))))

(defmethod sample/cauchy ((location number) (scale number) &optional (n 1))
  (cond ((= n 1) (+ location
                    (* scale (tan (* pi (- (random 1D0) 0.5D0))))))
        ((> n 1) (tensor (loop :repeat n
                               :for u = (tan (* pi (- (random 1D0) 0.5D0)))
                               :collect (+ location (* scale u)))))))

(defmethod sample/cauchy ((location node) (scale number) &optional (n 1))
  (cond ((= n 1) (+ ($data location)
                    (* scale (tan (* pi (- (random 1D0) 0.5D0))))))
        ((> n 1) (tensor (loop :repeat n
                               :for u = (tan (* pi (- (random 1D0) 0.5D0)))
                               :collect (+ ($data location) (* scale u)))))))

(defmethod sample/cauchy ((location number) (scale node) &optional (n 1))
  (cond ((= n 1) (+ location
                    (* ($data scale) (tan (* pi (- (random 1D0) 0.5D0))))))
        ((> n 1) (tensor (loop :repeat n
                               :for u = (tan (* pi (- (random 1D0) 0.5D0)))
                               :collect (+ location (* ($data scale) u)))))))

(defmethod sample/cauchy ((location node) (scale node) &optional (n 1))
  (cond ((= n 1) (+ ($data location)
                    (* ($data scale) (tan (* pi (- (random 1D0) 0.5D0))))))
        ((> n 1) (tensor (loop :repeat n
                               :for u = (tan (* pi (- (random 1D0) 0.5D0)))
                               :collect (+ ($data location) (* ($data scale) u)))))))

(defclass r/cauchy (r/continuous)
  ((location :initform 0)
   (scale :initform 1)))

(defun r/cauchy (&key (location 0) (scale 1) observation)
  (let ((l location)
        (s scale)
        (rv (make-instance 'r/cauchy)))
    (with-slots (location scale) rv
      (setf location l
            scale s))
    (r/set-observation! rv observation)
    (r/set-sample! rv)
    rv))

(defmethod r/sample ((rv r/cauchy))
  (with-slots (location scale) rv
    (sample/cauchy location scale)))

(defmethod r/score ((rv r/cauchy))
  (with-slots (location scale) rv
    (ll/cauchy (r/value rv) location scale)))
