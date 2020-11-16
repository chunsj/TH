(in-package :th.pp)

(defgeneric ll/gaussian (data mean sd))
(defgeneric sample/gaussian (mean sd &optional n))

(defun ll/normal (data mean sd) (ll/gaussian data mean sd))
(defun sample/normal (mean sd &optional (n 1)) (sample/gaussian mean sd n))

(defun of-gaussian-p (sd) (of-plusp sd))

(defmethod ll/gaussian ((data number) (mean number) (sd number))
  (when (of-gaussian-p sd)
    (let ((c (- (log (sqrt (* 2 pi)))))
          (var2 (* 2 ($square sd)))
          (lsd (log sd))
          (z (- data mean)))
      (- c lsd (/ ($square z) var2)))))

(defmethod ll/gaussian ((data number) (mean node) (sd number))
  (when (of-gaussian-p sd)
    (let ((c (- (log (sqrt (* 2 pi)))))
          (var2 (* 2 ($square sd)))
          (lsd (log sd))
          (z ($sub data mean)))
      ($sub (- c lsd) ($div ($square z) var2)))))

(defmethod ll/gaussian ((data number) (mean number) (sd node))
  (when (of-gaussian-p ($data sd))
    (let ((c (- (log (sqrt (* 2 pi)))))
          (var2 ($mul 2 ($square sd)))
          (lsd ($log sd))
          (z (- data mean)))
      ($sub ($sub c lsd) ($div ($square z) var2)))))

(defmethod ll/gaussian ((data number) (mean node) (sd node))
  (when (of-gaussian-p ($data sd))
    (let ((c (- (log (sqrt (* 2 pi)))))
          (var2 ($mul 2 ($square sd)))
          (lsd ($log sd))
          (z ($sub data mean)))
      ($sub ($sub c lsd) ($div ($square z) var2)))))

(defmethod ll/gaussian ((data tensor) (mean number) (sd number))
  (when (of-gaussian-p sd)
    (let ((c (- (log (sqrt (* 2 pi)))))
          (var2 (* 2 ($square sd)))
          (lsd (log sd))
          (z ($sub data mean))
          (n ($count data)))
      ($sub (* n (- c lsd)) ($div ($sum ($square z)) var2)))))

(defmethod ll/gaussian ((data tensor) (mean node) (sd number))
  (when (of-gaussian-p sd)
    (let ((c (- (log (sqrt (* 2 pi)))))
          (var2 (* 2 ($square sd)))
          (lsd (log sd))
          (z ($sub data mean))
          (n ($count data )))
      ($sub (* n (- c lsd)) ($div ($sum ($square z)) var2)))))

(defmethod ll/gaussian ((data tensor) (mean number) (sd node))
  (when (of-gaussian-p ($data sd))
    (let ((c (- (log (sqrt (* 2 pi)))))
          (var2 ($mul 2 ($square sd)))
          (lsd ($log sd))
          (z ($sub data mean))
          (n ($count data)))
      ($sub ($mul n ($sub c lsd)) ($div ($sum ($square z)) var2)))))

(defmethod ll/gaussian ((data tensor) (mean node) (sd node))
  (when (of-gaussian-p ($data sd))
    (let ((c (- (log (sqrt (* 2 pi)))))
          (var2 ($mul 2 ($square sd)))
          (lsd ($log sd))
          (z ($sub data mean))
          (n ($count data)))
      ($sub ($mul n ($sub c lsd)) ($div ($sum ($square z)) var2)))))

(defmethod sample/gaussian ((mean number) (sd number) &optional (n 1))
  (cond ((= n 1) (random/normal mean sd))
        ((> n 1) ($normal! (tensor n) mean sd))))

(defmethod sample/gaussian ((mean node) (sd number) &optional (n 1))
  (cond ((= n 1) (random/normal ($data mean) sd))
        ((> n 1) ($normal! (tensor n) ($data mean) sd))))

(defmethod sample/gaussian ((mean number) (sd node) &optional (n 1))
  (cond ((= n 1) (random/normal mean ($data sd)))
        ((> n 1) ($normal! (tensor n) mean ($data sd)))))

(defmethod sample/gaussian ((mean node) (sd node) &optional (n 1))
  (cond ((= n 1) (random/normal ($data mean) ($data sd)))
        ((> n 1) ($normal! (tensor n) ($data mean) ($data sd)))))

(defclass r/gaussian (r/continuous)
  ((mean :initform 0)
   (sd :initform 1)))

(defun r/gaussian (&key (mean 0) (sd 1) observation)
  (let ((m mean)
        (s sd)
        (rv (make-instance 'r/gaussian)))
    (with-slots (mean sd) rv
      (setf mean m
            sd s))
    (r/set-observation! rv observation)
    (r/set-sample! rv)
    rv))

(defun r/normal (&key (mean 0) (sd 1) observation)
  (r/gaussian :mean mean :sd sd :observation observation))

(defmethod r/sample ((rv r/gaussian))
  (with-slots (mean sd) rv
    (sample/gaussian mean sd)))

(defmethod r/score ((rv r/gaussian))
  (with-slots (mean sd) rv
    (ll/gaussian (r/value rv) mean sd)))
