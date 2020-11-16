(in-package :th.pp)

(defgeneric score/beta (data alpha beta))
(defgeneric sample/beta (alpha beta &optional n))

(defun of-beta-p (data alpha beta)
  (and (of-unit-interval-p data) (of-plusp alpha) (of-plusp beta)))

(defmethod score/beta ((data number) (alpha number) (beta number))
  (when (of-beta-p data alpha beta)
    (+ (* (- alpha 1) (log data))
       (* (- beta 1) (log (- 1 data)))
       (- ($lbetaf alpha beta)))))

(defmethod score/beta ((data number) (alpha node) (beta number))
  (when (of-beta-p data ($data alpha) beta)
    ($sub ($add ($mul ($sub alpha 1) (log data))
                (* (- beta 1) (log (- 1 data))))
          ($lbetaf alpha beta))))

(defmethod score/beta ((data number) (alpha number) (beta node))
  (when (of-beta-p data alpha ($data beta))
    ($sub ($add (* (- alpha 1) (log data))
                ($mul ($sub beta 1) (log (- 1 data))))
          ($lbetaf alpha beta))))

(defmethod score/beta ((data number) (alpha node) (beta node))
  (when (of-beta-p data ($data alpha) ($data beta))
    ($sub ($add ($mul ($sub alpha 1) (log data))
                ($mul ($sub beta 1) (log (- 1 data))))
          ($lbetaf alpha beta))))

(defmethod score/beta ((data tensor) (alpha number) (beta number))
  (when (of-beta-p data alpha beta)
    ($sum ($sub ($add ($mul (- alpha 1) ($log data))
                      ($mul (- beta 1) ($log ($sub 1 data))))
                ($lbetaf alpha beta)))))

(defmethod score/beta ((data tensor) (alpha node) (beta number))
  (when (of-beta-p data ($data alpha) beta)
    ($sum ($sub ($add ($mul ($sub alpha 1) ($log data))
                      ($mul (- beta 1) ($log ($sub 1 data))))
                ($lbetaf alpha beta)))))

(defmethod score/beta ((data tensor) (alpha number) (beta node))
  (when (of-beta-p data alpha ($data beta))
    ($sum ($sub ($add ($mul (- alpha 1) ($log data))
                      ($mul ($sub beta 1) ($log ($sub 1 data))))
                ($lbetaf alpha beta)))))

(defmethod score/beta ((data tensor) (alpha node) (beta node))
  (when (of-beta-p data ($data alpha) ($data beta))
    ($sum ($sub ($add ($mul ($sub alpha 1) ($log data))
                      ($mul ($sub beta 1) ($log ($sub 1 data))))
                ($lbetaf alpha beta)))))

(defmethod sample/beta ((alpha number) (beta number) &optional (n 1))
  (cond ((= n 1) (random/beta alpha beta))
        ((> n 1) ($beta! (tensor n) alpha beta))))

(defmethod sample/beta ((alpha node) (beta number) &optional (n 1))
  (cond ((= n 1) (random/beta ($data alpha) beta))
        ((> n 1) ($beta! (tensor n) ($data alpha) beta))))

(defmethod sample/beta ((alpha number) (beta node) &optional (n 1))
  (cond ((= n 1) (random/beta alpha ($data beta)))
        ((> n 1) ($beta! (tensor n) alpha ($data beta)))))

(defmethod sample/beta ((alpha node) (beta node) &optional (n 1))
  (cond ((= n 1) (random/beta ($data alpha) ($data beta)))
        ((> n 1) ($beta! (tensor n) ($data alpha) ($data beta)))))

(defclass r/beta (r/continuous)
  ((alpha :initform 1)
   (beta :initform 1)))

(defun r/beta (&key (alpha 1) (beta 1) observation)
  (let ((a alpha)
        (b beta)
        (rv (make-instance 'r/beta)))
    (with-slots (alpha beta) rv
      (setf alpha a
            beta b))
    (r/set-observation! rv observation)
    (r/set-sample! rv)
    rv))

(defmethod r/sample ((rv r/beta))
  (with-slots (alpha beta) rv
    (sample/beta alpha beta)))

(defmethod r/score ((rv r/beta))
  (with-slots (alpha beta) rv
    (score/beta (r/value rv) alpha beta)))
