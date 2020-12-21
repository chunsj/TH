(in-package :th)

(defgeneric score/gaussian (data mean sd))
(defgeneric sample/gaussian (mean sd &optional n))

(defun score/normal (data mean sd) (score/gaussian data mean sd))
(defun sample/normal (mean sd &optional (n 1)) (sample/gaussian mean sd n))

(defun of-gaussian-p (sd) (of-plusp sd))

(defun log-gaussian (data mean sd)
  (when (of-gaussian-p sd)
    (if ($tensorp data)
        (let ((n ($count data))
              (z ($sub data mean)))
          (if ($tensorp mean)
              (let ((zsqr ($square z)))
                (list ($neg ($sum ($add ($add ($log sd) (* 0.5 (log (* 2 pi))))
                                        ($div zsqr ($mul 2 ($square sd))))))
                      z))
              (list (- (- (* n (+ (log sd) (* 0.5 (log (* 2 pi))))))
                       (* (/ 1 (* 2 sd sd)) ($sum ($square z))))
                    z)))
        (let ((z (- data mean)))
          (list (- (- (+ (log sd) (* 0.5 (log (* 2 pi)))))
                   (* (/ 1 (* 2 sd sd)) ($square z)))
                z)))))

(defun dlog-gaussian/dmean (z sd)
  (if ($tensorp z)
      (* (/ 1 (* sd sd)) ($sum z))
      (* (/ 1 (* sd sd)) z)))

(defun dlog-gaussian/ddata (z sd)
  (if ($tensorp z)
      (* (/ -1 (* sd sd)) ($sum z))
      (* (/ -1 (* sd sd)) z)))

(defun dlog-gaussian/dsd (z sd)
  (if ($tensorp z)
      (* (- (/ 1 sd)) (- ($count z) (/ ($sum ($square z)) (* sd sd))))
      (* (- (/ 1 sd)) (- 1 (/ ($square z) (* sd sd))))))

(defmethod score/gaussian ((data number) (mean number) (sd number))
  (car (log-gaussian data mean sd)))

(defmethod score/gaussian ((data number) (mean node) (sd number))
  (let ((res (log-gaussian data ($data mean) sd)))
    (when res
      (node (car res)
            :name :gaussian
            :link (link (to mean ($mul (dlog-gaussian/dmean (cadr res) sd) gv)))))))

(defmethod score/gaussian ((data number) (mean number) (sd node))
  (let ((res (log-gaussian data mean ($data sd))))
    (when res
      (node (car res)
            :name :gaussian
            :link (link (to sd ($mul (dlog-gaussian/dsd (cadr res) ($data sd)) gv)))))))

(defmethod score/gaussian ((data number) (mean node) (sd node))
  (let ((res (log-gaussian data ($data mean) ($data sd))))
    (when res
      (node (car res)
            :name :gaussian
            :link (link
                    (to mean ($mul (dlog-gaussian/dmean (cadr res) ($data sd)) gv))
                    (to sd ($mul (dlog-gaussian/dsd (cadr res) ($data sd)) gv)))))))

(defmethod score/gaussian ((data tensor) (mean number) (sd number))
  (car (log-gaussian data mean sd)))

(defmethod score/gaussian ((data tensor) (mean tensor) (sd number))
  (car (log-gaussian data mean sd)))

(defmethod score/gaussian ((data tensor) (mean node) (sd number))
  (let ((res (log-gaussian data ($data mean) sd)))
    (when res
      (node (car res)
            :name :gaussian
            :link (link (to mean ($mul (dlog-gaussian/dmean (cadr res) sd) gv)))))))

(defmethod score/gaussian ((data tensor) (mean number) (sd node))
  (let ((res (log-gaussian data mean ($data sd))))
    (when res
      (node (car res)
            :name :gaussian
            :link (link (to sd ($mul (dlog-gaussian/dsd (cadr res) ($data sd)) gv)))))))

(defmethod score/gaussian ((data tensor) (mean node) (sd node))
  (let ((res (log-gaussian data ($data mean) ($data sd))))
    (when res
      (node (car res)
            :name :gaussian
            :link (link
                    (to mean ($mul (dlog-gaussian/dmean (cadr res) ($data sd)) gv))
                    (to sd ($mul (dlog-gaussian/dsd (cadr res) ($data sd)) gv)))))))

(defmethod score/gaussian ((data node) (mean number) (sd number))
  (let ((res (log-gaussian ($data data) mean sd)))
    (when res
      (node (car res)
            :name :gaussian
            :link (link (to data ($mul (dlog-gaussian/ddata (cadr res) sd) gv)))))))

(defmethod score/gaussian ((data node) (mean node) (sd number))
  (let ((res (log-gaussian ($data data) ($data mean) sd)))
    (when res
      (node (car res)
            :name :gaussian
            :link (link
                    (to data ($mul (dlog-gaussian/ddata (cadr res) sd) gv))
                    (to mean ($mul (dlog-gaussian/dmean (cadr res) sd) gv)))))))

(defmethod score/gaussian ((data node) (mean number) (sd node))
  (let ((res (log-gaussian ($data data) mean ($data sd))))
    (when res
      (node (car res)
            :name :gaussian
            :link (link
                    (to data ($mul (dlog-gaussian/ddata (cadr res) ($data sd)) gv))
                    (to sd ($mul (dlog-gaussian/dsd (cadr res) ($data sd)) gv)))))))

(defmethod score/gaussian ((data node) (mean node) (sd node))
  (let ((res (log-gaussian ($data data) ($data mean) ($data sd))))
    (when res
      (node (car res)
            :name :gaussian
            :link (link
                    (to data ($mul (dlog-gaussian/ddata (cadr res) ($data sd)) gv))
                    (to mean ($mul (dlog-gaussian/dmean (cadr res) ($data sd)) gv))
                    (to sd ($mul (dlog-gaussian/dsd (cadr res) ($data sd)) gv)))))))

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
