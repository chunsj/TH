(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defun broadcast-number (c m)
  (node ($mul! ($one m) ($data c))
        :name :broadcast
        :bps (bps c (lambda (dv gv) (declare (ignore dv)) ($dot ($one m) gv)))))

(defun broadcast-1x1 (c m)
  (node ($mul! ($one m) ($data c))
        :name :broadcast
        :bps (bps c (lambda (dv gv) (declare (ignore dv)) ($dot ($one m) gv)))))

(defmethod $broadcast ((c node) (m node))
  (cond ((numberp ($data c)) (broadcast-number c ($data m)))
        ((eq 1 ($count ($data c))) (broadcast-1x1 c ($data m)))
        (t "cannot broadcast automatically other than number.")))

(defmethod $broadcast ((c node) (m tensor))
  (cond ((numberp ($data c)) (broadcast-number c m))
        ((eq 1 ($count ($data c))) (broadcast-1x1 c m))
        (t "cannot broadcast automatically other than number.")))

(defmethod $broadcast ((c number) (m node))
  (node ($mul! ($one ($data m)) c) :name :broadcast))

(defmethod $add ((a node) (b node))
  (node ($add ($data a) ($data b))
        :name :add
        :bps (bps a (lambda (dv gv) (declare (ignore dv)) gv)
                  b (lambda (dv gv) (declare (ignore dv)) gv))))

(defmethod $add ((a node) (b tensor))
  (node ($add ($data a) b)
        :name :add
        :bps (bps a (lambda (dv gv) (declare (ignore dv)) gv))))

(defmethod $add ((a node) (b number))
  (node ($add ($data a) b)
        :name :add
        :bps (bps a (lambda (dv gv) (declare (ignore dv)) gv))))

(defmethod $add ((a tensor) (b node)) ($add b a))
(defmethod $add ((a number) (b node)) ($add b a))

(defmethod $sub ((a node) (b node))
  (node ($sub ($data a) ($data b))
        :name :sub
        :bps (bps a (lambda (dv gv) (declare (ignore dv)) gv)
                  b (lambda (dv gv) (declare (ignore dv)) ($neg gv)))))

(defmethod $sub ((a node) (b tensor))
  (node ($sub ($data a) b)
        :name :sub
        :bps (bps a (lambda (dv gv) (declare (ignore dv)) gv))))

(defmethod $sub ((a node) (b number))
  (node ($sub ($data a) b)
        :name :sub
        :bps (bps a (lambda (dv gv) (declare (ignore dv)) gv))))

(defmethod $sub ((a tensor) (b node))
  (node ($sub a ($data b))
        :name :sub
        :bps (bps b (lambda (dv gv) (declare (ignore dv)) ($neg gv)))))

(defmethod $sub ((a number) (b node))
  (node ($sub a ($data b))
        :name :sub
        :bps (bps b (lambda (dv gv) (declare (ignore dv)) ($neg gv)))))

(defmethod $neg ((a node))
  (node ($neg ($data a)) :name :neg :bps (bps a (lambda (dv gv) (declare (ignore dv)) ($neg gv)))))

(defmethod $dot ((a node) (b node))
  (node ($dot ($data a) ($data b))
        :name :dot
        :bps (bps a (lambda (dv gv) (declare (ignore dv)) ($mul ($data b) gv))
                  b (lambda (dv gv) (declare (ignore dv)) ($mul ($data a) gv)))))

(defmethod $dot ((a node) (b tensor))
  (node ($dot ($data a) b)
        :name :dot
        :bps (bps a (lambda (dv gv) (declare (ignore dv)) ($mul b gv)))))

(defmethod $dot ((a tensor) (b node)) ($dot b a))

(defmethod $mv ((m node) (v node))
  (node ($mv ($data m) ($data v))
        :name :mv
        :bps (bps m (lambda (dv gv) (declare (ignore dv)) ($vv gv ($data v)))
                  v (lambda (dv gv) (declare (ignore dv)) ($@ ($transpose ($data m)) gv)))))

(defmethod $mv ((m node) (v tensor))
  (node ($mv ($data m) v)
        :name :mv
        :bps (bps m (lambda (dv gv) (declare (ignore dv)) ($vv gv v)))))

(defmethod $mv ((m tensor) (v node))
  (node ($mv m ($data v))
        :name :mv
        :bps (bps v (lambda (dv gv) (declare (ignore dv)) ($@ ($transpose m) gv)))))

(defmethod $mm ((a node) (b node))
  (node ($mm ($data a) ($data b))
        :name :mm
        :bps (bps a (lambda (dv gv) (declare (ignore dv)) ($mm gv ($transpose ($data b))))
                  b (lambda (dv gv) (declare (ignore dv)) ($mm ($transpose ($data a)) gv)))))

(defmethod $mm ((a node) (b tensor))
  (node ($mm ($data a) b)
        :name :mm
        :bps (bps a (lambda (dv gv) (declare (ignore dv)) ($mm gv ($transpose b))))))

(defmethod $mm ((a tensor) (b node))
  (node ($mm a ($data b))
        :name :mm
        :bps (bps b (lambda (dv gv) (declare (ignore dv)) ($mm ($transpose a) gv)))))

(defmethod $mul ((a node) (b node))
  (node ($mul ($data a) ($data b))
        :name :mul
        :bps (bps a (lambda (dv gv) (declare (ignore dv)) ($mul ($data b) gv))
                  b (lambda (dv gv) (declare (ignore dv)) ($mul ($data a) gv)))))

(defmethod $mul ((a node) (b T))
  (node ($mul ($data a) b)
        :name :mul
        :bps (bps a (lambda (dv gv) (declare (ignore dv)) ($mul b gv)))))

(defmethod $mul ((a T) (b node))
  (node ($mul a ($data b))
        :name :mul
        :bps (bps b (lambda (dv gv) (declare (ignore dv)) ($mul a gv)))))

(defmethod $bmm ((bx node) (by node))
  (node ($bmm ($data bx) ($data by))
        :name :bmm
        :bps (bps bx (lambda (dv gv) (declare (ignore dv)) ($bmm gv ($transpose ($data by) 2 1)))
                  by (lambda (dv gv) (declare (ignore dv)) ($bmm ($transpose ($data bx) 2 1) gv)))))

(defmethod $bmm ((bx node) (by tensor))
  (node ($bmm ($data bx) by)
        :name :bmm
        :bps (bps bx (lambda (dv gv) (declare (ignore dv)) ($bmm gv ($transpose by 2 1))))))

(defmethod $bmm ((bx tensor) (by node))
  (node ($bmm bx ($data by))
        :name :bmm
        :bps (bps by (lambda (dv gv) (declare (ignore dv)) ($bmm ($transpose bx 2 1) gv)))))

(defmethod $mml ((x node) (y node))
  (cond ((and (eq 1 ($ndim x)) (eq 1 ($ndim y))) ($dot x y))
        ((and (eq 2 ($ndim x)) (eq 1 ($ndim y))) ($mv x y))
        ((and (eq 2 ($ndim x)) (eq 2 ($ndim y))) ($mm x y))
        ((and (eq 3 ($ndim x)) (eq 3 ($ndim y))) ($bmm x y))))

(defmethod $mml ((x node) (y tensor))
  (cond ((and (eq 1 ($ndim x)) (eq 1 ($ndim y))) ($dot x y))
        ((and (eq 2 ($ndim x)) (eq 1 ($ndim y))) ($mv x y))
        ((and (eq 2 ($ndim x)) (eq 2 ($ndim y))) ($mm x y))
        ((and (eq 3 ($ndim x)) (eq 3 ($ndim y))) ($bmm x y))))

(defmethod $mml ((x tensor) (y node))
  (cond ((and (eq 1 ($ndim x)) (eq 1 ($ndim y))) ($dot x y))
        ((and (eq 2 ($ndim x)) (eq 1 ($ndim y))) ($mv x y))
        ((and (eq 2 ($ndim x)) (eq 2 ($ndim y))) ($mm x y))
        ((and (eq 3 ($ndim x)) (eq 3 ($ndim y))) ($bmm x y))))

(defmethod $mml ((x node) (y number)) ($mul x y))
(defmethod $mml ((x number) (y node)) ($mul x y))

(defmethod $div ((a node) (b node))
  (node ($div ($data a) ($data b))
        :name :div
        :bps (bps a (lambda (dv gv) (declare (ignore dv)) ($div gv ($data b)))
                  b (lambda (dv gv)
                      (declare (ignore dv))
                      ($neg! ($div! ($mul ($data a) gv) ($expt ($data b) 2)))))))

(defmethod $div ((a node) (b tensor))
  (node ($div ($data a) b)
        :name :div
        :bps (bps a (lambda (dv gv) (declare (ignore dv)) ($div gv b)))))

(defmethod $div ((a tensor) (b node))
  (node ($div a ($data b))
        :name :div
        :bps (bps b (lambda (dv gv)
                      (declare (ignore dv))
                      ($neg! ($div! ($mul a gv) ($expt ($data b) 2)))))))

(defmethod $div ((a node) (b number))
  (node ($div ($data a) b)
        :name :div
        :bps (bps a (lambda (dv gv) (declare (ignore dv)) ($div gv b)))))

(defmethod $div ((a number) (b node))
  (cond ((numberp ($data b))
         (node ($div a ($data b))
               :name :div
               :bps (bps b (lambda (dv gv)
                             (declare (ignore dv))
                             (/ (* (- a) gv) (expt ($data b) 2))))))
        (T (node ($div a ($data b))
                 :name :div
                 :bps (bps b (lambda (dv gv)
                               (declare (ignore dv))
                               ($neg! ($div! ($mul a gv) ($expt ($data b) 2)))))))))

(defmethod $vv ((a node) (b node))
  (node ($vv ($data a) ($data b))
        :name :vv
        :bps (bps a (lambda (dv gv) (declare (ignore dv)) ($mv gv ($data b)))
                  b (lambda (dv gv) (declare (ignore dv)) ($mv ($transpose gv) ($data a))))))

(defmethod $vv ((a node) (b tensor))
  (node ($vv ($data a) b)
        :name :vv
        :bps (bps a (lambda (dv gv) (declare (ignore dv)) ($mv gv b)))))

(defmethod $vv ((a tensor) (b node))
  (node ($vv a ($data b))
        :name :vv
        :bps (bps b (lambda (dv gv) (declare (ignore dv)) ($mv ($transpose gv) a)))))

(defmethod $inverse ((a node))
  (node ($inverse ($data a))
        :name :inverse
        :bps (bps a (lambda (dv gv)
                      (declare (ignore dv))
                      (let ((tn ($transpose gv)))
                        ($neg! ($mm ($mm tn gv) tn)))))))

(defmethod $view ((a node) &rest sizes)
  (node (apply #'$view ($data a) sizes)
        :name :view
        :bps (bps a (lambda (dv gv) (declare (ignore dv)) ($view gv ($data a))))))

(defmethod $expand ((a node) size)
  (node ($expand ($data a) size)
        :name :expand
        :bps (bps a (lambda (dv gv)
                      (declare (ignore dv))
                      (let* ((ad ($data a))
                             (as ($size ad))
                             (out gv))
                        (loop :for dim :from 0 :below ($count as)
                              :for sz = ($ as dim)
                              :do (when (eq sz 1)
                                    (setf out ($sum out dim))))
                        out)))))

(defmethod $sum ((x node) &optional (dimension -1))
  (node ($sum ($data x) dimension)
        :name :sum
        :bps (bps x (lambda (dv gv)
                      (declare (ignore dv))
                      (if (< dimension 0)
                          ($broadcast gv ($data x))
                          ($expand gv ($size x)))))))

(defmethod $mean ((x node) &optional (dimension -1))
  (node ($mean ($data x) dimension)
        :name :mean
        :bps (bps x (lambda (dv gv)
                      (declare (ignore dv))
                      (if (< dimension 0)
                          ($broadcast (/ gv ($count ($data x))) ($data x))
                          ($div! ($expand gv ($size x)) ($size ($data x) dimension)))))))

(defun seteq! (a b v)
  (let ((m ($eq a b)))
    ($mul! ($copy! ($resize! ($empty a) a) m) v)))

(defmethod $min ((x node) &optional (dimension -1))
  (node ($min ($data x) dimension)
        :name :min
        :bps (bps x (lambda (dv gv)
                      (if (< dimension 0)
                          (seteq! ($data x) dv gv)
                          (seteq! ($data x) ($expand dv ($size x)) ($expand gv ($size x))))))))

(defmethod $max ((x node) &optional (dimension -1))
  (node ($max ($data x) dimension)
        :name :max
        :bps (bps x (lambda (dv gv)
                      (if (< dimension 0)
                          (seteq! ($data x) dv gv)
                          (seteq! ($data x) ($expand dv ($size x)) ($expand gv ($size x))))))))

(defmethod $transpose ((x node) &optional dimension0 dimension1)
  (node ($transpose ($data x) dimension0 dimension1)
        :name :transpose
        :bps (bps x (lambda (dv gv)
                      (declare (ignore dv))
                      ($transpose gv dimension0 dimension1)))))

(defmethod $reshape ((x node) &rest sizes)
  (node (apply #'$reshape ($data x) sizes)
        :name :reshape
        :bps (bps x (lambda (dv gv) (declare (ignore dv)) ($view gv ($data x))))))

(defmethod $clone ((x node))
  (node ($clone ($data x))
        :name :clone
        :bps (bps x (lambda (dv gv) (declare (ignore dv)) gv))))

(defmethod $cat ((x node) (y node) &optional (dimension 0))
  (node ($cat ($data x) ($data y) dimension)
        :name :cat
        :bps (bps x (lambda (dv gv) (declare (ignore dv)) ($narrow gv dimension 0 ($size ($data x) 1)))
                  y (lambda (dv gv) (declare (ignore dv)) ($narrow gv dimension ($size ($data x) 1)
                                                              ($size ($data y) 1))))))

(defmethod $cat ((x node) (y tensor) &optional (dimension 0))
  (node ($cat ($data x) y dimension)
        :name :cat
        :bps (bps x (lambda (dv gv) (declare (ignore dv)) ($narrow gv dimension 0 ($size ($data x) 1))))))

(defmethod $cat ((x tensor) (y node) &optional (dimension 0))
  (node ($cat x ($data y) dimension)
        :name :cat
        :bps (bps y (lambda (dv gv) (declare (ignore dv)) ($narrow gv dimension ($size x 1)
                                                              ($size ($data y) 1))))))

(defmethod $concat ((x node) nodes &rest others)
  (let ((pd ($last others)))
    (if (numberp pd)
        (let ((dimension pd)
              (xs (cons x (cons nodes (butlast others)))))
          (reduce (lambda (r n) ($cat r n dimension)) xs))
        (let ((dimension 0)
              (xs (cons x (cons nodes others))))
          (reduce (lambda (r n) ($cat r n dimension)) xs)))))

(defmethod $index ((x node) dimension (indices list))
  (node ($index ($data x) dimension indices)
        :name :index
        :bps (bps x (lambda (dv gv)
                      (declare (ignore dv))
                      (let* ((g ($zero ($data x)))
                             (gs ($index g dimension indices)))
                        (setf ($index g dimension indices)
                              (apply #'$reshape gv ($size gs)))
                        g)))))

(defmethod $index ((x node) dimension (index number))
  (node ($index ($data x) dimension index)
        :name :index
        :bps (bps x (lambda (dv gv)
                      (declare (ignore dv))
                      (let* ((indices (list index))
                             (g ($zero ($data x)))
                             (gs ($index g dimension indices)))
                        (setf ($index g dimension indices)
                              (apply #'$reshape gv ($size gs)))
                        g)))))

(defmethod $ ((x node) location &rest others-and-default)
  (node (apply #'$ ($data x) (cons location others-and-default))
        :name :$
        :bps (bps x (lambda (dv gv)
                      (declare (ignore dv))
                      (let ((z ($zero ($data x)))
                            (locs (cons location others-and-default)))
                        (if (= ($count locs) ($ndim z))
                            (setf (apply #'$ z locs) gv)
                            ($copy! (apply #'$ z (cons location others-and-default)) gv))
                        z)))))

(defmethod (setf $) (value (x node) location &rest others)
  (let ((nx ($clone ($data x))))
    (setf (apply #'$ nx (cons location others)) value)
    (cond ((typep value 'node)
           (node nx
                 :name :setf$
                 :bps (bps x (lambda (dv gv)
                               (declare (ignore dv))
                               (let ((ng ($clone gv)))
                                 (setf (apply #'$ ng (cons location others)) 0)
                                 ng))
                           value (lambda (dv gv)
                                   (declare (ignore dv))
                                   (let ((gk (apply #'$ gv (cons location others))))
                                     (if (numberp gk)
                                         gk
                                         ($clone gk)))))))
          (T (node nx
                   :name :setf$
                   :bps (bps x (lambda (dv gv)
                                 (declare (ignore dv))
                                 (let ((ng ($clone gv)))
                                   (setf (apply #'$ ng (cons location others)) 0)
                                   ng))))))))
