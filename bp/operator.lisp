(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defun broadcast-number (c m)
  (node ($mul! ($one m) ($data c))
        :name :broadcast
        :link (link (to c ($dot ($one m) gv)))))

(defun broadcast-1x1 (c m)
  (node ($mul! ($one m) ($data c))
        :name :broadcast
        :link (link (to c ($dot ($one m) gv)))))

(defmethod $broadcast ((c node) (m node))
  (cond ((numberp ($data c)) (broadcast-number c ($data m)))
        ((eq 1 ($count ($data c))) (broadcast-1x1 c ($data m)))
        (t "cannot broadcast automatically other than number.")))

(defmethod $broadcast ((c node) (m tensor))
  (cond ((numberp ($data c)) (broadcast-number c m))
        ((eq 1 ($count ($data c))) (broadcast-1x1 c m))
        (t "cannot broadcast automatically other than number.")))

(defmethod $broadcast ((c number) (m node)) ($mul! ($one ($data m)) c))

(defun fit (gv to)
  (cond (($tensorp to) gv)
        ((numberp to) ($sum gv))))

(defmethod $add ((a node) (b node))
  (node ($add ($data a) ($data b))
        :name :add
        :link (link
                (to a (fit gv ($data a)))
                (to b (fit gv ($data b))))))

(defmethod $add ((a node) (b tensor))
  (node ($add ($data a) b)
        :name :add
        :link (link (to a (fit gv ($data a))))))

(defmethod $add ((a node) (b number))
  (node ($add ($data a) b)
        :name :add
        :link (link (to a gv))))

(defmethod $add ((a tensor) (b node)) ($add b a))
(defmethod $add ((a number) (b node)) ($add b a))

(defmethod $sub ((a node) (b node))
  (node ($sub ($data a) ($data b))
        :name :sub
        :link (link
                (to a gv)
                (to b ($neg gv)))))

(defmethod $sub ((a node) (b node))
  (node ($sub ($data a) ($data b))
        :name :sub
        :link (link
                (to a (fit gv ($data a)))
                (to b (fit ($neg gv) ($data b))))))

(defmethod $sub ((a node) (b tensor))
  (node ($sub ($data a) b)
        :name :sub
        :link (link (to a (fit gv ($data a))))))

(defmethod $sub ((a node) (b number))
  (node ($sub ($data a) b)
        :name :sub
        :link (link (to a gv))))

(defmethod $sub ((a tensor) (b node))
  (node ($sub a ($data b))
        :name :sub
        :link (link (to b (fit ($neg gv) ($data b))))))

(defmethod $sub ((a number) (b node))
  (node ($sub a ($data b))
        :name :sub
        :link (link (to b ($neg gv)))))

(defmethod $neg ((a node))
  (node ($neg ($data a))
        :name :neg
        :link (link (to a ($neg gv)))))

(defmethod $dot ((a node) (b node))
  (node ($dot ($data a) ($data b))
        :name :dot
        :link (link
                (to a ($mul ($data b) gv))
                (to b ($mul ($data a) gv)))))

(defmethod $dot ((a node) (b tensor))
  (node ($dot ($data a) b)
        :name :dot
        :link (link (to a ($mul b gv)))))

(defmethod $dot ((a tensor) (b node)) ($dot b a))

(defun $tvv (m v)
  (let* ((mt ($transpose m))
         (r ($mv mt v)))
    r))

(defmethod $mv ((m node) (v node))
  (node ($mv ($data m) ($data v))
        :name :mv
        :link (link
                (to m ($vv gv ($data v)))
                (to v ($tvv ($data m) gv)))))

(defmethod $mv ((m node) (v tensor))
  (node ($mv ($data m) v)
        :name :mv
        :link (link (to m ($vv gv v)))))

(defmethod $mv ((m tensor) (v node))
  (node ($mv m ($data v))
        :name :mv
        :link (link (to v ($tvv m gv)))))

(defun $mmt (a b)
  (let* ((bt ($transpose b))
         (r ($mm a bt)))
    r))

(defun $tmm (a b)
  (let* ((at ($transpose a))
         (r ($mm at b)))
    r))

(defmethod $mm ((a node) (b node))
  (node ($mm ($data a) ($data b))
        :name :mm
        :link (link
                (to a ($mmt gv ($data b)))
                (to b ($tmm ($data a) gv)))))

(defmethod $mm ((a node) (b tensor))
  (node ($mm ($data a) b)
        :name :mm
        :link (link (to a ($mmt gv b)))))

(defmethod $mm ((a tensor) (b node))
  (node ($mm a ($data b))
        :name :mm
        :link (link (to b ($tmm a gv)))))

(defmethod $mul ((a node) (b node))
  (node ($mul ($data a) ($data b))
        :name :mul
        :link (link
                (to a (fit ($mul ($data b) gv) ($data a)))
                (to b (fit ($mul ($data a) gv) ($data b))))))

(defmethod $mul ((a node) (b T))
  (node ($mul ($data a) b)
        :name :mul
        :link (link (to a ($mul b gv)))))

(defmethod $mul ((a T) (b node))
  (node ($mul a ($data b))
        :name :mul
        :link (link (to b ($mul a gv)))))

(defmethod $bmm ((bx node) (by node))
  (node ($bmm ($data bx) ($data by))
        :name :bmm
        :link (link
                (to bx ($bmm gv ($transpose ($data by) 2 1)))
                (to by ($bmm ($transpose ($data bx) 2 1) gv)))))

(defmethod $bmm ((bx node) (by tensor))
  (node ($bmm ($data bx) by)
        :name :bmm
        :link (link (to bx ($bmm gv ($transpose by 2 1))))))

(defmethod $bmm ((bx tensor) (by node))
  (node ($bmm bx ($data by))
        :name :bmm
        :link (link (to by ($bmm ($transpose bx 2 1) gv)))))

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
        :link (link
                (to a (fit ($div gv ($data b)) ($data a)))
                (to b (fit ($div! ($mul ($neg gv) ($data a)) ($square ($data b))) ($data b))))))

(defmethod $div ((a node) (b tensor))
  (node ($div ($data a) b)
        :name :div
        :link (link (to a (fit ($div gv b) ($data a))))))

(defmethod $div ((a tensor) (b node))
  (node ($div a ($data b))
        :name :div
        :link (link (to b (fit ($div! ($mul ($neg gv) a) ($square ($data b))) ($data b))))))

(defmethod $div ((a node) (b number))
  (node ($div ($data a) b)
        :name :div
        :link (link (to a ($div gv b)))))

(defmethod $div ((a number) (b node))
  (cond ((numberp ($data b))
         (node ($div a ($data b))
               :name :div
               :link (link (to b (/ (* (- a) gv) (expt ($data b) 2))))))
        (T (node ($div a ($data b))
                 :name :div
                 :link (link (to b ($div! ($mul ($neg gv) a) ($square ($data b)))))))))

(defmethod $vv ((a node) (b node))
  (node ($vv ($data a) ($data b))
        :name :vv
        :link (link
                (to a ($mv gv ($data b)))
                (to b ($mv ($transpose gv) ($data a))))))

(defmethod $vv ((a node) (b tensor))
  (node ($vv ($data a) b)
        :name :vv
        :link (link (to a ($mv gv b)))))

(defmethod $vv ((a tensor) (b node))
  (node ($vv a ($data b))
        :name :vv
        :link (link (to b ($mv ($transpose gv) a)))))

(defmethod $inverse ((a node))
  (node ($inverse ($data a))
        :name :inverse
        :link (link (to a (let ((tn ($transpose gv))) ($neg! ($mm ($mm tn gv) tn)))))))

(defmethod $view ((a node) &rest sizes)
  (node (apply #'$view ($data a) sizes)
        :name :view
        :link (link (to a ($view gv ($data a))))))

(defmethod $expand ((a node) size)
  (node ($expand ($data a) size)
        :name :expand
        :link (link (to a (let* ((ad ($data a))
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
        :link (link (to x (let ((xd ($data x)))
                            (if (< dimension 0)
                                ($broadcast gv xd)
                                ($expand gv ($size xd))))))))

(defmethod $mean ((x node) &optional (dimension -1))
  (node ($mean ($data x) dimension)
        :name :mean
        :link (link (to x (let ((xd ($data x)))
                            (if (< dimension 0)
                                ($broadcast (/ gv ($count xd)) xd)
                                ($div! ($expand gv ($size x)) ($size xd dimension))))))))

(defmethod $var ((x node) &optional (dimension -1) biased)
  (node ($var ($data x) dimension biased)
        :name :var
        :link (link (to x (let* ((xd ($data x))
                                 (md ($mean xd dimension)))
                            (if (< dimension 0)
                                ($mul! ($sub xd md) (/ (* 2 gv) ($size x)))
                                ($mul! ($div! ($expand ($mul gv 2) ($size x))
                                              ($size xd dimension))
                                       ($sub xd md))))))))

(defmethod $sd ((x node) &optional (dimension -1) biased)
  (node ($sd ($data x) dimension biased)
        :name :sd
        :link (link (to x (let* ((xd ($data x))
                                 (md ($mean xd dimension)))
                            (if (< dimension 0)
                                ($mul! ($sub xd md) (/ gv ($size x) dv))
                                ($mul! ($div! ($expand ($div gv dv) ($size x))
                                              ($size xd dimension))
                                       ($sub xd md))))))))

(defun seteq! (a b v)
  (let ((m ($eq a b)))
    ($mul! ($copy! ($clear a) m) v)))

(defmethod $min ((x node) &optional (dimension -1))
  (node ($min ($data x) dimension)
        :name :min
        :link (link (to x (let* ((xd ($data x)) (xs ($size xd)))
                            (if (< dimension 0)
                                (seteq! xd dv gv)
                                (seteq! xd ($expand dv xs) ($expand gv xs))))))))

(defmethod $max ((x node) &optional (dimension -1))
  (node ($max ($data x) dimension)
        :name :max
        :link (link (to x (let* ((xd ($data x)) (xs ($size xd)))
                            (if (< dimension 0)
                                (seteq! xd dv gv)
                                (seteq! xd ($expand dv xs) ($expand gv xs))))))))

(defmethod $clamp ((x node) min max)
  (node ($clamp ($data x) min max)
        :name :clamp
        :link (link (to x (let* ((xd ($data x)))
                            (seteq! xd dv gv))))))

(defmethod $transpose ((x node) &optional dimension0 dimension1)
  (node ($transpose ($data x) dimension0 dimension1)
        :name :transpose
        :link (link (to x ($transpose gv dimension0 dimension1)))))

(defmethod $reshape ((x node) &rest sizes)
  (node (apply #'$reshape ($data x) sizes)
        :name :reshape
        :link (link (to x ($view gv ($data x))))))

(defmethod $clone ((x node))
  (node ($clone ($data x))
        :name :clone
        :link (link (to x gv))))

(defmethod $cat ((x node) (y node) &optional (dimension 0))
  (node ($cat ($data x) ($data y) dimension)
        :name :cat
        :link (link
                (to x ($narrow gv dimension 0 ($size ($data x) dimension)))
                (to y ($narrow gv dimension ($size ($data x) dimension)
                               ($size ($data y) dimension))))))

(defmethod $cat ((x node) (y tensor) &optional (dimension 0))
  (node ($cat ($data x) y dimension)
        :name :cat
        :link (link (to x ($narrow gv dimension 0 ($size ($data x) dimension))))))

(defmethod $cat ((x tensor) (y node) &optional (dimension 0))
  (node ($cat x ($data y) dimension)
        :name :cat
        :link (link (to y ($narrow gv dimension ($size x dimension)
                                   ($size ($data y) dimension))))))

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
        :link (link (to x (let* ((g ($zero ($data x)))
                                 (gs ($index g dimension indices)))
                            (setf ($index g dimension indices)
                                  (apply #'$reshape gv ($size gs)))
                            g)))))

(defmethod $index ((x node) dimension (indices tensor.long))
  (node ($index ($data x) dimension indices)
        :name :index
        :link (link (to x (let* ((g ($zero ($data x)))
                                 (gs ($index g dimension indices)))
                            (setf ($index g dimension indices)
                                  (apply #'$reshape gv ($size gs)))
                            g)))))

(defmethod $index ((x node) dimension (indices tensor.int))
  (node ($index ($data x) dimension indices)
        :name :index
        :link (link (to x (let* ((g ($zero ($data x)))
                                 (gs ($index g dimension indices)))
                            (setf ($index g dimension indices)
                                  (apply #'$reshape gv ($size gs)))
                            g)))))

(defmethod $index ((x node) dimension (index number))
  (node ($index ($data x) dimension index)
        :name :index
        :link (link (to x (let* ((indices (list index))
                                 (g ($zero ($data x)))
                                 (gs ($index g dimension indices)))
                            (setf ($index g dimension indices)
                                  (apply #'$reshape gv ($size gs)))
                            g)))))

(defmethod $ ((x node) location &rest others-and-default)
  (node (apply #'$ ($data x) (cons location others-and-default))
        :name :$
        :link (link (to x (let ((z ($zero ($data x)))
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
                 :link (link
                         (to x (let ((ng ($clone gv)))
                                 (setf (apply #'$ ng (cons location others)) 0)
                                 ng))
                         (to value (let ((gk (apply #'$ gv (cons location others))))
                                     (if (numberp gk)
                                         gk
                                         ($clone gk)))))))
          (T (node nx
                   :name :setf$
                   :link (link (to x (let ((ng ($clone gv)))
                                       (setf (apply #'$ ng (cons location others)) 0)
                                       ng))))))))

(defmethod $squeeze ((x node) &optional dimension)
  (if dimension
      (node ($squeeze ($data x) dimension)
            :name :squeeze
            :link (link (to x ($unsqueeze gv dimension))))
      (error "backprop does not work with implicit dimension")))

(defmethod $unsqueeze ((x node) dimension)
  (node ($unsqueeze ($data x) dimension)
        :name :unsqueeze
        :link (link (to x ($squeeze gv dimension)))))

(defmethod $narrow ((x node) dimension first-index size)
  (node ($narrow ($data x) dimension first-index size)
        :name :narrow
        :link (link (to x (let ((g ($zero ($data x))))
                            (setf ($narrow g dimension first-index size) gv)
                            g)))))

(defmethod $gather ((x node) dimension indices)
  (node ($gather ($data x) dimension indices)
        :name :gather
        :link (link (to x (let ((g ($zero ($data x))))
                            ($scatter! g dimension indices gv)
                            g)))))
