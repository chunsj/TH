(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defmethod $add ((y parameter) (x parameter))
  ($operation ($add ($data y) ($data x))
              :creators (list y x)
              :name :add
              :bfn (lambda (self gradient yd xd)
                     (declare (ignore yd xd))
                     ($bp! y gradient self)
                     ($bp! x gradient self))))

(defmethod $add ((y t) (x parameter))
  ($operation ($add y ($data x))
              :creators (list x)
              :name :add
              :bfn (lambda (self gradient xd)
                     (declare (ignore xd))
                     ($bp! x gradient self))))

(defmethod $add ((y parameter) (x t))
  ($operation ($add ($data y) x)
              :creators (list y)
              :name :add
              :bfn (lambda (self gradient yd)
                     (declare (ignore yd))
                     ($bp! y gradient self))))

(defmethod $sub ((y parameter) (x parameter))
  ($operation ($sub ($data y) ($data x))
              :creators (list y x)
              :name :sub
              :bfn (lambda (self gradient yd xd)
                     (declare (ignore yd xd))
                     ($bp! y gradient self)
                     ($bp! x ($neg gradient) self))))

(defmethod $sub ((y t) (x parameter))
  ($operation ($sub y ($data x))
              :creators (list x)
              :name :sub
              :bfn (lambda (self gradient xd)
                     (declare (ignore xd))
                     ($bp! x ($neg gradient) self))))

(defmethod $sub ((y parameter) (x t))
  ($operation ($sub ($data y) x)
              :creators (list y)
              :name :sub
              :bfn (lambda (self gradient yd)
                     (declare (ignore yd))
                     ($bp! y gradient self))))

(defmethod $neg ((y parameter))
  ($operation ($neg ($data y))
              :creators (list y)
              :name :neg
              :bfn (lambda (self gradient yd)
                     (declare (ignore yd))
                     ($bp! y ($neg gradient) self))))

(defmethod $mul ((y parameter) (x parameter))
  ($operation ($mul ($data y) ($data x))
              :creators (list y x)
              :name :mul
              :bfn (lambda (self gradient yd xd)
                     ($bp! y ($mul xd gradient) self)
                     ($bp! x ($mul yd gradient) self))))

(defmethod $mul ((y t) (x parameter))
  ($operation ($mul y ($data x))
              :creators (list x)
              :name :mul
              :bfn (lambda (self gradient xd)
                     (declare (ignore xd))
                     ($bp! x ($mul y gradient) self))))

(defmethod $mul ((y parameter) (x t))
  ($operation ($mul ($data y) x)
              :creators (list y)
              :name :mul
              :bfn (lambda (self gradient yd)
                     (declare (ignore yd))
                     ($bp! y ($mul x gradient) self))))

(defmethod $div ((a parameter) (b parameter))
  ($operation ($div ($data a) ($data b))
              :creators (list a b)
              :name :div
              :bfn (lambda (self gradient ad bd)
                     ($bp! a ($div gradient bd) self)
                     ($bp! b ($div! ($mul! ($neg ad) gradient) ($expt bd 2D0)) self))))

(defmethod $div ((a parameter) (b t))
  ($operation ($div ($data a) b)
              :creators (list a)
              :name :div
              :bfn (lambda (self gradient ad)
                     (declare (ignore ad))
                     ($bp! a ($div gradient b) self))))

(defmethod $div ((a t) (b parameter))
  ($operation ($div a ($data b))
              :creators (list b)
              :name :div
              :bfn (lambda (self gradient bd)
                     ($bp! b ($div! ($mul! ($neg a) gradient) ($expt bd 2D0)) self))))

(defmethod $dot ((a parameter) (b parameter))
  ($operation ($dot ($data a) ($data b))
              :creators (list a b)
              :name :dot
              :bfn (lambda (self gradient ad bd)
                     ($bp! a ($mul bd gradient) self)
                     ($bp! b ($mul ad gradient) self))))

(defmethod $dot ((a t) (b parameter))
  ($operation ($dot a ($data b))
              :creators (list b)
              :name :dot
              :bfn (lambda (self gradient bd)
                     (declare (ignore bd))
                     ($bp! b ($mul a gradient) self))))

(defmethod $dot ((a parameter) (b t))
  ($operation ($dot ($data a) b)
              :creators (list a)
              :name :dot
              :bfn (lambda (self gradient ad)
                     (declare (ignore ad))
                     ($bp! a ($mul b gradient) self))))

(defmethod $vv ((a parameter) (b parameter))
  ($operation ($vv ($data a) ($data b))
              :creators (list a b)
              :name :vv
              :bfn (lambda (self gradient ad bd)
                     ($bp! a ($mv gradient bd) self)
                     ($bp! b ($mv ($transpose gradient) ad) self))))

(defmethod $vv ((a parameter) (b tensor))
  ($operation ($vv ($data a) b)
              :creators (list a)
              :name :vv
              :bfn (lambda (self gradient ad)
                     (declare (ignore ad))
                     ($bp! a ($mv gradient b) self))))

(defmethod $vv ((a tensor) (b parameter))
  ($operation ($vv a ($data b))
              :creators (list b)
              :name :vv
              :bfn (lambda (self gradient bd)
                     (declare (ignore bd))
                     ($bp! b ($mv ($transpose gradient) a) self))))

(defmethod $mv ((m parameter) (v parameter))
  ($operation ($mv ($data m) ($data v))
              :creators (list m v)
              :name :mv
              :bfn (lambda (self gradient md vd)
                     ($bp! m ($vv gradient vd) self)
                     ($bp! v ($@ ($transpose md) gradient) self))))

(defmethod $mv ((m parameter) (v tensor))
  ($operation ($mv ($data m) v)
              :creators (list m)
              :name :mv
              :bfn (lambda (self gradient md)
                     (declare (ignore md))
                     ($bp! m ($vv gradient v) self))))

(defmethod $mv ((m tensor) (v parameter))
  ($operation ($mv m ($data v))
              :creators (list v)
              :name :mv
              :bfn (lambda (self gradient vd)
                     (declare (ignore vd))
                     ($bp! v ($@ ($transpose m) gradient) self))))

(defmethod $mm ((a parameter) (b parameter))
  ($operation ($mm ($data a) ($data b))
              :creators (list a b)
              :name :mm
              :bfn (lambda (self gradient ad bd)
                     ($bp! a ($mm gradient ($transpose bd)) self)
                     ($bp! b ($mm ($transpose ad) gradient) self))))

(defmethod $mm ((a parameter) (b tensor))
  ($operation ($mm ($data a) b)
              :creators (list a)
              :name :mm
              :bfn (lambda (self gradient ad)
                     (declare (ignore ad))
                     ($bp! a ($mm gradient ($transpose b)) self))))

(defmethod $mm ((a tensor) (b parameter))
  ($operation ($mm a ($data b))
              :creators (list b)
              :name :mm
              :bfn (lambda (self gradient bd)
                     (declare (ignore bd))
                     ($bp! b ($mm ($transpose a) gradient) self))))

(defmethod $bmm ((bx parameter) (by parameter))
  ($operation ($bmm ($data bx) ($data by))
              :creators (list bx by)
              :name :bmm
              :bfn (lambda (self gradient bxd byd)
                     ($bp! bx ($bmm gradient ($transpose byd 2 1)) self)
                     ($bp! by ($bmm ($transpose bxd 2 1) gradient) self))))

(defmethod $bmm ((bx parameter) (by tensor))
  ($operation ($bmm ($data bx) by)
              :creators (list bx)
              :name :bmm
              :bfn (lambda (self gradient bxd)
                     (declare (ignore bxd))
                     ($bp! bx ($bmm gradient ($transpose by 2 1)) self))))

(defmethod $bmm ((bx tensor) (by parameter))
  ($operation ($bmm bx ($data by))
              :creators (list by)
              :name :bmm
              :bfn (lambda (self gradient byd)
                     (declare (ignore byd))
                     ($bp! by ($bmm ($transpose bx 2 1) gradient) self))))

(defmethod $mml ((x parameter) (y parameter))
  (cond ((and (eq 1 ($ndim x)) (eq 1 ($ndim y))) ($dot x y))
        ((and (eq 2 ($ndim x)) (eq 1 ($ndim y))) ($mv x y))
        ((and (eq 2 ($ndim x)) (eq 2 ($ndim y))) ($mm x y))
        ((and (eq 3 ($ndim x)) (eq 3 ($ndim y))) ($bmm x y))))

(defmethod $mml ((x parameter) (y tensor))
  (cond ((and (eq 1 ($ndim x)) (eq 1 ($ndim y))) ($dot x y))
        ((and (eq 2 ($ndim x)) (eq 1 ($ndim y))) ($mv x y))
        ((and (eq 2 ($ndim x)) (eq 2 ($ndim y))) ($mm x y))
        ((and (eq 3 ($ndim x)) (eq 3 ($ndim y))) ($bmm x y))))

(defmethod $mml ((x tensor) (y parameter))
  (cond ((and (eq 1 ($ndim x)) (eq 1 ($ndim y))) ($dot x y))
        ((and (eq 2 ($ndim x)) (eq 1 ($ndim y))) ($mv x y))
        ((and (eq 2 ($ndim x)) (eq 2 ($ndim y))) ($mm x y))
        ((and (eq 3 ($ndim x)) (eq 3 ($ndim y))) ($bmm x y))))

(defmethod $mml ((x parameter) (y number)) ($mul x y))
(defmethod $mml ((x number) (y parameter)) ($mul x y))

(defmethod $sum ((x parameter) &optional (dimension -1))
  ($operation ($sum ($data x) dimension)
              :creators (list x)
              :name :sum
              :bfn (lambda (self gradient xd)
                     ($bp! x
                           (if (< dimension 0)
                               ($mul gradient ($one xd))
                               ($expand gradient ($size xd)))
                           self))))

(defmethod $mean ((x parameter) &optional (dimension -1))
  ($operation ($mean ($data x) dimension)
              :creators (list x)
              :name :mean
              :bfn (lambda (self gradient xd)
                     ($bp! x
                           (if (< dimension 0)
                               ($mul ($div gradient ($count x)) ($one xd))
                               ($div ($expand gradient ($size xd))
                                     ($size xd dimension)))
                           self))))

(defun meqv (a b v) ($mul (tensor ($eq a b)) v))

(defmethod $min ((x parameter) &optional (dimension -1))
  ($operation ($min ($data x) dimension)
              :creators (list x)
              :name :min
              :bfn (lambda (self gradient xd)
                     ($bp! x
                           (if (< dimension 0)
                               (meqv xd self gradient)
                               (meqv xd ($expand self ($size xd)) ($expand gradient ($size xd))))
                           self))))

(defmethod $max ((x parameter) &optional (dimension -1))
  ($operation ($max ($data x) dimension)
              :creators (list x)
              :name :max
              :bfn (lambda (self gradient xd)
                     ($bp! x
                           (if (< dimension 0)
                               (meqv xd self gradient)
                               (meqv xd ($expand self ($size xd)) ($expand gradient ($size xd))))
                           self))))
