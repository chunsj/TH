(defpackage :yy
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :yy)

(defclass tape () ((objects :initform nil :accessor elms)))

(defmethod $ ((tape tape) location &rest args)
  (declare (ignore args))
  ($ (elms tape) location))

(defmethod (setf $) (value (tape tape) location &rest args)
  (declare (ignore args))
  (setf ($ (elms tape) location) value))

(defmethod $count ((tape tape)) ($count (elms tape)))

(defun $append (tape object) (setf (elms tape) (append (elms tape) (list object))))
(defun $reverse (tape) (reverse (elms tape)))

(defmethod print-object ((tape tape) stream)
  (print-object (elms tape) stream))

(defclass opnode ()
  ((fun :initform nil :accessor opnodefun)
   (args :initform nil :accessor opnodeargs)
   (val :initform nil :accessor opnodeval)
   (tape :initform nil :accessor opnodetape)
   (ograd :initform 0 :accessor opnodeograd)))

(defmethod print-object ((opnode opnode) stream)
  (print-object (opnodeval opnode) stream))

(defparameter *grad-funcs* #{})

(defun opnode (val fun args tape)
  (let ((self (make-instance 'opnode)))
    (setf (opnodefun self) fun
          (opnodeargs self) args
          (opnodeval self) val
          (opnodetape self) tape)
    ($append tape self)
    self))

(defun opnodep (x) (typep x 'opnode))
(defun getval (x) (if (opnodep x) (opnodeval x) x))

(defun bp (tape)
  (loop :for node :in (reverse (elms tape))
        :for args = (opnodeargs node)
        :do (loop :for i :from 0 :below ($count args)
                  :for arg = ($ args i)
                  :for ndf = (opnodefun node)
                  :for gfs = ($ *grad-funcs* ndf nil)
                  :for ndog = (opnodeograd node)
                  :for ndargs = (opnodeargs node)
                  :do (when (opnodep arg)
                        (let ((gf ($ gfs i)))
                          (when gf
                            (setf (opnodeograd arg)
                                  ($+ (opnodeograd arg)
                                      (apply gf ndog (mapcar #'getval ndargs)))))))))
  (opnodeograd (car (elms tape))))

(defun grad (function &optional (argnum 0))
  (lambda (&rest args)
    (let ((tape (make-instance 'tape))
          (ans nil))
      (setf ($ args argnum) (opnode ($ args argnum) nil nil tape))
      (setf ans (apply function args))
      (if (not (opnodep ans))
          0.0
          (progn
            (setf (opnodeograd ans) 1)
            (bp tape))))))

(defun kyapply (function &rest args)
  (let ((parents (remove-if-not #'opnodep args)))
    (if parents
        (let ((v (apply #'kyapply function (mapcar #'getval args))))
          (opnode v function args (opnodetape (car parents))))
        (apply function args))))

(defmethod $transpose ((opnode opnode) &optional dimension0 dimension1)
  (kyapply #'$transpose opnode dimension0 dimension1))

(defmethod $ndim ((self opnode)) (kyapply #'$ndim self))
(defmethod $neg ((self opnode)) (kyapply #'$neg self))

(defmethod $add ((self opnode) (other opnode)) (kyapply #'$add self other))
(defmethod $add ((self t) (other opnode)) (kyapply #'$add self other))
(defmethod $add ((self opnode) (other t)) (kyapply #'$add self other))
(defmethod $sub ((self opnode) (other opnode)) (kyapply #'$sub self other))
(defmethod $sub ((self t) (other opnode)) (kyapply #'$sub self other))
(defmethod $sub ((self opnode) (other t)) (kyapply #'$sub self other))
(defmethod $mul ((self opnode) (other opnode)) (kyapply #'$mul self other))
(defmethod $mul ((self t) (other opnode)) (kyapply #'$mul self other))
(defmethod $mul ((self opnode) (other t)) (kyapply #'$mul self other))
(defmethod $div ((self opnode) (other opnode)) (kyapply #'$div self other))
(defmethod $div ((self t) (other opnode)) (kyapply #'$div self other))
(defmethod $div ((self opnode) (other t)) (kyapply #'$div self other))
(defmethod $dot ((self opnode) (other opnode)) (kyapply #'$dot self other))
(defmethod $dot ((self t) (other opnode)) (kyapply #'$dot self other))
(defmethod $dot ((self opnode) (other t)) (kyapply #'$dot self other))
(defmethod $expt ((self opnode) (other opnode)) (kyapply #'$expt self other))
(defmethod $expt ((self t) (other opnode)) (kyapply #'$expt self other))
(defmethod $expt ((self opnode) (other t)) (kyapply #'$expt self other))
(defmethod $vv ((self opnode) (other opnode)) (kyapply #'$vv self other))
(defmethod $vv ((self t) (other opnode)) (kyapply #'$vv self other))
(defmethod $vv ((self opnode) (other t)) (kyapply #'$vv self other))
(defmethod $mv ((self opnode) (other opnode)) (kyapply #'$mv self other))
(defmethod $mv ((self t) (other opnode)) (kyapply #'$mv self other))
(defmethod $mv ((self opnode) (other t)) (kyapply #'$mv self other))
(defmethod $mm ((self opnode) (other opnode)) (kyapply #'$mm self other))
(defmethod $mm ((self t) (other opnode)) (kyapply #'$mm self other))
(defmethod $mm ((self opnode) (other t)) (kyapply #'$mm self other))
(defmethod $mml ((self opnode) (other opnode)) (kyapply #'$mml self other))
(defmethod $mml ((self t) (other opnode)) (kyapply #'$mml self other))
(defmethod $mml ((self opnode) (other t)) (kyapply #'$mml self other))

(defun gf! (f grads) (setf ($ *grad-funcs* f) grads))

(gf! #'$abs (list (lambda (g x) ($mul (kyapply #'$sign x) g))))
(gf! #'$exp (list (lambda (g x) ($mul (kyapply #'$exp x) g))))
(gf! #'$log (list (lambda (g x) ($div g x))))
(gf! #'$sin (list (lambda (g x) ($mul g (kyapply #'$cos x)))))
(gf! #'$cos (list (lambda (g x) ($mul ($neg g) (kyapply #'$sin x)))))
(gf! #'$sign (list (lambda (g x) (declare (ignore g x)) 0)))
(gf! #'$transpose (list (lambda (g x)
                          (declare (ignore x))
                          (kyapply #'$transpose g))))

(gf! #'$add (list (lambda (g x y) (declare (ignore x y)) g)
                  (lambda (g x y) (declare (ignore x y)) g)))
(gf! #'$mul (list (lambda (g x y) (declare (ignore x)) ($mul y g))
                  (lambda (g x y) (declare (ignore y)) ($mul x g))))
(gf! #'$expt (list (lambda (g x y) ($mul! ($mul g y) ($expt x ($sub y 1))))
                   (lambda (g x y) ($mul! ($mul g ($log x)) ($expt x y)))))
(gf! #'$sub (list (lambda (g x y) (declare (ignore x y)) g)
                  (lambda (g x y) (declare (ignore x y)) ($neg g))))
(gf! #'$neg (list (lambda (g x) (declare (ignore x)) ($neg g))))
(gf! #'$div (list (lambda (g x y) (declare (ignore x)) ($div g y))
                  (lambda (g x y) ($neg! ($div! ($mul g x) ($expt y 2))))))

(gf! #'$dot (list (lambda (g a b)
                    (declare (ignore a))
                    (kyapply #'$dot g b))
                  (lambda (g a b)
                    (declare (ignore b))
                    (kyapply #'$dot a g))))
(gf! #'$mv (list (lambda (g m v)
                   (declare (ignore m))
                   (kyapply #'$vv g v))
                 (lambda (g m v)
                   (declare (ignore v))
                   (kyapply #'$mm ($transpose m) g))))
(gf! #'$mm (list (lambda (g x y)
                   (declare (ignore x))
                   (kyapply #'$mm g ($transpose y)))
                 (lambda (g x y)
                   (declare (ignore y))
                   (kyapply #'$mm ($transpose x) g))))


(defmacro defgrad (df f &optional (argnum 0)) `(setf (symbol-function ',df) (grad #',f ,argnum)))

(defun f (x) (kyapply #'$sin x))
(defgrad df f)
(defgrad ddf df)

(let ((v (* 0.5 pi)))
  (print (f v))
  (print (df v))
  (print (ddf v)))

(defun ftanh (x)
  ($/ ($- 1 (kyapply #'$exp ($* -2 x)))
      ($+ 1 (kyapply #'$exp ($* -2 x)))))
(defgrad dftanh ftanh)
(defgrad ddftanh dftanh)

(print (ftanh 0))
(print ($tanh 0))
(print (dftanh 0))
(print ($- 1 ($expt ($tanh 0) 2)))
(print (ftanh 2.2))
(print ($tanh 2.2))

(defun mv (a b) (kyapply #'$mv a b))
(defgrad dmv mv)
(defgrad dmv2 mv 1)

(let* ((m (tensor '((2 0) (0 2))))
       (v (tensor '(2 3))))
  (print (mv m v))
  (print (dmv m v))
  (print (dmv2 m v)))
