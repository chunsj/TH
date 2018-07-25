(defpackage :var-test
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :var-test)

(defclass var ()
  ((value :initform nil :accessor $val)
   (childs :initform nil :accessor $chs)
   (grad :initform nil :accessor $grd)))

(defun var (val)
  (let ((n (make-instance 'var)))
    (setf ($val n) val)
    n))

(defgeneric $grad (var))

(defmethod $grad ((self var))
  (unless ($grd self)
    (setf ($grd self)
          (reduce #'+ (mapcar (lambda (c)
                                (let ((weight (car c))
                                      (g ($grad (cdr c))))
                                  (* weight g)))
                              ($chs self)))))
  ($grd self))

(defmethod $add ((self var) (other var))
  (let ((z (var (+ ($val self) ($val other)))))
    (setf ($chs self) (append ($chs self) (list (cons 1.0 z))))
    (setf ($chs other) (append ($chs other) (list (cons 1.0 z))))
    z))

(defmethod $mul ((self var) (other var))
  (let ((z (var (* ($val self) ($val other)))))
    (setf ($chs self) (append ($chs self) (list (cons ($val other) z))))
    (setf ($chs other) (append ($chs other) (list (cons ($val self) z))))
    z))

(defmethod $sin ((x var))
  (let ((z (var ($sin ($val x)))))
    (setf ($chs x) (append ($chs x) (list (cons ($cos ($val x)) z))))
    z))

(let* ((x (var 0.5))
       (y (var 4.2))
       (z ($add ($mul x y) ($sin x))))
  (setf ($grd z) 1.0)
  (prn ($val z))
  (prn ($grad x))
  (prn ($grad y))
  (prn (- ($val z) (+ (* ($val x) ($val y)) (sin ($val x)))))
  (prn (- ($grad x) (+ ($val y) ($cos ($val x)))))
  (prn (- ($grad y) ($val x))))

(let* ((x ($variable '(0.5)))
       (y ($variable '(4.2)))
       (z ($add ($mul x y) ($sin x))))
  (prn z)
  (prn ($gradient x))
  (prn ($gradient y)))

(prn ($mm (ones 3 1) (tensor '((1 2 3 4)))))
(prn ($mm (tensor '((1) (2) (3) (4))) (ones 1 3)))
