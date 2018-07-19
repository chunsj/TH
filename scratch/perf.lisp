(defpackage :perfcheck
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :perfcheck)

(defun cs (node) (th::$children node))

(defparameter *step-size* 10)
(defparameter *x-size* 5)
(defparameter *h-size* 8)

(defparameter *wx* ($variable (rnd *x-size* *h-size*)))
(defparameter *wh* ($variable (rnd *h-size* *h-size*)))
(defparameter *bh* ($variable (ones 1 *h-size*)))
(defparameter *wc* ($variable (rnd *h-size* *h-size*)))
(defparameter *bc* ($variable (ones 1 *h-size*)))
(defparameter *wy* ($variable (rnd *h-size* *x-size*)))
(defparameter *by* ($variable (ones 1 *x-size*)))

(defparameter *names* #{})

(setf ($ *names* *wx*) "wx")
(setf ($ *names* *wh*) "wh")
(setf ($ *names* *bh*) "bh")
(setf ($ *names* *wc*) "wc")
(setf ($ *names* *bc*) "bc")
(setf ($ *names* *wy*) "wy")
(setf ($ *names* *by*) "by")

(defun children (node)
  (let* ((children (cs node))
         (cnt ($count children)))
    (if (= cnt 0)
        nil
        (apply #'append children
               (loop :for c :in children
                     :collect (children c))))))

(defun allchildren (node) (remove-duplicates (children node)))

(let ((ph ($constant (zeros 1 *h-size*)))
      (yy nil)
      (ys nil))
  (loop :for step :from 0 :below 2
        :for xt = ($constant (ones 1 *x-size*))
        :for ht = ($tanh ($+ ($@ xt *wx*) ($@ ph *wh*) *bh*))
        :for yt = ($+ ($@ ht *wy*) *by*)
        :do (progn
              (setf ($ *names* xt) (format nil "xt:~A" step))
              (setf ($ *names* ht) (format nil "ht:~A" step))
              (setf ($ *names* yt) (format nil "yt:~A" step))
              (setf ph ht)
              (setf yy yt)
              (push yt ys)))
  (loop :for n :in (allchildren yy)
        :for nn = ($ *names* n)
        :do (progn
              (prn nn)
              (prn n))))

(let ((ph ($constant (zeros 1 *h-size*)))
      (pc ($constant (zeros 1 *h-size*)))
      (yy nil)
      (ys nil))
  (loop :for step :from 0 :below 100
        :for xt = ($constant (ones 1 *x-size*))
        :for ht = ($tanh ($+ ($@ xt *wx*) ($@ ph *wh*) *bh*))
        :for ct = ($sigmoid ($+ ($@ xt *wx*) ($@ pc *wc*) *bc*))
        :for yt = ($+ ($@ ($+ ht ct) *wy*) *by*)
        :do (progn
              (setf ph ht)
              (setf pc ct)
              (setf yy yt)
              (push yt ys)))
  (prn (reduce #'+ (mapcar (lambda (y) ($count (children y))) ys)))
  (prn ($bp! yy)))

(- 60 42)
(- 42 24)
(- 22 13)
(- 31 22)
