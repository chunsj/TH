(in-package :th)

(defclass tensor (th.object) ())
(defclass tensor.integral (tensor) ())
(defclass tensor.fractional (tensor) ())

(defclass tensor.byte (tensor.integral) ())
(defclass tensor.char (tensor.integral) ())
(defclass tensor.short (tensor.integral) ())
(defclass tensor.int (tensor.integral) ())
(defclass tensor.long (tensor.integral) ())
(defclass tensor.float (tensor.fractional) ())
(defclass tensor.double (tensor.fractional) ())

(defgeneric allocate-tensor (tensor &optional dimensions))

(defun mkdims (seqs)
  (if (listp seqs)
      (cons ($count seqs) (mkdims (car seqs)))
      nil))

(defun setseqs (ts seqs dims)
  (let ((nd ($count dims)))
    (cond ((eq nd 1) (loop :for i :from 0 :below ($0 dims)
                           :do (setf (tensor-at ts i) ($ seqs i))))
          ((eq nd 2) (loop :for i :from 0 :below ($0 dims)
                           :do (loop :for j :from 0 :below ($1 dims)
                                     :for v = ($ ($ seqs i) j)
                                     :do (setf (tensor-at ts i j) v))))
          ((eq nd 3) (loop :for i :from 0 :below ($0 dims)
                           :do (loop :for j :from 0 :below ($1 dims)
                                     :do (loop :for k :from 0 :below ($2 dims)
                                               :for v = ($ ($ ($ seqs i) j) k)
                                               :do (setf (tensor-at ts i j k) v)))))
          ((eq nd 4) (loop :for i :from 0 :below ($0 dims)
                           :do (loop :for j :from 0 :below ($1 dims)
                                     :do (loop :for k :from 0 :below ($2 dims)
                                               :do (loop :for l :from 0 :below ($3 dims)
                                                         :for v = ($ ($ ($ ($ seqs i) j) k) l)
                                                         :do (setf (tensor-at ts i j k l) v)))))))
    ts))

(defun make-tensor (cls &optional dimensions)
  (let ((tensor (make-instance cls)))
    (allocate-tensor tensor dimensions)
    tensor))

(defun make-tensor-seqs (cls seqs)
  (let* ((dims (mkdims seqs))
         (tensor (make-tensor cls dims)))
    (setseqs tensor seqs dims)))

(defun make-tensor-args (cls args)
  (cond ((and (eq 1 ($count args)) (listp (car args))) (make-tensor-seqs cls (car args)))
        ((and (eq 1 ($count args)) ($tensorp (car args)))
         (if (typep (car args) cls)
             (tensor-with-tensor (car args))
             (let ((nt (make-tensor cls))
                   (src (car args)))
               (tensor-resize-as nt src)
               (tensor-copy nt src)
               nt)))
        ((and (eq 2 ($count args)) (listp ($0 args)) (listp ($1 args)))
         (tensor-with-storage nil 0 ($0 args) ($1 args)))
        (($storagep (car args)) (apply #'tensor-with-storage args))
        (t (make-tensor cls args))))

(defun tensor.byte (&rest args) (make-tensor-args 'tensor.byte args))
(defun tensor.char (&rest args) (make-tensor-args 'tensor.char args))
(defun tensor.short (&rest args) (make-tensor-args 'tensor.short args))
(defun tensor.int (&rest args) (make-tensor-args 'tensor.int args))
(defun tensor.long (&rest args) (make-tensor-args 'tensor.long args))
(defun tensor.float (&rest args) (make-tensor-args 'tensor.float args))
(defun tensor.double (&rest args) (make-tensor-args 'tensor.double args))

(defparameter *default-tensor-class* 'tensor.double)

(defun tensor (&rest args)
  "Creates a new tensor with size specification or list contents or other tensor."
  (make-tensor-args *default-tensor-class* args))

(defun eye (m &optional n)
  "Creates a identity matrix."
  ($eye! (tensor) m n))

(defun linspace (a b &optional (n 100))
  "Returns a one-dimensional tensor with equally spaced points between a and b."
  (tensor-linspace (tensor) a b n))

(defun logspace (a b &optional (n 100))
  "Returns a one-dimensional tensor with logarithmically equally spaced points between a and b."
  (tensor-logspace (tensor) a b n))

(defun zeros (&rest sizes)
  "Returns a new tensor filled with zero with given sizes"
  (let ((x (make-tensor *default-tensor-class* sizes)))
    ($zero! x)
    x))

(defun ones (&rest sizes)
  "Returns a new tensor filled with one with given sizes"
  (let ((x (make-tensor *default-tensor-class* sizes)))
    ($one! x)
    x))

(defun filled (value &rest sizes)
  "Returns a new tensor filled with value with given sizes"
  (let ((x (make-tensor *default-tensor-class* (or sizes '(1)))))
    ($fill x value)
    x))

(defun rnd (&rest sizes)
  "Returns a new tensor filled with uniform random numbers between 0 and 1."
  (let ((tensor (tensor)))
    (tensor-randn tensor sizes)
    tensor))

(defun rndn (&rest sizes)
  "Returns a new tensor filled with normal random numbers N(0,1)."
  (let ((tensor (tensor)))
    (tensor-randn tensor sizes)
    tensor))

(defun rndperm (n)
  "Returns random permutation of integers from 1 to n."
  (let ((result (tensor)))
    (tensor-rand-perm result n)
    result))

(defun range (from to &optional (step 1) type)
  (let* ((type (or type *default-tensor-class*))
         (tensor (make-tensor type)))
    (tensor-range tensor from to step)
    tensor))

(defun arange (from to &optional (step 1) type)
  (let* ((type (or type *default-tensor-class*))
         (tensor (make-tensor type)))
    (tensor-arange tensor from to step)
    tensor))
