(declaim (optimize (speed 3) (debug 1) (safety 0)))

(defpackage :th.layers
  (:use #:common-lisp
        #:mu
        #:th)
  (:export #:$execute
           #:sequence-layer
           #:affine-layer))

(in-package :th.layers)

(defgeneric $execute (layer x &key train))

(defgeneric $train-parameters (layer))
(defgeneric $evaluation-parameters (layer))

(defclass layer () ())

(defmethod $train-parameters ((l layer)) nil)
(defmethod $evaluation-parameters ((l layer)) (mapcar #'$data ($train-parameters l)))

(defmethod $execute ((l layer) x &key (train t))
  (declare (ignore x train))
  nil)

(defmethod $gd! ((l layer) &optional (learning-rate 0.01))
  ($gd! ($train-parameters l) learning-rate))

(defmethod $mgd! ((l layer) &optional (learning-rate 0.01) (momentum 0.9))
  ($mgd! ($train-parameters l) learning-rate momentum))

(defmethod $agd! ((l layer) &optional (learning-rate 0.01))
  ($agd! ($train-parameters l) learning-rate))

(defmethod $amgd! ((l layer) &optional (learning-rate 0.01) (β1 0.9) (β2 0.999))
  ($amgd! ($train-parameters l) learning-rate β1 β2))

(defmethod $rmgd! ((l layer) &optional (learning-rate 0.001) (decay-rate 0.99))
  ($rmgd! ($train-parameters l) learning-rate decay-rate))

(defmethod $adgd! ((l layer) &optional (decay-rate 0.95))
  ($adgd! ($train-parameters l) decay-rate))

(defmethod $cg! ((l layer)) ($cg! ($train-parameters l)))

(defclass sequence-layer (layer)
  ((ls :initform nil)))

(defun sequence-layer (&rest layers)
  (let ((n (make-instance 'sequence-layer)))
    (with-slots (ls) n
      (setf ls layers))
    n))

(defmethod $train-parameters ((l sequence-layer))
  (with-slots (ls) l
    (loop :for e :in ls
          :appending ($train-parameters e))))

(defmethod $execute ((l sequence-layer) x &key (train t))
  (with-slots (ls) l
    (let ((r ($execute (car ls) x :train train)))
      (loop :for e :in (cdr ls)
            :do (let ((nr ($execute e r :train train)))
                  (setf r nr)))
      r)))

(defclass affine-layer (layer)
  ((w :initform nil)
   (b :initform nil)
   (a :initform nil)))

(defun affine-layer (input-size output-size
                     &key (activation :sigmoid) (weight-initializer :xavier-uniform))
  (let ((n (make-instance 'affine-layer)))
    (with-slots (w b a) n
      (setf a (cond ((eq activation :sigmoid) #'$sigmoid)
                    ((eq activation :tanh) #'$tanh)
                    ((eq activation :relu) #'$relu)
                    ((eq activation :selu) #'$selu)
                    ((eq activation :swish) #'$swish)
                    ((eq activation :mish) #'$mish)
                    ((eq activation :softmax) #'$softmax)
                    ((eq activation :nil) nil)
                    (t #'$sigmoid)))
      (setf b ($parameter (zeros output-size)))
      (setf w (let ((sz (list input-size output-size)))
                (cond ((eq weight-initializer :random-uniform) (vru sz))
                      ((eq weight-initializer :random-normal) (vrn sz))
                      ((eq weight-initializer :random-normal-truncated) (vrnt sz))
                      ((eq weight-initializer :xavier-uniform) (vxavier sz :uniform))
                      ((eq weight-initializer :xavier-normal) (vxavier sz :normal))
                      ((eq weight-initializer :he-uniform) (vhe sz :uniform))
                      ((eq weight-initializer :he-normal) (vhe sz :normal))
                      ((eq weight-initializer :lecun-uniform) (vlecun sz :uniform))
                      ((eq weight-initializer :lecun-normal) (vlecun sz :normal))
                      ((eq weight-initializer :selu-uniform) (vselu sz :uniform))
                      ((eq weight-initializer :selu-normal) (vselu sz :normal))
                      (t (vru sz))))))
    n))

(defmethod $train-parameters ((l affine-layer))
  (with-slots (w b) l
    (list w b)))

(defmethod $execute ((l affine-layer) x &key (train t))
  (with-slots (w b a) l
    (if a
        (if train
            (funcall a ($affine x w b))
            (funcall a ($affine x ($data w) ($data b))))
        (if train
            ($affine x w b)
            ($affine x ($data w) ($data b))))))
