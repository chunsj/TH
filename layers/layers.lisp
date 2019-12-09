(declaim (optimize (speed 3) (debug 1) (safety 0)))

(defpackage :th.layers
  (:use #:common-lisp
        #:mu
        #:th)
  (:export #:$execute
           #:$parameters
           #:sequence-layer
           #:affine-layer
           #:batch-normalization-layer
           #:convolution-2d-layer
           #:maxpool-2d-layer
           #:avgpool-2d-layer
           #:flatten-layer))

(in-package :th.layers)

(defgeneric $execute (layer x &key trainp))

(defgeneric $train-parameters (layer))
(defgeneric $evaluation-parameters (layer))

(defclass layer () ())

(defmethod $train-parameters ((l layer)) nil)
(defmethod $evaluation-parameters ((l layer)) (mapcar #'$data ($train-parameters l)))

(defmethod $parameters ((l layer)) ($train-parameters l))

(defmethod $execute ((l layer) x &key (trainp t))
  (declare (ignore x trainp))
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

(defmethod $execute ((l sequence-layer) x &key (trainp t))
  (with-slots (ls) l
    (let ((r ($execute (car ls) x :trainp trainp)))
      (loop :for e :in (cdr ls)
            :do (let ((nr ($execute e r :trainp trainp)))
                  (setf r nr)))
      r)))

(defclass batch-normalization-layer (layer)
  ((g :initform nil)
   (e :initform nil)
   (rm :initform nil)
   (rv :initform nil)
   (sm :initform nil)
   (sd :initform nil)))

(defun batch-normalization-layer (input-size)
  (let ((n (make-instance 'batch-normalization-layer)))
    (with-slots (g e rm rv sm sd) n
      (setf g ($parameter (ones input-size))
            e ($parameter (zeros input-size))
            rm (zeros input-size)
            rv (ones input-size)
            sm (zeros input-size)
            sd (zeros input-size)))
    n))

(defmethod $train-parameters ((l batch-normalization-layer))
  (with-slots (g e) l
    (list g e)))

(defmethod $execute ((l batch-normalization-layer) x &key (trainp t))
  (with-slots (g e rm rv sm sd) l
    (if (and trainp (not (eq 1 ($ndim x))) (not (eq 3 ($ndim x))) (not (eq 1 ($size x 0))))
        ($bn x g e rm rv sm sd)
        (if (and (not (eq 1 ($ndim x))) (not (eq 3 ($ndim x))) (not (eq 1 ($size x 0))))
            ($bn (if ($parameterp x) ($data x) x) ($data g) ($data e) rm rv)
            ($bnorm (if ($parameterp x) ($data x) x) ($data g) ($data e) rm rv)))))

(defclass affine-layer (layer)
  ((w :initform nil)
   (b :initform nil)
   (a :initform nil)
   (bn :initform nil)))

(defun affine-layer (input-size output-size
                     &key (activation :sigmoid) (weight-initializer :he-normal)
                       batch-normalization-p)
  (let ((n (make-instance 'affine-layer)))
    (with-slots (w b a bn) n
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
                      (t (vru sz)))))
      (when batch-normalization-p
        (setf bn (batch-normalization-layer output-size))))
    n))

(defmethod $train-parameters ((l affine-layer))
  (with-slots (w b bn) l
    (if bn
        (append (list w b) ($train-parameters bn))
        (list w b))))

(defmethod $execute ((l affine-layer) x &key (trainp t))
  (with-slots (w b a bn ) l
    (if a
        (if trainp
            (if bn
                (funcall a ($execute bn ($affine x w b) :trainp trainp))
                (funcall a ($affine x w b)))
            (if bn
                (funcall a ($execute bn ($affine (if ($parameterp x) ($data x) x)
                                                 ($data w) ($data b)) :trainp trainp))
                (funcall a ($affine (if ($parameterp x) ($data x) x) ($data w) ($data b)))))
        (if trainp
            (if bn
                ($execute bn ($affine x w b) :trainp trainp)
                ($affine x w b))
            (if bn
                ($execute bn ($affine (if ($parameterp x) ($data x) x)
                                      ($data w) ($data b)) :trainp trainp)
                ($affine (if ($parameterp x) ($data x) x) ($data w) ($data b)))))))

(defclass convolution-2d-layer (layer)
  ((w :initform nil)
   (b :initform nil)
   (dw :initform 1)
   (dh :initform 1)
   (pw :initform 0)
   (ph :initform 0)
   (a :initform nil)
   (bn :initform nil)))

(defun convolution-2d-output-size (l x)
  (cond ((eq 4 ($ndim x))
         (let* ((sz ($size x))
                (nbatch ($ sz 0))
                (input-channel-size ($ sz 1))
                (input-height ($ sz 2))
                (input-width ($ sz 3)))
           (with-slots (w dw dh pw ph) l
             (when (eq input-channel-size ($size w 1))
               (let ((output-width (1+ (/ (+ input-width (- ($size w 3)) (* 2 pw)) dw)))
                     (output-height (1+ (/ (+ input-height (- ($size w 2)) (* 2 ph)) dh))))
                 (list nbatch ($size w 0) output-height output-width))))))
        ((eq 3 ($ndim x))
         (let* ((sz ($size x))
                (nbatch 1)
                (input-channel-size ($ sz 0))
                (input-height ($ sz 1))
                (input-width ($ sz 2)))
           (with-slots (w dw dh pw ph) l
             (when (eq input-channel-size ($size w 1))
               (let ((output-width (1+ (/ (+ input-width (- ($size w 3)) (* 2 pw)) dw)))
                     (output-height (1+ (/ (+ input-height (- ($size w 2)) (* 2 ph)) dh))))
                 (list nbatch ($size w 0) output-height output-width))))))))

(defun convolution-2d-layer (input-channel-size output-channel-size
                             filter-width filter-height
                             &key (stride-width 1) (stride-height 1)
                               (padding-width 0) (padding-height 0)
                               (activation :sigmoid) (weight-initializer :he-normal)
                               batch-normalization-p)
  (let ((n (make-instance 'convolution-2d-layer)))
    (with-slots (w b dw dh pw ph a bn) n
      (setf dw stride-width
            dh stride-height
            pw padding-width
            ph padding-height)
      (setf a (cond ((eq activation :sigmoid) #'$sigmoid)
                    ((eq activation :tanh) #'$tanh)
                    ((eq activation :relu) #'$relu)
                    ((eq activation :selu) #'$selu)
                    ((eq activation :swish) #'$swish)
                    ((eq activation :mish) #'$mish)
                    ((eq activation :softmax) #'$softmax)
                    ((eq activation :nil) nil)
                    (t #'$sigmoid)))
      (setf b ($parameter (zeros output-channel-size)))
      (setf w (let ((sz (list output-channel-size input-channel-size
                              filter-height filter-width)))
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
                      (t (vru sz)))))
      (when batch-normalization-p
        (setf bn (batch-normalization-layer output-channel-size))))
    n))

(defmethod $train-parameters ((l convolution-2d-layer))
  (with-slots (w b bn) l
    (if bn
        (append (list w b) ($train-parameters bn))
        (list w b))))

(defmethod $execute ((l convolution-2d-layer) x &key (trainp t))
  (with-slots (w b dw dh pw ph a) l
    (if a
        (if trainp
            (funcall a ($conv2d x w b dw dh pw ph))
            (funcall a ($conv2d (if ($parameterp x) ($data x) x) ($data w) ($data b)
                                dw dh pw ph)))
        (if trainp
            ($conv2d x w b dw dh pw ph)
            ($conv2d (if ($parameterp x) ($data x) x) ($data w) ($data b) dw dh pw ph)))))

(defclass maxpool-2d-layer (layer)
  ((kw :initform nil)
   (kh :initform nil)
   (dw :initform nil)
   (dh :initform nil)
   (pw :initform nil)
   (ph :initform nil)
   (ceil-p :initform nil)))

(defun maxpool-2d-layer (pool-width pool-height
                         &key (stride-width 1) (stride-height 1)
                           (padding-width 0) (padding-height 0)
                           ceilp)
  (let ((n (make-instance 'maxpool-2d-layer)))
    (with-slots (kw kh dw dh pw ph ceil-p) n
      (setf kw pool-width
            kh pool-height
            dw stride-width
            dh stride-height
            pw padding-width
            ph padding-height
            ceil-p ceilp))
    n))

(defmethod $execute ((l maxpool-2d-layer) x &key (trainp t))
  (with-slots (kw kh dw dh pw ph ceil-p) l
    (if trainp
        ($maxpool2d x kw kh dw dh pw ph ceil-p)
        ($maxpool2d (if ($parameterp x) ($data x) x) kw kh dw dh pw ph ceil-p))))

(defclass avgpool-2d-layer (layer)
  ((kw :initform nil)
   (kh :initform nil)
   (dw :initform nil)
   (dh :initform nil)
   (pw :initform nil)
   (ph :initform nil)
   (ceil-p :initform nil)
   (count-p :initform nil)))

(defun avgpool-2d-layer (pool-width pool-height
                         &key (stride-width 1) (stride-height 1)
                           (padding-width 0) (padding-height 0)
                           ceilp count-pad-p)
  (let ((n (make-instance 'avgpool-2d-layer)))
    (with-slots (kw kh dw dh pw ph ceil-p count-p) n
      (setf kw pool-width
            kh pool-height
            dw stride-width
            dh stride-height
            pw padding-width
            ph padding-height
            ceil-p ceilp
            count-p count-pad-p))
    n))

(defmethod $execute ((l avgpool-2d-layer) x &key (trainp t))
  (with-slots (kw kh dw dh pw ph ceil-p count-p) l
    (if trainp
        ($avgpool2d x kw kh dw dh pw ph ceil-p)
        ($avgpool2d (if ($parameterp x) ($data x) x) kw kh dw dh pw ph ceil-p count-p))))

(defclass flatten-layer (layer) ())

(defun flatten-layer () (make-instance 'flatten-layer))

(defmethod $execute ((l flatten-layer) x &key (trainp t))
  (let ((sz0 ($size x 0))
        (rsz (reduce #'* ($size x) :start 1)))
    (if trainp
        ($reshape x sz0 rsz)
        ($reshape (if ($parameterp x) ($data x) x) sz0 rsz))))
