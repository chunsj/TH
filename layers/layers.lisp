(declaim (optimize (speed 3) (debug 1) (safety 0)))

(defpackage :th.layers
  (:use #:common-lisp
        #:mu
        #:th)
  (:export #:$initialize
           #:$execute
           #:$parameters
           #:$save-weights
           #:$load-weights
           #:sequential-layer
           #:parallel-layer
           #:affine-layer
           #:batch-normalization-layer
           #:convolution-2d-layer
           #:maxpool-2d-layer
           #:avgpool-2d-layer
           #:flatten-layer
           #:full-convolution-2d-layer
           #:reshape-layer
           #:functional-layer
           #:$function-arguments))

(in-package :th.layers)

(defgeneric $initialize (layer))
(defgeneric $execute (layer x &key trainp))

(defgeneric $train-parameters (layer))

(defclass layer () ())

(defmethod $train-parameters ((l layer)) nil)

(defmethod $parameters ((l layer)) ($train-parameters l))

(defmethod $initialize ((l layer)))

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
(defmethod $reset! ((l layer)) ($reset! ($train-parameters l)))

(defun $save-weights (filename network)
  (ensure-directories-exist (strcat filename "/"))
  (loop :for p :in ($parameters network)
        :for i :from 0
        :for tensor = (if ($parameterp p) ($data p) p)
        :do (let* ((tfn (strcat filename "/" (format nil "~A" i) ".dat"))
                   (f (file.disk tfn "w")))
              (setf ($fbinaryp f) t)
              ($fwrite tensor f)
              ($fclose f))))

(defun $load-weights (filename network)
  (loop :for p :in ($parameters network)
        :for i :from 0
        :do (let* ((tfn (strcat filename "/" (format nil "~A" i) ".dat"))
                   (f (file.disk tfn "r")))
              (setf ($fbinaryp f) t)
              ($fread (if ($parameterp p) ($data p) p) f)
              ($fclose f))))

(defclass sequential-layer (layer)
  ((ls :initform nil)))

(defmethod $ ((l sequential-layer) location &rest others-and-default)
  (declare (ignore others-and-default))
  (with-slots (ls) l
    ($ ls location)))

(defun sequential-layer (&rest layers)
  (let ((n (make-instance 'sequential-layer)))
    (with-slots (ls) n
      (setf ls layers))
    n))

(defmethod $train-parameters ((l sequential-layer))
  (with-slots (ls) l
    (loop :for e :in ls
          :appending ($train-parameters e))))

(defmethod $parameters ((l sequential-layer))
  (with-slots (ls) l
    (loop :for e :in ls
          :appending ($parameters e))))

(defmethod $initialize ((l sequential-layer))
  (with-slots (ls) l
    (loop :for e :in ls :do ($initialize e))))

(defmethod $execute ((l sequential-layer) x &key (trainp t))
  (with-slots (ls) l
    (let ((r ($execute (car ls) x :trainp trainp)))
      (loop :for e :in (cdr ls)
            :do (let ((nr ($execute e r :trainp trainp)))
                  (setf r nr)))
      r)))

(defclass parallel-layer (sequential-layer) ())

(defun parallel-layer (&rest layers)
  (let ((n (make-instance 'parallel-layer)))
    (with-slots (ls) n
      (setf ls layers))
    n))

(defmethod $execute ((l parallel-layer) x &key (trainp t))
  (with-slots (ls) l
    (mapcar (lambda (l) ($execute l x :trainp trainp)) ls)))

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

(defmethod $parameters ((l batch-normalization-layer))
  (with-slots (g e rm rv) l
    (list g e rm rv)))

(defmethod $initialize ((l batch-normalization-layer))
  (with-slots (g e rm rv sm sd) l
    (let ((input-size ($size g 0)))
      (setf g ($parameter (ones input-size))
            e ($parameter (zeros input-size))
            rm (zeros input-size)
            rv (ones input-size)
            sm (zeros input-size)
            sd (zeros input-size)))))

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
   (bn :initform nil)
   (os :initform #{})
   (wi :initform nil)))

(defun affine-layer (input-size output-size
                     &key (activation :sigmoid) (weight-initializer :he-normal)
                       batch-normalization-p (biasp t))
  (let ((n (make-instance 'affine-layer)))
    (with-slots (w b a bn wi) n
      (setf wi weight-initializer)
      (setf a (cond ((eq activation :sigmoid) #'$sigmoid)
                    ((eq activation :tanh) #'$tanh)
                    ((eq activation :relu) #'$relu)
                    ((eq activation :lrelu) #'$lrelu)
                    ((eq activation :selu) #'$selu)
                    ((eq activation :swish) #'$swish)
                    ((eq activation :mish) #'$mish)
                    ((eq activation :softmax) #'$softmax)
                    ((eq activation :nil) nil)
                    (t #'$sigmoid)))
      (when biasp (setf b ($parameter (zeros output-size))))
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
        (append (if b (list w b) (list w)) ($train-parameters bn))
        (if (list w b) (list w)))))

(defmethod $parameters ((l affine-layer))
  (with-slots (w b bn) l
    (if bn
        (append (if b (list w b) (list w)) ($parameters bn))
        (if b (list w b) (list w)))))

(defun affine-ones (l x)
  (when (eq 2 ($ndim x))
    (with-slots (os) l
      (let* ((n ($size x 0))
             (o ($ os n)))
        (unless o
          (setf ($ os n) (ones n))
          (setf o ($ os n)))
        o))))

(defmethod $initialize ((l affine-layer))
  (with-slots (w b bn wi) l
    (let ((weight-initializer wi)
          (input-size ($size w 0))
          (output-size ($size w 1)))
      (when b (setf b ($parameter (zeros output-size))))
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
      (when bn ($initialize bn)))))

(defmethod $execute ((l affine-layer) x &key (trainp t))
  (with-slots (w b a bn) l
    (if a
        (if trainp
            (if bn
                (funcall a ($execute bn ($affine x w b (affine-ones l x))))
                (funcall a ($affine x w b (affine-ones l x))))
            (if bn
                (funcall a ($execute bn ($affine (if ($parameterp x) ($data x) x)
                                                 ($data w) (when b ($data b))
                                                 (when b (affine-ones l x)))
                                     :trainp trainp))
                (funcall a ($affine (if ($parameterp x) ($data x) x) ($data w)
                                    (when b ($data b))
                                    (when b (affine-ones l x))))))
        (if trainp
            (if bn
                ($execute bn ($affine x w b (affine-ones l x)) :trainp trainp)
                ($affine x w b))
            (if bn
                ($execute bn ($affine (if ($parameterp x) ($data x) x)
                                      ($data w) (when  b ($data b)) (when b (affine-ones l x)))
                          :trainp trainp)
                ($affine (if ($parameterp x) ($data x) x) ($data w) (when b ($data b))
                         (when b (affine-ones l x))))))))

(defclass convolution-2d-layer (layer)
  ((w :initform nil)
   (b :initform nil)
   (dw :initform 1)
   (dh :initform 1)
   (pw :initform 0)
   (ph :initform 0)
   (a :initform nil)
   (bn :initform nil)
   (wi :initform nil)))

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
                               batch-normalization-p
                               (biasp t))
  (let ((n (make-instance 'convolution-2d-layer)))
    (with-slots (w b dw dh pw ph a bn wi) n
      (setf wi weight-initializer)
      (setf dw stride-width
            dh stride-height
            pw padding-width
            ph padding-height)
      (setf a (cond ((eq activation :sigmoid) #'$sigmoid)
                    ((eq activation :tanh) #'$tanh)
                    ((eq activation :relu) #'$relu)
                    ((eq activation :lrelu) #'$lrelu)
                    ((eq activation :selu) #'$selu)
                    ((eq activation :swish) #'$swish)
                    ((eq activation :mish) #'$mish)
                    ((eq activation :softmax) #'$softmax)
                    ((eq activation :nil) nil)
                    (t #'$sigmoid)))
      (when biasp (setf b ($parameter (zeros output-channel-size))))
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
        (append (if b (list w b) (list w)) ($train-parameters bn))
        (if b (list w b) (list w)))))

(defmethod $parameters ((l convolution-2d-layer))
  (with-slots (w b bn) l
    (if bn
        (append (if b (list w b) (list w)) ($parameters bn))
        (if b (list w b) (list w)))))

(defmethod $initialize ((l convolution-2d-layer))
  (with-slots (w b bn wi) l
    (let ((weight-initializer wi)
          (output-channel-size ($size w 0))
          (input-channel-size ($size w 1))
          (filter-height ($size w 2))
          (filter-width ($size w 3)))
      (when b (setf b ($parameter (zeros output-channel-size))))
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
      (when bn ($initialize bn)))))

(defmethod $execute ((l convolution-2d-layer) x &key (trainp t))
  (with-slots (w b dw dh pw ph a bn) l
    (if a
        (if trainp
            (if bn
                (funcall a ($execute bn ($conv2d x w b dw dh pw ph)))
                (funcall a ($conv2d x w b dw dh pw ph)))
            (if bn
                (funcall a ($execute bn
                                     ($conv2d (if ($parameterp x) ($data x) x) ($data w)
                                              (when b ($data b))
                                              dw dh pw ph)
                                     :trainp nil))
                (funcall a ($conv2d (if ($parameterp x) ($data x) x) ($data w)
                                    (when b ($data b))
                                    dw dh pw ph))))
        (if trainp
            (if bn
                ($execute bn ($conv2d x w b dw dh pw ph))
                ($conv2d x w b dw dh pw ph))
            (if bn
                ($execute bn
                          ($conv2d (if ($parameterp x) ($data x) x) ($data w)
                                   (when b ($data b))
                                   dw dh pw ph)
                          :trainp nil)
                ($conv2d (if ($parameterp x) ($data x) x) ($data w)
                         (when b ($data b))
                         dw dh pw ph))))))

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

(defclass full-convolution-2d-layer (layer)
  ((w :initform nil)
   (b :initform nil)
   (dw :initform 1)
   (dh :initform 1)
   (pw :initform 0)
   (ph :initform 0)
   (aw :initform 0)
   (ah :initform 0)
   (a :initform nil)
   (bn :initform nil)
   (wi :initform nil)))

(defun full-convolution-2d-output-size (l x)
  (cond ((eq 4 ($ndim x))
         (let* ((sz ($size x))
                (nbatch ($ sz 0))
                (input-channel-size ($ sz 1))
                (input-height ($ sz 2))
                (input-width ($ sz 3)))
           (with-slots (w dw dh pw ph aw ah) l
             (when (eq input-channel-size ($size w 1))
               (let ((output-w (+ (* (- input-width 1) dw) (* -2 pw) ($size w 3) aw))
                     (output-h (+ (* (- input-height 1) dh) (* -2 ph) ($size w 2) ah)))
                 (list nbatch ($size w 0) output-h output-w))))))
        ((eq 3 ($ndim x))
         (let* ((sz ($size x))
                (nbatch 1)
                (input-channel-size ($ sz 0))
                (input-height ($ sz 1))
                (input-width ($ sz 2)))
           (with-slots (w dw dh pw ph aw ah) l
             (when (eq input-channel-size ($size w 1))
               (let ((output-w (+ (* (- input-width 1) dw) (* -2 pw) ($size w 3) aw))
                     (output-h (+ (* (- input-height 1) dh) (* -2 ph) ($size w 2) ah)))
                 (list nbatch ($size w 0) output-h output-w))))))))

(defun full-convolution-2d-layer (input-channel-size output-channel-size
                                  filter-width filter-height
                                  &key (stride-width 1) (stride-height 1)
                                    (padding-width 0) (padding-height 0)
                                    (adjust-width 0) (adjust-height 0)
                                    (activation :sigmoid) (weight-initializer :he-normal)
                                    batch-normalization-p
                                    (biasp t))
  (let ((n (make-instance 'full-convolution-2d-layer)))
    (with-slots (w b dw dh pw ph aw ah a bn wi) n
      (setf wi weight-initializer)
      (setf dw stride-width
            dh stride-height
            pw padding-width
            ph padding-height
            aw adjust-width
            ah adjust-height)
      (setf a (cond ((eq activation :sigmoid) #'$sigmoid)
                    ((eq activation :tanh) #'$tanh)
                    ((eq activation :relu) #'$relu)
                    ((eq activation :lrelu) #'$lrelu)
                    ((eq activation :selu) #'$selu)
                    ((eq activation :swish) #'$swish)
                    ((eq activation :mish) #'$mish)
                    ((eq activation :softmax) #'$softmax)
                    ((eq activation :nil) nil)
                    (t #'$sigmoid)))
      (when biasp (setf b ($parameter (zeros output-channel-size))))
      (setf w (let ((sz (list input-channel-size output-channel-size
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

(defmethod $train-parameters ((l full-convolution-2d-layer))
  (with-slots (w b bn) l
    (if bn
        (append (if b (list w b) (list w)) ($train-parameters bn))
        (if b (list w b) (list w)))))

(defmethod $parameters ((l full-convolution-2d-layer))
  (with-slots (w b bn) l
    (if bn
        (append (if b (list w b) (list w)) ($parameters bn))
        (if b (list w b) (list w)))))

(defmethod $initialize ((l full-convolution-2d-layer))
  (with-slots (w b bn wi) l
    (let ((weight-initializer wi)
          (output-channel-size ($size w 1))
          (input-channel-size ($size w 0))
          (filter-height ($size w 2))
          (filter-width ($size w 3)))
      (when b (setf b ($parameter (zeros output-channel-size))))
      (setf w (let ((sz (list input-channel-size output-channel-size
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
      (when bn ($initialize bn)))))

(defmethod $execute ((l full-convolution-2d-layer) x &key (trainp t))
  (with-slots (w b dw dh pw ph aw ah a bn) l
    (if a
        (if trainp
            (if bn
                (funcall a ($execute bn ($dconv2d x w b dw dh pw ph aw ah)))
                (funcall a ($dconv2d x w b dw dh pw ph aw ah)))
            (if bn
                (funcall a ($execute bn
                                     ($dconv2d (if ($parameterp x) ($data x) x) ($data w)
                                               (when b ($data b))
                                               dw dh pw ph aw ah)
                                     :trainp nil))
                (funcall a ($dconv2d (if ($parameterp x) ($data x) x) ($data w)
                                     (when b ($data b))
                                     dw dh pw ph aw ah))))
        (if trainp
            (if bn
                ($execute bn ($dconv2d x w b dw dh pw ph aw ah))
                ($dconv2d x w b dw dh pw ph aw ah))
            (if bn
                ($execute bn
                          ($dconv2d (if ($parameterp x) ($data x) x) ($data w)
                                    (when b ($data b))
                                    dw dh pw ph aw ah)
                          :trainp nil)
                ($dconv2d (if ($parameterp x) ($data x) x) ($data w)
                          (when b ($data b))
                          dw dh pw ph aw ah))))))

(defclass reshape-layer (layer)
  ((rsizes :initform nil)))

(defun reshape-layer (&rest sizes)
  (let ((n (make-instance 'reshape-layer)))
    (with-slots (rsizes) n
      (setf rsizes sizes))
    n))

(defmethod $execute ((l reshape-layer) x &key (trainp t))
  (with-slots (rsizes) l
    (if trainp
        (apply #'$reshape x (cons ($size x 0) rsizes))
        (apply #'$reshape (if ($parameterp x) ($data x) x) (cons ($size x 0) rsizes)))))

(defclass functional-layer (layer)
  ((f :initform nil)
   (args :initform nil :reader $function-arguments)))

(defun functional-layer (function)
  (let ((n (make-instance 'functional-layer)))
    (with-slots (f) n
      (setf f function))
    n))

(defmethod $execute ((l functional-layer) x &key (trainp t))
  (with-slots (f args) l
    (setf args x)
    (if f
        (cond ((listp x) (apply f (append x (list :trainp trainp))))
              (t (apply f (list x :trainp trainp))))
        x)))
