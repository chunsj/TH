(declaim (optimize (speed 3) (debug 1) (safety 0)))

(defpackage :th.layers
  (:use #:common-lisp
        #:mu
        #:th)
  (:export #:layer
           #:$execute
           #:$evaluate
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
           #:$function-arguments
           #:droptout-layer
           #:rnn-cell
           #:lstm-cell
           #:gru-cell
           #:affine-cell
           #:attention-cell
           #:dropout-cell
           #:$keep-state!
           #:recurrent-layer
           #:$generate-sequence
           #:$cell-state
           #:$cell
           #:$update-cell-state!
           #:$set-memory!
           #:bidirectional-recurrent-layer
           #:$fcell
           #:$bcell
           #:with-keeping-state
           #:concat-sequence))

;; XXX cell state control api sucks!

(in-package :th.layers)

(defgeneric $execute (layer x &key trainp))
(defgeneric $evaluate (layer x))

(defgeneric $train-parameters (layer))

(defgeneric $keep-state! (layer statefulp &optional truncatedp))
(defgeneric $cell-state (layer))
(defgeneric $update-cell-state! (recurrent-layer h))
(defgeneric $set-memory! (cell hs))

(defclass layer () ())

(defmethod $train-parameters ((l layer)) nil)

(defmethod $keep-state! ((l layer) statefulp &optional (truncatedp T)) l)
(defmethod $cell-state ((l layer)) nil)
(defmethod $update-cell-state! ((l layer) h) l)
(defmethod $set-memory! ((l layer) hs) l)

(defmethod $parameters ((l layer)) ($train-parameters l))

(defmethod $execute ((l layer) x &key (trainp t))
  (declare (ignore x trainp))
  nil)

(defmethod $evaluate ((l layer) x) ($execute l x :trainp nil))

(defmethod $gd! ((l layer) &optional (learning-rate 0.01))
  ($gd! ($train-parameters l) learning-rate))

(defmethod $mgd! ((l layer) &optional (learning-rate 0.01) (momentum 0.9))
  ($mgd! ($train-parameters l) learning-rate momentum))

(defmethod $agd! ((l layer) &optional (learning-rate 0.01))
  ($agd! ($train-parameters l) learning-rate))

(defmethod $amgd! ((l layer) &optional (learning-rate 0.001) (β1 0.9) (β2 0.999))
  ($amgd! ($train-parameters l) learning-rate β1 β2))

(defmethod $rmgd! ((l layer) &optional (learning-rate 0.001) (decay-rate 0.99))
  ($rmgd! ($train-parameters l) learning-rate decay-rate))

(defmethod $adgd! ((l layer) &optional (learning-rate 1) (decay-rate 0.95))
  ($adgd! ($train-parameters l) learning-rate decay-rate))

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

(defmethod $keep-state! ((l sequential-layer) statefulp &optional (truncatedp T))
  (with-slots (ls) l
    (loop :for e :in ls
          :do ($keep-state! e statefulp truncatedp)))
  l)

(defmethod $parameters ((l sequential-layer))
  (with-slots (ls) l
    (loop :for e :in ls
          :appending ($parameters e))))

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

(defmethod $execute ((l batch-normalization-layer) x &key (trainp t))
  (with-slots (g e rm rv sm sd) l
    (if (and trainp (not (eq 1 ($ndim x))) (not (eq 3 ($ndim x))) (not (eq 1 ($size x 0))))
        ($bn x g e rm rv sm sd)
        (if (and (not (eq 1 ($ndim x))) (not (eq 3 ($ndim x))) (not (eq 1 ($size x 0))))
            ($bn x ($data g) ($data e) rm rv)
            ($bnorm x ($data g) ($data e) rm rv)))))

(defun afn (activation)
  (cond ((eq activation :sigmoid) #'$sigmoid)
        ((eq activation :tanh) #'$tanh)
        ((eq activation :relu) #'$relu)
        ((eq activation :lrelu) #'$lrelu)
        ((eq activation :selu) #'$selu)
        ((eq activation :swish) #'$swish)
        ((eq activation :mish) #'$mish)
        ((eq activation :gelu) #'$gelu)
        ((eq activation :celu) #'$celu)
        ((eq activation :softmax) #'$softmax)
        ((eq activation :logsoftmax) #'$logsoftmax)
        ((eq activation :nil) nil)
        (t #'$sigmoid)))

(defun wif (weight-initializer sz as &optional factor)
  (let ((w (cond ((eq weight-initializer :random-uniform) (apply #'vru (cons sz as)))
                 ((eq weight-initializer :random-normal) (apply #'vrn (cons sz as)))
                 ((eq weight-initializer :random-normal-truncated) (apply #'vrnt (cons sz as)))
                 ((eq weight-initializer :xavier-uniform) (vxavier sz :uniform))
                 ((eq weight-initializer :xavier-normal) (vxavier sz :normal))
                 ((eq weight-initializer :he-uniform) (vhe sz :uniform))
                 ((eq weight-initializer :he-normal) (vhe sz :normal))
                 ((eq weight-initializer :lecun-uniform) (vlecun sz :uniform))
                 ((eq weight-initializer :lecun-normal) (vlecun sz :normal))
                 ((eq weight-initializer :selu-uniform) (vselu sz :uniform))
                 ((eq weight-initializer :selu-normal) (vselu sz :normal))
                 (t (vru sz)))))
    (when factor ($mul! ($data w) factor))
    w))

(defclass affine-layer (layer)
  ((w :initform nil)
   (b :initform nil)
   (a :initform nil)
   (bn :initform nil)))

(defun affine-layer (input-size output-size
                     &key (activation :sigmoid) (weight-initializer :he-normal)
                       weight-initialization
                       weight-factor
                       batch-normalization-p (biasp t))
  (let ((n (make-instance 'affine-layer)))
    (with-slots (w b a bn wi) n
      (setf a (afn activation))
      (when biasp (setf b ($parameter (zeros output-size))))
      (setf w (wif weight-initializer (list input-size output-size) weight-initialization
                   weight-factor))
      (when batch-normalization-p
        (setf bn (batch-normalization-layer output-size))))
    n))

(defmethod $train-parameters ((l affine-layer))
  (with-slots (w b bn) l
    (if bn
        (append (if b (list w b) (list w)) ($train-parameters bn))
        (if b (list w b) (list w)))))

(defmethod $parameters ((l affine-layer))
  (with-slots (w b bn) l
    (if bn
        (append (if b (list w b) (list w)) ($parameters bn))
        (if b (list w b) (list w)))))

(defmethod $execute ((l affine-layer) x &key (trainp t))
  (with-slots (w b a bn) l
    (if a
        (if trainp
            (if bn
                (funcall a ($execute bn ($affine x w b)))
                (funcall a ($affine x w b)))
            (if bn
                (funcall a ($execute bn ($affine x ($data w) (when b ($data b)))
                                     :trainp trainp))
                (funcall a ($affine x ($data w) (when b ($data b))))))
        (if trainp
            (if bn
                ($execute bn ($affine x w b) :trainp trainp)
                ($affine x w b))
            (if bn
                ($execute bn ($affine x ($data w) (when b ($data b)))
                          :trainp trainp)
                ($affine x ($data w) (when b ($data b))))))))

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
                               weight-initialization
                               weight-factor
                               batch-normalization-p
                               (biasp t))
  (let ((n (make-instance 'convolution-2d-layer)))
    (with-slots (w b dw dh pw ph a bn wi) n
      (setf dw stride-width
            dh stride-height
            pw padding-width
            ph padding-height)
      (setf a (afn activation))
      (when biasp (setf b ($parameter (zeros output-channel-size))))
      (setf w (wif weight-initializer
                   (list output-channel-size input-channel-size filter-height filter-width)
                   weight-initialization weight-factor))
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

(defmethod $execute ((l convolution-2d-layer) x &key (trainp t))
  (with-slots (w b dw dh pw ph a bn) l
    (if a
        (if trainp
            (if bn
                (funcall a ($execute bn ($conv2d x w b dw dh pw ph)))
                (funcall a ($conv2d x w b dw dh pw ph)))
            (if bn
                (funcall a ($execute bn
                                     ($conv2d x ($data w)
                                              (when b ($data b))
                                              dw dh pw ph)
                                     :trainp nil))
                (funcall a ($conv2d x ($data w)
                                    (when b ($data b))
                                    dw dh pw ph))))
        (if trainp
            (if bn
                ($execute bn ($conv2d x w b dw dh pw ph))
                ($conv2d x w b dw dh pw ph))
            (if bn
                ($execute bn
                          ($conv2d x ($data w)
                                   (when b ($data b))
                                   dw dh pw ph)
                          :trainp nil)
                ($conv2d x ($data w)
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
  (declare (ignore trainp))
  (with-slots (kw kh dw dh pw ph ceil-p) l
    ($maxpool2d x kw kh dw dh pw ph ceil-p)))

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
  (declare (ignore trainp))
  (with-slots (kw kh dw dh pw ph ceil-p count-p) l
    ($avgpool2d x kw kh dw dh pw ph ceil-p)))

(defclass flatten-layer (layer) ())

(defun flatten-layer () (make-instance 'flatten-layer))

(defmethod $execute ((l flatten-layer) x &key (trainp t))
  (declare (ignore trainp))
  (let ((sz0 ($size x 0))
        (rsz (reduce #'* ($size x) :start 1)))
    ($reshape x sz0 rsz)))

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
   (bn :initform nil)))

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
                                    weight-initialization
                                    weight-factor
                                    batch-normalization-p
                                    (biasp t))
  (let ((n (make-instance 'full-convolution-2d-layer)))
    (with-slots (w b dw dh pw ph aw ah a bn wi) n
      (setf dw stride-width
            dh stride-height
            pw padding-width
            ph padding-height
            aw adjust-width
            ah adjust-height)
      (setf a (afn activation))
      (when biasp (setf b ($parameter (zeros output-channel-size))))
      (setf w (wif weight-initializer
                   (list input-channel-size output-channel-size filter-height filter-width)
                   weight-initialization weight-factor))
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

(defmethod $execute ((l full-convolution-2d-layer) x &key (trainp t))
  (with-slots (w b dw dh pw ph aw ah a bn) l
    (if a
        (if trainp
            (if bn
                (funcall a ($execute bn ($dconv2d x w b dw dh pw ph aw ah)))
                (funcall a ($dconv2d x w b dw dh pw ph aw ah)))
            (if bn
                (funcall a ($execute bn
                                     ($dconv2d x ($data w)
                                               (when b ($data b))
                                               dw dh pw ph aw ah)
                                     :trainp nil))
                (funcall a ($dconv2d x ($data w)
                                     (when b ($data b))
                                     dw dh pw ph aw ah))))
        (if trainp
            (if bn
                ($execute bn ($dconv2d x w b dw dh pw ph aw ah))
                ($dconv2d x w b dw dh pw ph aw ah))
            (if bn
                ($execute bn
                          ($dconv2d x ($data w)
                                    (when b ($data b))
                                    dw dh pw ph aw ah)
                          :trainp nil)
                ($dconv2d x ($data w)
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
  (declare (ignore trainp))
  (with-slots (rsizes) l
    (apply #'$reshape x (cons ($size x 0) rsizes))))

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

(defclass dropout-layer (layer)
  ((dop :initform (tensor '(0.1)))))

(defun dropout-layer (dropout-probability)
  (let ((n (make-instance 'dropout-layer)))
    (with-slots (dop) n
      (setf dop (tensor (list dropout-probability))))
    n))

(defmethod $parameters ((l dropout-layer))
  (with-slots (dop) l
    (list dop)))

(defmethod $execute ((l dropout-layer) x &key (trainp t))
  (with-slots (dop) l
    ($dropout x trainp ($ dop 0))))

(defclass dropout-cell (dropout-layer) ())

(defclass affine-cell (layer)
  ((wx :initform nil)
   (a :initform nil)
   (b :initform nil)))

(defun affine-cell (input-size output-size
                    &key (activation :sigmoid) (weight-initializer :he-normal)
                      weight-initialization weight-factor (biasp t))
  (let ((n (make-instance 'affine-cell)))
    (with-slots (wx b wi a) n
      (setf a (afn activation))
      (when biasp (setf b ($parameter (zeros output-size))))
      (setf wx (wif weight-initializer (list input-size output-size)
                    weight-initialization weight-factor)))

    n))

(defmethod $train-parameters ((l affine-cell))
  (with-slots (wx b) l
    (if b (list wx b) (list wx))))

(defmethod $parameters ((l affine-cell))
  (with-slots (wx b) l
    (if b (list wx b) (list wx))))

(defun embeddedp (x)
  (or (typep x 'tensor.long)
      (typep x 'tensor.int)))

(defun affine-cell-forward (x wx b)
  (if (embeddedp x)
      ($emb x wx b)
      ($affine x wx b)))

(defmethod $execute ((l affine-cell) x &key (trainp t))
  (with-slots (wx b a) l
    (let ((b0 (when b ($data b))))
      (if a
          (if trainp
              (funcall a (affine-cell-forward x wx b))
              (funcall a (affine-cell-forward x ($data wx) b0)))
          (if trainp
              (affine-cell-forward x wx b)
              (affine-cell-forward x ($data wx) b0))))))

(defun concat-sequence (seq)
  (let ((concat-args (append seq '(0)))
        (reshape-args (cons ($count seq) ($size (car seq)))))
    (apply #'$reshape (cons (apply #'$concat concat-args) reshape-args))))

(defun compute-dot-product-attention (hs q)
  "computes attention context from hs(TxBxD) and q(BxD)"
  (let* ((d ($size q 1))
         (q (-> (apply #'$reshape q (cons 1 ($size q)))
                ($transpose 0 1)))
         (k ($transpose hs 0 1))
         (kt ($transpose k 1 2))
         (qkt ($div ($bmm q kt) ($sqrt d)))
         (a (-> ($softmax ($reshape qkt ($size qkt 0) ($size qkt 2)))
                ($reshape ($size qkt 0) 1 ($size qkt 2))))
         (ctx (-> ($bmm a k)
                  ($reshape ($size k 0) ($size k 2)))))
    ctx))

(defclass attention-cell (layer)
  ((hs :initform nil :accessor $memory)
   (fn :initform #'compute-dot-product-attention :accessor $computer)))

(defun attention-cell (&key (computer :dot-product))
  (let ((n (make-instance 'attention-cell)))
    (with-slots (fn) n
      (cond ((eq computer :dot-product) (setf fn #'compute-dot-product-attention))
            (T (setf fn #'compute-dot-product-attention))))
    n))

(defmethod $set-memory! ((cell attention-cell) hs)
  (setf ($memory cell) hs)
  cell)

(defmethod $execute ((cell attention-cell) q &key (trainp t))
  (declare (ignore trainp))
  (with-slots (hs fn) cell
    (if (and hs fn)
        (funcall fn hs q)
        (error "no memory to compute"))))

(defclass rnn-cell (layer)
  ((wx :initform nil)
   (wh :initform nil)
   (a :initform nil)
   (bh :initform nil)
   (ph :initform nil)))

(defun rnn-cell (input-size output-size
                 &key (activation :tanh) (weight-initializer :he-normal)
                   weight-initialization weight-factor (biasp t))
  (let ((n (make-instance 'rnn-cell)))
    (with-slots (wx wh bh ph wi a) n
      (setf a (afn activation))
      (when biasp (setf bh ($parameter (zeros output-size))))
      (setf wx (wif weight-initializer (list input-size output-size)
                    weight-initialization weight-factor))
      (setf wh (wif weight-initializer (list output-size output-size)
                    weight-initialization weight-factor)))

    n))

(defmethod $keep-state! ((l rnn-cell) statefulp &optional (truncatedp T))
  (with-slots (ph) l
    (when ph
      (if statefulp
          (if ($parameterp ph)
              (if truncatedp
                  (setf ph ($clone ($data ph)))))
          (setf ph nil)))
    l))

(defmethod $cell-state ((l rnn-cell))
  (with-slots (ph) l
    ph))

(defmethod $update-cell-state! ((l rnn-cell) h)
  (with-slots (ph) l
    (setf ph h))
  l)

(defmethod $train-parameters ((l rnn-cell))
  (with-slots (wx wh bh) l
    (if bh (list wx wh bh) (list wx wh))))

(defmethod $parameters ((l rnn-cell))
  (with-slots (wx wh bh) l
    (if bh (list wx wh bh) (list wx wh))))

(defun embedding-forward (xi wx ph wh b)
  (let ((xp ($index wx 0 xi))
        (hp ($affine ph wh b)))
    ($+ xp hp)))

(defun rnn-cell-forward (x wx ph wh bh)
  (if (embeddedp x)
      (embedding-forward x wx ph wh bh)
      ($affine2 x wx ph wh bh)))

(defmethod $execute ((l rnn-cell) x &key (trainp t))
  (with-slots (wx wh bh ph a) l
    (let ((ph0 (if ph ph (zeros ($size x 0) ($size wx 1))))
          (bh0 (when bh ($data bh))))
      (let ((ph1 (if a
                     (if trainp
                         (funcall a (rnn-cell-forward x wx ph0 wh bh))
                         (funcall a (rnn-cell-forward x ($data wx) ph0 ($data wh) bh0)))
                     (if trainp
                         (rnn-cell-forward x wx ph0 wh bh)
                         (rnn-cell-forward x ($data wx) ph0 ($data wh) bh0)))))
        (setf ph ph1)
        ph1))))

;; lstm alternative implementation - faster
(defclass lstm-cell (layer)
  ((wx :initform nil)
   (wh :initform nil)
   (bh :initform nil)
   (ph :initform nil)
   (pc :initform nil)))

(defun lstm-cell (input-size output-size
                  &key (weight-initializer :he-normal)
                    (weight-initialization) weight-factor (biasp t))
  (let ((n (make-instance 'lstm-cell)))
    (with-slots (wx wh bh) n
      (when biasp
        (setf bh ($parameter (zeros (* 4 output-size)))))
      (setf wx (wif weight-initializer (list input-size (* 4 output-size))
                    weight-initialization weight-factor))
      (setf wh (wif weight-initializer (list output-size (* 4 output-size))
                    weight-initialization weight-factor))
      (when biasp
        ($fill! ($narrow ($data bh) 0 0 output-size) 1D0))
      )
    n))

(defmethod $keep-state! ((l lstm-cell) statefulp &optional (truncatedp T))
  (with-slots (ph pc) l
    (when (and ph pc)
      (if statefulp
          (if (and ($parameterp ph) ($parameterp pc))
              (if truncatedp
                  (setf ph ($clone ($data ph))
                        pc ($clone ($data pc)))))
          (setf ph nil
                pc nil)))
    l))

(defmethod $cell-state ((l lstm-cell))
  (with-slots (ph pc) l
    (list ph pc)))

(defmethod $update-cell-state! ((l lstm-cell) h)
  (with-slots (ph pc) l
    (setf ph (car h)
          pc (cadr h)))
  l)

(defmethod $train-parameters ((l lstm-cell))
  (with-slots (wx wh bh) l
    (if bh
        (list wx wh bh)
        (list wx wh))))

(defmethod $parameters ((l lstm-cell))
  (with-slots (wx wh bh) l
    (if bh
        (list wx wh bh)
        (list wx wh))))

(defmethod $execute ((l lstm-cell) x &key (trainp t))
  (with-slots (wx wh bh ph pc fspec ispec ospec aspec) l
    (let ((ph0 (if ph ph (zeros ($size x 0) ($size wh 0))))
          (pc0 (if pc pc (zeros ($size x 0) ($size wh 0))))
          (bh0 (when bh ($data bh)))
          (szf (/ ($size wx 1) 4)))
      (if trainp
          (let* ((ra (rnn-cell-forward x wx ph0 wh bh))
                 (ft ($sigmoid ($narrow ra 1 0 szf)))
                 (it ($sigmoid ($narrow ra 1 szf szf)))
                 (ot ($sigmoid ($narrow ra 1 (* 2 szf) szf)))
                 (at ($tanh ($narrow ra 1 (* 3 szf) szf)))
                 (ct ($+ ($* ft pc0) ($* at it)))
                 (ht ($* ot ($tanh ct))))
            (setf ph ht
                  pc ct)
            ht)
          (let* ((ra (rnn-cell-forward x ($data wx) ph0 ($data wh) bh0))
                 (ft ($sigmoid ($narrow ra 1 0 szf)))
                 (it ($sigmoid ($narrow ra 1 szf szf)))
                 (ot ($sigmoid ($narrow ra 1 (* 2 szf) szf)))
                 (at ($tanh ($narrow ra 1 (* 3 szf) szf)))
                 (ct ($+ ($* ft pc0) ($* at it)))
                 (ht ($* ot ($tanh ct))))
            (setf ph ht
                  pc ct)
            ht)))))

(defclass gru-cell (layer)
  ((wz :initform nil)
   (uz :initform nil)
   (bz :initform nil)
   (wr :initform nil)
   (ur :initform nil)
   (br :initform nil)
   (wh :initform nil)
   (uh :initform nil)
   (bh :initform nil)
   (ph :initform nil)))

(defun gru-cell (input-size output-size
                 &key (weight-initializer :he-normal)
                   weight-initialization weight-factor (biasp t))
  (let ((n (make-instance 'gru-cell)))
    (with-slots (wz uz bz wr ur br wh uh bh) n
      (when biasp
        (setf bz ($parameter (zeros output-size))
              br ($parameter ($* -1 (ones output-size)))
              bh ($parameter (zeros output-size))))
      (setf wz (wif weight-initializer (list input-size output-size)
                    weight-initialization weight-factor))
      (setf uz (wif weight-initializer (list output-size output-size)
                    weight-initialization weight-factor))
      (setf wr (wif weight-initializer (list input-size output-size)
                    weight-initialization weight-factor))
      (setf ur (wif weight-initializer (list output-size output-size)
                    weight-initialization weight-factor))
      (setf wh (wif weight-initializer (list input-size output-size)
                    weight-initialization weight-factor))
      (setf uh (wif weight-initializer (list output-size output-size)
                    weight-initialization weight-factor)))

    n))

(defmethod $keep-state! ((l gru-cell) statefulp &optional (truncatedp T))
  (with-slots (ph) l
    (when ph
      (if statefulp
          (if ($parameterp ph)
              (if truncatedp
                  (setf ph ($clone ($data ph)))))
          (setf ph nil)))
    l))

(defmethod $cell-state ((l gru-cell))
  (with-slots (ph) l
    ph))

(defmethod $update-cell-state! ((l gru-cell) h)
  (with-slots (ph) l
    (setf ph h))
  l)

(defmethod $train-parameters ((l gru-cell))
  (with-slots (wz uz bz wr ur br wh uh bh) l
    (if bz
        (list wz uz bz wr ur br wh uh bh)
        (list wz uz wr ur wh uh))))

(defmethod $parameters ((l gru-cell))
  (with-slots (wz uz bz wr ur br wh uh bh) l
    (if bz
        (list wz uz bz wr ur br wh uh bh)
        (list wz uz wr ur wh uh))))

(defmethod $execute ((l gru-cell) x &key (trainp t))
  (with-slots (wz uz bz wr ur br wh uh bh ph) l
    (let ((ph0 (if ph ph (zeros ($size x 0) ($size wz 1))))
          (bz0 (when bz ($data bz)))
          (br0 (when br ($data br)))
          (bh0 (when bh ($data bh))))
      (if trainp
          (let* ((zt ($sigmoid (rnn-cell-forward x wz ph0 uz bz)))
                 (rt ($sigmoid (rnn-cell-forward x wr ph0 ur br)))
                 (ht ($+ ($* zt ph0)
                         ($* ($- 1 zt)
                             ($tanh (rnn-cell-forward x wh
                                                      ($* rt ph0) uh bh))))))
            (setf ph ht)
            ht)
          (let* ((zt ($sigmoid (rnn-cell-forward x ($data wz) ph0 ($data uz) bz0)))
                 (rt ($sigmoid (rnn-cell-forward x ($data wr) ph0 ($data ur) br0)))
                 (ht ($+ ($* zt ph0)
                         ($* ($- 1 zt)
                             ($tanh (rnn-cell-forward x ($data wh)
                                                      ($* rt ph0) ($data uh) bh0))))))
            (setf ph ht)
            ht)))))

(defclass recurrent-layer (layer)
  ((stateful :initform nil)
   (truncated :initform nil)
   (cell :initform nil :accessor $cell)))

(defun recurrent-layer (cell &key statefulp truncatedp)
  (let ((n (make-instance 'recurrent-layer))
        (celli cell))
    (with-slots (stateful truncated cell) n
      (setf stateful statefulp)
      (setf truncated truncatedp)
      (setf cell celli))
    n))

(defmethod $train-parameters ((l recurrent-layer))
  (with-slots (cell) l
    ($train-parameters cell)))

(defmethod $keep-state! ((l recurrent-layer) statefulp &optional (truncatedp T))
  (with-slots (stateful truncated cell) l
    (setf stateful statefulp
          truncated truncatedp)
    ($keep-state! cell statefulp truncatedp))
  l)

(defmethod $parameters ((l recurrent-layer))
  (with-slots (cell) l
    ($parameters cell)))

(defmethod $execute ((l recurrent-layer) xs &key (trainp t))
  (with-slots (cell stateful truncated) l
    ($keep-state! cell stateful truncated)
    (loop :for x :in xs
          :collect ($execute cell x :trainp trainp))))

(defmethod $cell-state ((l recurrent-layer))
  ($cell-state ($cell l)))

(defmethod $update-cell-state! ((l recurrent-layer) h)
  ($update-cell-state! ($cell l) h)
  l)

(defmethod $set-memory! ((l recurrent-layer) hs)
  ($set-memory! ($cell l) hs)
  l)

(defclass bidirectional-recurrent-layer (layer)
  ((stateful :initform nil)
   (truncated :initform nil)
   (fcell :initform nil :accessor $fcell)
   (bcell :initform nil :accessor $bcell)))

(defun bidirectional-recurrent-layer (fcell bcell &key statefulp truncatedp)
  (let ((n (make-instance 'recurrent-layer))
        (fcelli fcell)
        (bcelli bcell))
    (with-slots (stateful truncated fcell bcell) n
      (setf stateful statefulp)
      (setf truncated truncatedp)
      (setf fcell fcelli)
      (setf bcell bcelli))
    n))

(defmethod $train-parameters ((l bidirectional-recurrent-layer))
  (with-slots (fcell bcell) l
    (append ($train-parameters fcell) ($train-parameters bcell))))

(defmethod $keep-state! ((l bidirectional-recurrent-layer) statefulp &optional (truncatedp T))
  (with-slots (stateful truncated fcell bcell) l
    (setf stateful statefulp
          truncated truncatedp)
    ($keep-state! fcell statefulp truncatedp)
    ($keep-state! bcell statefulp truncatedp))
  l)

(defmethod $parameters ((l bidirectional-recurrent-layer))
  (with-slots (fcell bcell) l
    (append ($parameters fcell) ($parameters bcell))))

(defmethod $execute ((l bidirectional-recurrent-layer) xs &key (trainp t))
  (with-slots (fcell bcell stateful truncated) l
    ($keep-state! fcell stateful truncated)
    ($keep-state! bcell stateful truncated)
    (let ((frs (loop :for x :in xs
                     :collect ($execute fcell x :trainp trainp)))
          (brs (loop :for x :in (reverse xs)
                     :collect ($execute bcell x :trainp trainp))))
      (mapcar (lambda (fr br) (list fr br)) frs brs))))

(defmethod $cell-state ((l bidirectional-recurrent-layer))
  (list ($cell-state ($fcell l)) ($cell-state ($bcell l))))

(defmethod $update-cell-state! ((l bidirectional-recurrent-layer) h)
  ($update-cell-state! ($fcell l) ($0 h))
  ($update-cell-state! ($bcell l) ($1 h))
  l)

(defmethod $set-memory! ((l bidirectional-recurrent-layer) hs)
  ($set-memory! ($fcell l) ($0 hs))
  ($set-memory! ($bcell l) ($1 hs))
  l)


(defmacro with-keeping-state ((rnn) &body body)
  `(progn
     ($keep-state! ,rnn T nil)
     (unwind-protect
          (progn ,@body)
       ($keep-state! ,rnn nil nil))))

(defgeneric $generate-sequence (rnn encoder seedseq n &optional temperature))
