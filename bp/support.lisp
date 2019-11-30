(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defgeneric $bnorm (x gamma beta mean sd &optional trainp momentum eps))
(defgeneric $bn (x gamma beta rm rv &optional sm sd momentum eps))
(defgeneric $dropout (x &optional trainp p))

;; rmean and rvar as zeros and ones with dimensionality of input
;; if weight & bias given, they're traiable parameters (uniform/zero respectively)
;; if sm and sd are given, it's in training mode, if not, it's in evaluation mode
;; x should be in batch form
(defmethod $bn ((x tensor) (gamma tensor) (beta tensor) (rm tensor) (rv tensor)
                &optional sm sd (momentum 0.1) (eps 1E-5))
  (let ((output ($empty x))
        (n ($size x 1)))
    (if (and sm sd)
        (nn-batch-normalization-update-output x output gamma beta rm rv sm sd t momentum eps)
        (let ((sm (zeros n))
              (sd (ones n)))
          (nn-batch-normalization-update-output x output gamma beta rm rv sm sd nil momentum eps)))
    output))

(defmethod $bn ((x node) (gamma node) (beta node) (rm tensor) (rv tensor)
                &optional sm sd (momentum 0.1) (eps 1E-5))
  (node ($bn ($data x) ($data gamma) ($data beta) rm rv sm sd momentum eps)
        :name :bn
        :link (link (let* ((dx nil)
                           (dgamma nil)
                           (dbeta nil)
                           (gfn (lambda (dv gv)
                                  (declare (ignore dv))
                                  (unless (and dx dgamma dbeta)
                                    (setf dgamma ($zero ($data gamma))
                                          dbeta ($zero ($data beta))
                                          dx ($zero ($data x)))
                                    (if (and sm sd)
                                        (nn-batch-normalization-backward
                                         ($data x) gv dx
                                         dgamma dbeta gamma
                                         rm rv sm sd t 1 eps)
                                        (let ((sm (zeros ($size x 1)))
                                              (sd (ones ($size x 1))))
                                          (nn-batch-normalization-backward
                                           ($data x) gv dx
                                           dgamma dbeta gamma
                                           rm rv sm sd nil 1 eps)))))))
                      (to x (funcall gfn dv gv) dx)
                      (to gamma (funcall gfn dv gv) dgamma)
                      (to beta (funcall gfn dv gv) dbeta)))))

(defmethod $bn ((x tensor) (gamma node) (beta node) (rm tensor) (rv tensor)
                &optional sm sd (momentum 0.1) (eps 1E-5))
  (node ($bn x ($data gamma) ($data beta) rm rv sm sd momentum eps)
        :name :bn
        :link (link (let* ((dx nil)
                           (dgamma nil)
                           (dbeta nil)
                           (gfn (lambda (dv gv)
                                  (declare (ignore dv))
                                  (unless (and dx dgamma dbeta)
                                    (setf dgamma ($zero ($data gamma))
                                          dbeta ($zero ($data beta))
                                          dx ($zero x))
                                    (if (and sm sd)
                                        (nn-batch-normalization-backward
                                         x gv dx
                                         dgamma dbeta gamma
                                         rm rv sm sd t 1 eps)
                                        (let ((sm (zeros ($size x 1)))
                                              (sd (ones ($size x 1))))
                                          (nn-batch-normalization-backward
                                           x gv dx
                                           dgamma dbeta gamma
                                           rm rv sm sd nil 1 eps)))))))
                      (to gamma (funcall gfn dv gv) dgamma)
                      (to beta (funcall gfn dv gv) dbeta)))))

(defun runstat (x mean var trainp momentum)
  (let* ((x (if (eq 1 ($ndim x))
                (apply #'$reshape x (cons 1 ($size x)))
                x))
         (nx ($size x 0)))
    (when (and trainp (not (eq nx 1)))
      (let* ((mx ($mean x 0))
             (vx ($var x 0)))
        ($mul! mx momentum)
        ($mul! vx momentum)
        ($mul! mean (- 1 momentum))
        ($mul! var (- 1 momentum))
        ($add! mean mx)
        ($add! var vx)))))

(defmethod $bnorm ((x tensor) (gamma tensor) (beta tensor) (mean tensor) (var tensor)
                   &optional (trainp t) (momentum 0.1) (eps 1E-7))
  (runstat x mean var trainp momentum)
  (let* ((x (apply #'$reshape x (cons 1 ($size x))))
         (os (ones ($size x 0)))
         (zx ($div! ($sub x ($vv os mean)) ($sqrt! ($add var eps)))))
    ($add! ($mul! zx ($vv os gamma)) ($vv os beta))))

(defmethod $bnorm ((x tensor) (gamma null) (beta null) (mean tensor) (var tensor)
                   &optional (trainp t) (momentum 0.1) (eps 1E-7))
  (runstat x mean var trainp momentum)
  (let* ((x (apply #'$reshape x (cons 1 ($size x))))
         (os (ones ($size x 0)))
         (zx ($div! ($sub x ($vv os mean)) ($sqrt! ($add var eps)))))
    zx))

(defmethod $bnorm ((x node) (gamma node) (beta node) (mean node) (var node)
                   &optional (trainp t) (momentum 0.1) (eps 1E-7))
  (runstat ($data x) ($data mean) ($data var) trainp momentum)
  (let* ((x (if (eq 1 ($ndim x))
                ($vv (ones 1) x)
                x))
         (os (ones ($size x 0)))
         (zx ($div ($sub x ($vv os mean)) ($vv os ($sqrt ($add var eps))))))
    ($add ($mul zx ($vv os gamma)) ($vv os beta))))

(defmethod $bnorm ((x node) (gamma node) (beta node) (mean tensor) (var tensor)
                   &optional (trainp t) (momentum 0.1) (eps 1E-7))
  (runstat ($data x) mean var trainp momentum)
  (let* ((x (if (eq 1 ($ndim x))
                ($vv (ones 1) x)
                x))
         (os (ones ($size x 0)))
         (zx ($div ($sub x ($vv os mean)) ($vv os ($sqrt ($add var eps))))))
    ($add ($mul zx ($vv os gamma)) ($vv os beta))))

(defmethod $bnorm ((x node) (gamma null) (beta null) (mean node) (var node)
                   &optional (trainp t) (momentum 0.1) (eps 1E-7))
  (runstat ($data x) ($data mean) ($data var) trainp momentum)
  (let* ((x (if (eq 1 ($ndim x))
                ($vv (ones 1) x)
                x))
         (os (ones ($size x 0)))
         (zx ($div ($sub x ($vv os mean)) ($vv os ($sqrt ($add var eps))))))
    zx))

(defmethod $dropout ((x tensor) &optional (trainp t) (p 0.1))
  (if trainp
      (let ((mask ($mul! ($bernoulli! ($resize! ($empty x) ($size x)) (- 1 p)) (/ 1 (- 1 p)))))
        ($mul! mask x))
      x))

(defmethod $dropout ((x node) &optional (trainp t) (p 0.1))
  (if trainp
      (let* ((xd ($data x))
             (mask ($mul! ($bernoulli! ($resize! ($empty xd) ($size xd)) (- 1 p)) (/ 1 (- 1 p)))))
        (node ($mul mask xd)
              :name :dropout
              :link (link (to x ($mul mask gv)))))
      (node ($data p)
            :name :dropout
            :link (link (to x gv)))))

(defun $rnn (x ph wx wh b &optional ones)
  "Simple RNN cell using tanh"
  ($tanh ($affine2 x wx ph wh b ones)))

(defun $lstm (x ph pc wi ui wf uf wo uo wa ua bi bf bo ba &optional ones)
  "Basic LSTM cell"
  (let ((it ($sigmoid ($affine2 x wi ph ui bi ones)))
        (ft ($sigmoid ($affine2 x wf ph uf bf ones)))
        (ot ($sigmoid ($affine2 x wo ph uo bo ones)))
        (at ($tanh ($affine2 x wa ph ua ba ones))))
    (let ((ct ($addm2 at it ft pc)))
      (list ($mul ($tanh ct) ot) ct))))
