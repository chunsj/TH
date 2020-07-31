(defpackage :gdrl-ch10
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers
        #:th.env
        #:th.env.cartpole))

(in-package :gdrl-ch10)

(defun decay-schedule (v0 minv decay-ratio max-steps &key (log-start -2) (log-base 10))
  (let* ((decay-steps (round (* max-steps decay-ratio)))
         (rem-steps (- max-steps decay-steps))
         (vs (-> ($/ (logspace log-start 0 decay-steps) (log log-base 10))
                 ($list)
                 (reverse)
                 (tensor)))
         (minvs ($min vs))
         (maxvs ($max vs))
         (rngv (- maxvs minvs))
         (vs ($/ ($- vs minvs) rngv))
         (vs ($+ minv ($* vs (- v0 minv)))))
    ($cat vs ($fill! (tensor rem-steps) ($last vs)))))

(defun model-common (&optional (ni 4))
  (let ((h1 5)
        (h2 5))
    (sequential-layer
     (affine-layer ni h1 :weight-initializer :random-uniform)
     (affine-layer h1 h2 :weight-initializer :random-uniform))))

(defun model-value (&optional (ni 5))
  (sequential-layer
   (affine-layer ni 1 :weight-initializer :random-uniform)))

(defun model-advantage (&optional (ni 5) (no 2))
  (sequential-layer
   (affine-layer ni no :weight-initializer :random-uniform)))

(defclass duel-ddqn-model (layer)
  ((cm :initform (model-common))
   (vm :initform (model-value))
   (am :initform (model-advantage))))

(defun model () (make-instance 'duel-ddqn-model))

(defmethod $execute ((m duel-ddqn-model) x &key (trainp T))
  (with-slots (cm vm am) m
    (let* ((hc ($execute cm x :trainp trainp))
           (hv ($execute vm hc :trainp trainp))
           (ha ($execute am hc :trainp trainp))
           (sa ($size ha))
           (ma ($mean ha 1)))
      ($add ($expand hv sa) ($sub ha ($expand ma sa))))))

(defmethod $train-parameters ((m duel-ddqn-model))
  (with-slots (cm vm am) m
    (append ($train-parameters cm)
            ($train-parameters vm)
            ($train-parameters am))))

(defun best-action-selector (model &optional (epsilon 0))
  (lambda (state)
    (if (> (random 1D0) epsilon)
        (let* ((state ($reshape state 1 4))
               (q ($evaluate model state)))
          ($scalar ($argmin q 1)))
        (random 2))))

(defclass replay-buffer ()
  ((entries :initform nil)
   (deltas :initform nil)
   (nsz :initform 0)
   (idx :initform -1)
   (alpha :initform 0.6)
   (beta :initform 0.1)
   (beta-rate :initform 0.99992)))

(defun replay-buffer (size)
  (let ((n (make-instance 'replay-buffer)))
    (with-slots (entries deltas nsz idx) n
      (setf entries (make-array size :initial-element nil)
            deltas ($fill! (tensor size) 1D0)
            nsz 0
            idx 0))
    n))

(defun add-sample (buffer sample)
  (with-slots (idx nsz entries deltas) buffer
    (let ((maxsz ($count entries))
          (maxd ($max deltas)))
      (setf ($ entries idx) sample
            ($ deltas idx) maxd)
      (setf nsz (min (1+ nsz) maxsz))
      (incf idx)
      (setf idx (rem idx maxsz)))
    buffer))

(defun update-deltas (buffer idcs tderrs)
  (with-slots (entries deltas) buffer
    (setf ($index deltas 0 idcs) ($abs ($reshape tderrs ($count tderrs))))))

(defun update-beta! (buffer)
  (with-slots (beta beta0 beta-rate) buffer
    (setf beta (min 1D0 (/ beta beta-rate)))))

(defconstant +eps+ 1E-6)

(defun sample-experiences (buffer nbatch)
  (with-slots (entries nsz deltas alpha beta) buffer
    (if (>= nsz nbatch)
        (let* ((prs ($expt ($+ ($subview deltas 0 nsz) +eps+) alpha))
               (pbs ($/ prs ($sum prs)))
               (wts ($expt ($* pbs nsz) beta))
               (nwts ($/ wts ($max wts)))
               (indices (tensor.long (loop :for i :from 0 :below nsz :collect i)))
               (idcs (loop :repeat nbatch :collect ($choice indices pbs))))
          (update-beta! buffer)
          (list idcs
                ($reshape ($index nwts 0 idcs) nbatch 1)
                (loop :for i :in idcs :collect ($ entries i))))
        (list (loop :for i :from 0 :below nsz :collect i)
              ($reshape (tensor (loop :repeat nsz :collect 1)) nsz 1)
              (loop :for i :from 0 :below nsz :collect ($ entries i))))))

(defvar *max-buffer-size* 4096)
(defvar *batch-size* 512)
(defvar *max-epochs* 1000)
(defvar *eps0* 1D0)
(defvar *min-eps* 0.1D0)
(defvar *eps-decay-ratio* 0.9D0)

(defun train-model (model-online model-target buffer &optional (gamma 0.95D0) (lr 0.003))
  (let* ((experiences0 (sample-experiences buffer *batch-size*))
         (indices ($ experiences0 0))
         (nweights ($ experiences0 1))
         (experiences ($ experiences0 2))
         (nr ($count experiences)))
    (let ((states (-> (apply #'$concat (mapcar #'$0 experiences))
                      ($reshape! nr 4)))
          (actions (-> (tensor.long (mapcar #'$1 experiences))
                       ($reshape! nr 1)))
          (costs (-> (tensor (mapcar #'$2 experiences))
                     ($reshape! nr 1)))
          (next-states (-> (apply #'$concat (mapcar #'$3 experiences))
                           ($reshape! nr 4)))
          (dones (-> (tensor (mapcar (lambda (e) (if ($4 e) 1 0)) experiences))
                     ($reshape! nr 1))))
      (let* ((argmins (-> ($evaluate model-online next-states)
                          ($argmin 1)))
             (qns (-> ($evaluate model-target next-states)
                      ($gather 1 argmins)))
             (xs states)
             (ts ($+ costs ($* gamma qns ($- 1 dones))))
             (ys (-> ($execute model-online xs)
                     ($gather 1 actions)))
             (tderrs ($- ys ts))
             (loss ($mean ($square ($* nweights tderrs)))))
        ($rmgd! model-online lr)
        (update-deltas buffer indices ($data tderrs))
        ($data loss)))))

(defun report (epoch loss ntrain ctrain neval ceval success)
  (when (or success (zerop (rem epoch 20)))
    (let ((fmt "EPOCH ~4D | TRAIN ~3D / ~4,2F | EVAL ~4D / ~5,2F | TRAIN.LOSS ~,4F"))
      (prn (format nil fmt epoch ntrain ctrain neval ceval loss)))))

(defun polyak-averaging (target online &optional (tau 0.1D0))
  ($cg! (list target online))
  (loop :for pt :in ($parameters target)
        :for po :in ($parameters online)
        :for a = ($* tau ($data po))
        :for b = ($* (- 1 tau) ($data pt))
        :do ($set! ($data pt) ($+ a b))))

(defun sync-models (target online)
  (polyak-averaging target online))

(defun generate-epsilons ()
  (decay-schedule *eps0* *min-eps* *eps-decay-ratio* *max-epochs*))

(defun duel-ddqn (&optional model)
  (let* ((train-env (cartpole-env :train))
         (eval-env (cartpole-env :eval))
         (model-target (model))
         (model-online (or model (model)))
         (buffer (replay-buffer *max-buffer-size*))
         (excount 0)
         (total-cost 0)
         (success nil)
         (epsilons (generate-epsilons)))
    (sync-models model-target model-online)
    (loop :for epoch :from 1 :to *max-epochs*
          :while (not success)
          :for eps = ($ epsilons (1- epoch))
          :do (let ((ctrain 0)
                    (ntrain 0))
                (let* ((exsi (collect-experiences train-env
                                                  (best-action-selector model-online eps)))
                       (exs (car exsi)))
                  (setf ctrain (cadr exsi))
                  (setf ntrain ($count exs))
                  (incf excount ntrain)
                  (loop :for e :in exs :do (add-sample buffer e))
                  (incf total-cost ctrain))
                (let* ((loss (train-model model-online model-target
                                          buffer
                                          0.95D0 0.008))
                       (eres (evaluate eval-env (best-action-selector model-online)))
                       (neval ($0 eres))
                       (ceval ($2 eres)))
                  (setf success ($1 eres))
                  (report epoch loss ntrain ctrain neval ceval success))
                (sync-models model-target model-online)))
    (when success
      (prn (format nil "*** TOTAL ~6D / ~4,2F" excount total-cost)))
    model-online))

(defparameter *m* nil)

(setf *m* (duel-ddqn *m*))

(let ((env (cartpole-env :eval)))
  (evaluate env (best-action-selector *m*)))
