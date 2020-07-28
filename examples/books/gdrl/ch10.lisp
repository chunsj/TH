(defpackage :gdrl-ch10
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers
        #:th.env
        #:th.env.cartpole-regulator))

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

(defun sample-experiences (experiences nbatch)
  (let ((nr ($count experiences)))
    (if (> nr nbatch)
        (loop :repeat nbatch :collect ($ experiences (random nr)))
        experiences)))

(defun train-model (model-online model-target experiences &optional (gamma 0.95D0) (lr 0.003))
  (let ((nr ($count experiences)))
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
             (loss ($mse ys ts)))
        ($rmgd! model-online lr)
        ($data loss)))))

(defvar *max-buffer-size* 4096)
(defvar *batch-size* 512)
(defvar *max-epochs* 2000)
(defvar *sync-period* 15)
(defvar *eps0* 1D0)
(defvar *min-eps* 0.1D0)
(defvar *eps-decay-ratio* 0.9D0)

(defun report (epoch loss ntrain ctrain neval ceval success)
  (when (or success (zerop (rem epoch 20)))
    (let ((fmt "EPOCH ~4D | TRAIN ~3D / ~4,2F | EVAL ~4D / ~5,2F | TRAIN.LOSS ~,4F"))
      (prn (format nil fmt epoch ntrain ctrain neval ceval loss)))))

(defun sync-models (target online)
  ($cg! (list target online))
  (loop :for pt :in ($parameters target)
        :for po :in ($parameters online)
        :do ($set! ($data pt) ($data po))))

(defun generate-epsilons ()
  (decay-schedule *eps0* *min-eps* *eps-decay-ratio* *max-epochs*))

(defun duel-ddqn (&optional model)
  (let* ((train-env (cartpole-regulator-env :train))
         (eval-env (cartpole-regulator-env :eval))
         (model-target (model))
         (model-online (or model (model)))
         (experiences '())
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
                  (setf experiences (let ((ne ($count experiences)))
                                      (if (> ne *max-buffer-size*)
                                          (append (nthcdr (- ne *max-buffer-size*) experiences)
                                                  exs)
                                          (append experiences exs))))
                  (incf total-cost ctrain))
                (let* ((loss (train-model model-online model-target
                                          (sample-experiences experiences *batch-size*)
                                          0.95D0 0.008))
                       (eres (evaluate eval-env (best-action-selector model-online)))
                       (neval ($0 eres))
                       (ceval ($2 eres)))
                  (setf success ($1 eres))
                  (report epoch loss ntrain ctrain neval ceval success))
                (when (zerop (rem epoch *sync-period*))
                  (sync-models model-target model-online))))
    (when success
      (prn (format nil "*** TOTAL ~6D / ~4,2F" ($count experiences) total-cost)))
    model-online))

(defparameter *m* nil)

(setf *m* (duel-ddqn *m*))

(let ((env (cartpole-regulator-env :eval)))
  (evaluate env (best-action-selector *m*)))
