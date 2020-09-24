(defpackage :mcmc-play
  (:use #:common-lisp
        #:mu
        #:th
        #:th.distributions))

(in-package :mcmc-play)

(defparameter *observations-a* ($sample (distribution/bernoulli 0.05) 1500))
(defparameter *observations-b* ($sample (distribution/bernoulli 0.04) 750))

(prn *observations-a*)
(prn ($sum *observations-a*))
(prn ($mean (tensor *observations-a*)))

(prn *observations-b*)
(prn ($sum *observations-b*))
(prn ($mean (tensor *observations-b*)))

(defun likelihood (dist data) ($ll dist data))

(defun mcmc (p0 data)
  (let ((nll (likelihood (distribution/bernoulli p0) data))
        (p p0)
        (accepted '())
        (rejected '())
        (n 20000)
        (nb 5000))
    (loop :repeat (+ n nb)
          :for i :from 1
          :for np = (random/uniform (max 0 (- p 0.1)) (min 0.99 (+ p 0.1)))
          :for nll-new = (likelihood (distribution/bernoulli np) data)
          :for lr = (log (random 1D0))
          :do (let ((dl (- nll-new nll)))
                (if (< lr dl)
                    (progn
                      (when (> i nb) (push np accepted))
                      (setf p np
                            nll nll-new))
                    (progn
                      (when (> i nb) (push np rejected))))))
    (list :accepted accepted
          :rejected rejected)))

(time
 (let ((dista (distribution/bernoulli ($parameter 0.5)))
       (distb (distribution/bernoulli ($parameter 0.5))))
   (loop :repeat 5000
         :for i :from 1
         :do (let ((nlla ($neg (likelihood dista *observations-a*)))
                   (nllb ($neg (likelihood distb *observations-b*))))
               (when (zerop (rem i 1000))
                 (prn i nlla nllb))
               ($amgd! ($parameters dista) 0.01)
               ($amgd! ($parameters distb) 0.01)))
   (let ((p0a ($scalar ($ dista :p)))
         (p0b ($scalar ($ distb :p))))
     (prn "P0:" p0a p0b)
     (let* ((mcmca (mcmc p0a *observations-a*))
            (mcmcb (mcmc p0b *observations-b*))
            (accepteda (getf mcmca :accepted))
            (acceptedb (getf mcmcb :accepted))
            (taa (tensor accepteda))
            (tab (tensor acceptedb)))
       (prn "MP:" ($mean taa) ($mean tab))
       (prn "SP:" ($sd taa) ($sd tab))
       (let ((na ($count taa))
             (nb ($count tab))
             (deltas nil))
         (setf deltas (tensor (loop :for i :from 0 :below (min na nb)
                                    :for pa = ($ taa i)
                                    :for pb = ($ tab i)
                                    :collect ( - pa pb))))
         (prn "ND:" na nb)
         (prn "MDP:" ($mean deltas))
         (prn "SDP:" ($sd deltas)))))))
