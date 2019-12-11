;; from
;; https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f

(defpackage :gan-simple
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :gan-simple)

;; this simple - in terms of execution speed and complexity - gan example fits gaussian
;; random distribution defined by following parameters
(defparameter *data-mean* 4)
(defparameter *data-sd* 1.25)

(defparameter *g-input-size* 1)
(defparameter *g-hidden-size* 50)
(defparameter *g-output-size* 1)

(defparameter *d-input-size* 100)
(defparameter *d-hidden-size* 50)
(defparameter *d-output-size* 1)

(defparameter *minibatch-size* *d-input-size*)

(defparameter *d-learning-rate* 2E-4)
(defparameter *g-learning-rate* 2E-4)
(defparameter *beta1* 0.9)
(defparameter *beta2* 0.999)

(defparameter *num-epochs* 30000)
(defparameter *print-interval* 500)
(defparameter *d-steps* 1)
(defparameter *g-steps* 1)

(defun get-distribution-sampler (mu sigma)
  (lambda (n)
    ($+ mu ($* (rndn 1 n) sigma))))

(defun get-generator-input-sampler ()
  (lambda (m n) (rnd m n)))

(defparameter *generator* (parameters))
(defparameter *gw1* ($push *generator* (vxavier (list *g-input-size* *g-hidden-size*))))
(defparameter *gb1* ($push *generator* (zeros *g-hidden-size*)))
(defparameter *gw2* ($push *generator* (vxavier (list *g-hidden-size* *g-hidden-size*))))
(defparameter *gb2* ($push *generator* (zeros *g-hidden-size*)))
(defparameter *gw3* ($push *generator* (vxavier (list *g-hidden-size* *g-output-size*))))
(defparameter *gb3* ($push *generator* (zeros *g-output-size*)))

(defun generate (x)
  (let* ((z1 ($affine x *gw1* *gb1*))
         (a1 ($elu z1))
         (z2 ($affine a1 *gw2* *gb2*))
         (a2 ($elu z2)))
    ($affine a2 *gw3* *gb3*)))

(defparameter *discriminator* (parameters))
(defparameter *dw1* ($push *discriminator* (vxavier (list *d-input-size* *d-hidden-size*))))
(defparameter *db1* ($push *discriminator* (zeros *d-hidden-size*)))
(defparameter *dw2* ($push *discriminator* (vxavier (list *d-hidden-size* *d-hidden-size*))))
(defparameter *db2* ($push *discriminator* (zeros *d-hidden-size*)))
(defparameter *dw3* ($push *discriminator* (vxavier (list *d-hidden-size* *d-output-size*))))
(defparameter *db3* ($push *discriminator* (zeros *d-output-size*)))

(defun discriminate (x)
  (let* ((z1 ($affine x *dw1* *db1*))
         (a1 ($elu z1))
         (z2 ($affine a1 *dw2* *db2*))
         (a2 ($elu z2))
         (z3 ($affine a2 *dw3* *db3*)))
    ($sigmoid z3)))

(defparameter *d-sampler-fn* (get-distribution-sampler *data-mean* *data-sd*))
(defun d-sampler (n) (funcall *d-sampler-fn* n))

(defparameter *gi-sampler-fn* (get-generator-input-sampler))
(defun gi-sampler (m n) (funcall *gi-sampler-fn* m n))

(time
 (with-foreign-memory-limit ()
   (loop :for epoch :from 1 :to *num-epochs*
         :do (progn
               (loop :for dstep :from 0 :below *d-steps*
                     :do (progn
                           ($cg! *generator*)
                           ($cg! *discriminator*)
                           (let* ((d-real-data (d-sampler *d-input-size*))
                                  (d-real-decision (discriminate d-real-data))
                                  (d-real-error ($bce d-real-decision (ones 1)))
                                  (d-gen-input (gi-sampler *minibatch-size* *g-input-size*))
                                  (d-fake-data (generate d-gen-input))
                                  (d-fake-decision (discriminate ($transpose d-fake-data)))
                                  (d-fake-error ($bce d-fake-decision (zeros 1))))
                             (when (zerop (rem epoch *print-interval*))
                               (prn "EPOCH =>" epoch)
                               (prn "DRE/DFE:" d-real-error d-fake-error)
                               (prn " DSTAT:" ($mean d-real-data) ($sd d-real-data))
                               (prn " FSTAT:" ($mean d-fake-data) ($sd d-fake-data)))
                             ($amgd! *discriminator* *d-learning-rate* *beta1* *beta2*)
                             ($cg! *generator*)
                             ($cg! *discriminator*))))
               (loop :for gstep :from 0 :below *g-steps*
                     :do (progn
                           ($cg! *generator*)
                           ($cg! *discriminator*)
                           (let* ((gen-input (gi-sampler *minibatch-size* *g-input-size*))
                                  (g-fake-data (generate gen-input))
                                  (dg-fake-decision (discriminate ($transpose g-fake-data)))
                                  (g-error ($bce dg-fake-decision (ones 1))))
                             (when (zerop (rem epoch *print-interval*))
                               (prn "GE:" ($data g-error)))
                             ($amgd! *generator* *d-learning-rate* *beta1* *beta2*)
                             ($cg! *generator*)
                             ($cg! *discriminator*))))))))

(gcf)
