;; from
;; https://github.com/Abhipanda4/Sparse-Autoencoders

(defpackage :autoencoder
  (:use #:common-lisp
        #:mu
        #:th
        #:th.db.mnist))

(in-package :autoencoder)

;; load mnist data, takes ~22 secs in macbook 2017
(defparameter *mnist* (read-mnist-data))

;; mnist data has following dataset
;; train-images, train-labels and test-images, test-labels
(print *mnist*)

;; training data - uses batches for performance
(defparameter *mnist-train-image-batches*
  (loop :for i :from 0 :below 60
        :for rng = (loop :for k :from (* i 1000) :below (* (1+ i) 1000)
                         :collect k)
        :collect ($contiguous! ($index ($ *mnist* :train-images) 0 rng))))

(defparameter *num-input* 784)
(defparameter *num-hidden* 300)
(defparameter *num-batch* 1000)
(defparameter *epochs* 30)

(defparameter *rho* 0.01)
(defparameter *beta* 3)

(defparameter *ae* (parameters))

(defparameter *wenc* ($parameter *ae* (vxavier (list *num-input* *num-hidden*))))
(defparameter *benc* ($parameter *ae* (ones 1 *num-hidden*)))
(defparameter *wdec* ($parameter *ae* (vxavier (list *num-hidden* *num-input*))))
(defparameter *bdec* ($parameter *ae* (ones 1 *num-input*)))

(defparameter *os* (ones *num-batch* 1))
(defparameter *p* ($fill! (tensor *num-hidden*) *rho*))

(defun validate ()
  (let ((we ($data *wenc*))
        (be ($data *benc*))
        (wd ($data *wdec*))
        (bd ($data *bdec*)))
    (let* ((x ($ *mnist* :test-images))
           (os (ones ($size x 0) 1))
           (encoded ($sigmoid ($+ ($@ x we) ($@ os be))))
           (decoded ($sigmoid ($+ ($@ encoded wd) ($@ os bd))))
           (d ($- decoded x))
           (loss ($/ ($dot d d) ($size x 0))))
      loss)))

(defun kl-divergence (q &optional (usesf t))
  (let* ((q (if usesf ($softmax q) q))
         (p ($constant (if usesf ($softmax *p*) *p*)))
         (lpq ($log ($div p q)))
         (mp ($- ($constant 1) p))
         (mq ($- ($constant 1) q))
         (lmpq ($log ($div mp mq)))
         (s1 ($sum ($* p lpq)))
         (s2 ($sum ($* mp lmpq))))
    ($+ s1 s2)))

($cg! *ae*)

(loop :for epoch :from 1 :to (min 5 *epochs*)
      :do (progn
            (loop :for input :in *mnist-train-image-batches*
                  :for bidx :from 1
                  :for x = ($constant input)
                  :for encoded = ($sigmoid ($+ ($@ x *wenc*) ($@ ($constant *os*) *benc*)))
                  :for decoded = ($sigmoid ($+ ($@ encoded *wdec*) ($@ ($constant *os*) *bdec*)))
                  :for d = ($- decoded x)
                  :for mse = ($/ ($dot d d) ($constant *num-batch*))
                  :for rho-hat = ($mean encoded 0)
                  :for kld = (kl-divergence rho-hat)
                  :for esparsity = ($* kld ($constant *beta*))
                  :for loss = ($+ mse esparsity)
                  :do (progn
                        ($adgd! *ae*)
                        (when (zerop (rem bidx 10))
                          (prn "LOSS:" bidx "/" epoch ($data loss) ($data mse) ($data esparsity)))))
            (prn "[TEST]" epoch (validate))
            (gcf)))
