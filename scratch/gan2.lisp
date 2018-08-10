;; from
;; https://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/
;; pytprch code referenced

(ql:quickload :opticl)

(defpackage :gan-work
  (:use #:common-lisp
        #:mu
        #:th
        #:th.db.mnist))

(in-package :gan-work)

;; load mnist data, takes ~22 secs in macbook 2017
(defparameter *mnist* (read-mnist-data))

;; mnist data has following dataset
;; train-images, train-labels and test-images, test-labels
(print *mnist*)

(defun bced (dr df)
  ($+ ($bce dr ($constant ($one dr)))
      ($bce df ($constant ($zero df)))))

(defun bceg (df) ($bce df ($constant ($one df))))

(defun lossd (dr df) (bced dr df))
(defun lossg (df) (bceg df))

(defun optm (params) ($adgd! params))

(defun outpng (data fname &optional (w 28) (h 28))
  (let ((img (opticl:make-8-bit-gray-image w h))
        (d ($reshape data w h)))
    (loop :for i :from 0 :below h
          :do (loop :for j :from 0 :below w
                    :do (progn
                          (setf (aref img i j) (round (* 255 ($ d i j)))))))
    (opticl:write-png-file fname img)))

(defun norm (x &optional (mean 0.5) (sd 0.5))
  (let* ((mx ($mean x))
         (sx ($sd x))
         (nx ($/ ($- x mx) sx)))
    ($+ ($* nx sd) mean)))

;; training data - uses batches for performance
(defparameter *batch-size* 1000)
(defparameter *batch-count* (/ 60000 *batch-size*))

(defparameter *mnist-train-image-batches*
  (loop :for i :from 0 :below *batch-count*
        :for rng = (loop :for k :from (* i *batch-size*) :below (* (1+ i) *batch-size*)
                         :collect k)
        :collect ($contiguous! (norm ($index ($ *mnist* :train-images) 0 rng)))))

(defparameter *discriminator* (parameters))
(defparameter *generator* (parameters))

(defparameter *gen-size* 100)
(defparameter *hidden-size* 128)
(defparameter *img-size* (* 28 28))

(defun xinit (size)
  ($variable ($* (apply #'rndn size) (/ 1 (sqrt (/ ($ size 0) 2))))))

;; discriminator network
(defparameter *dw1* ($parameter *discriminator* (xinit (list *img-size* *hidden-size*))))
(defparameter *db1* ($parameter *discriminator* (zeros 1 *hidden-size*)))
(defparameter *dw2* ($parameter *discriminator* (xinit (list *hidden-size* 1))))
(defparameter *db2* ($parameter *discriminator* (zeros 1 1)))

;; generator network
(defparameter *gw1* ($parameter *generator* (xinit (list *gen-size* *hidden-size*))))
(defparameter *gb1* ($parameter *generator* (zeros 1 *hidden-size*)))
(defparameter *gw2* ($parameter *generator* (xinit (list *hidden-size* *img-size*))))
(defparameter *gb2* ($parameter *generator* (zeros 1 *img-size*)))

(defparameter *os* (ones *batch-size* 1))

(defun generate (z)
  (let* ((gh1 ($relu ($+ ($@ z *gw1*) ($@ ($constant *os*) *gb1*))))
         (glogprob ($+ ($@ gh1 *gw2*) ($@ ($constant *os*) *gb2*))))
    ($sigmoid glogprob)))

(defun discriminate (x)
  (let* ((dh1 ($relu ($+ ($@ x *dw1*) ($@ ($constant *os*) *db1*))))
         (dlogit ($+ ($@ dh1 *dw2*) ($@ ($constant *os*) *db2*))))
    ($sigmoid dlogit)))

(defun samplez () ($constant (rndn *batch-size* *gen-size*)))

(defparameter *epoch* 100)
(defparameter *k* 1)

($cg! *discriminator*)
($cg! *generator*)

(defparameter *train-data-batches* (subseq *mnist-train-image-batches* 0))
(defparameter *train-count* ($count *train-data-batches*))

(loop :for epoch :from 1 :to *epoch*
      :for dloss = 0
      :for gloss = 0
      :do (progn
            ($cg! *discriminator*)
            ($cg! *generator*)
            (loop :for data :in *train-data-batches*
                  :for bidx :from 0
                  :for x = ($constant data)
                  :do (progn
                        ;; discriminator
                        (dotimes (k *k*)
                          (let ((dr (discriminate x))
                                (df (discriminate (generate (samplez)))))
                            (incf dloss ($data (lossd dr df)))
                            (optm *discriminator*)
                            ($cg! *discriminator*)
                            ($cg! *generator*)))
                        ;; generator
                        (let ((df (discriminate (generate (samplez)))))
                          (incf gloss ($data (lossg df)))
                          (optm *generator*)
                          ($cg! *discriminator*)
                          ($cg! *generator*))))
            (when (zerop (rem epoch 10))
              (let ((g (generate (samplez))))
                ($cg! *discriminator*)
                ($cg! *generator*)
                (loop :for i :from 1 :to 1
                      :for s = (random *batch-size*)
                      :for fname = (format nil "/Users/Sungjin/Desktop/i~A-~A.png" epoch i)
                      :do (outpng ($index ($data g) 0 s) fname))))
            (prn "LOSS:" epoch (/ dloss *train-count*) (/ gloss *train-count*))
            (gcf)))

(let ((indices (loop :for i :from 0 :below 10
                     :collect (random *batch-size*))))
  (loop :for i :in indices
        :do (outpng ($index ($data (generate (samplez))) 0 i)
                    (format nil "/Users/Sungjin/Desktop/test-~A.png" i))))
