;; from
;; https://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/

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

;; training data - uses batches for performance
(defparameter *batch-size* 100)
(defparameter *batch-count* (/ 60000 *batch-size*))

(defparameter *mnist-train-image-batches*
  (loop :for i :from 0 :below *batch-count*
        :for rng = (loop :for k :from (* i *batch-size*) :below (* (1+ i) *batch-size*)
                         :collect k)
        :collect ($contiguous! ($index ($ *mnist* :train-images) 0 rng))))

(defparameter *discriminator* (parameters))
(defparameter *generator* (parameters))

(defparameter *gen-size* 100)
(defparameter *hidden-size* 128)
(defparameter *img-size* (* 28 28))

;; discriminator network
(defparameter *dw1* ($parameter *discriminator* (vxavier (list *img-size* *hidden-size*))))
(defparameter *db1* ($parameter *discriminator* (zeros 1 *hidden-size*)))
(defparameter *dw2* ($parameter *discriminator* (vxavier (list *hidden-size* 1))))
(defparameter *db2* ($parameter *discriminator* (zeros 1 1)))

;; generator network
(defparameter *gw1* ($parameter *generator* (vxavier (list *gen-size* *hidden-size*))))
(defparameter *gb1* ($parameter *generator* (zeros 1 *hidden-size*)))
(defparameter *gw2* ($parameter *generator* (vxavier (list *hidden-size* *img-size*))))
(defparameter *gb2* ($parameter *generator* (zeros 1 *img-size*)))

(defparameter *os* (ones *batch-size* 1))

(defun generate (z)
  (let* ((gh1 ($relu ($+ ($@ z *gw1*) ($@ ($constant *os*) *gb1*))))
         (glogprob ($+ ($@ gh1 *gw2*) ($@ ($constant *os*) *gb2*))))
    (th::tensor-clamp ($data glogprob) ($data glogprob) -50 50)
    ($sigmoid glogprob)))

(defun discriminate (x)
  (let* ((dh1 ($relu ($+ ($@ x *dw1*) ($@ ($constant *os*) *db1*))))
         (dlogit ($+ ($@ dh1 *dw2*) ($@ ($constant *os*) *db2*))))
    (th::tensor-clamp ($data dlogit) ($data dlogit) -50 50)
    ($sigmoid dlogit)))

(defun samplez () ($constant ($ru! (tensor *batch-size* *gen-size*) -1 1)))

(defun outpng (data fname)
  (let ((img (opticl:make-8-bit-gray-image 28 28))
        (d ($reshape data 28 28)))
    (loop :for i :from 0 :below 28
          :do (loop :for j :from 0 :below 28
                    :do (progn
                          (setf (aref img i j) (round (* 255 ($ d i j)))))))
    (opticl:write-png-file fname img)))

($cg! *discriminator*)
($cg! *generator*)

(defparameter *epoch* 1)
(defparameter *k* 1)
(defparameter *j* 1)

(defparameter *eps* ($constant 1E-7))

(defun bced (dr df)
  ($+ ($bce dr ($constant ($* 0.9 ($one dr))))
      ($bce df ($constant ($zero df)))))

(defun bceg (df) ($bce df ($constant ($one df))))

(loop :for epoch :from 1 :to *epoch*
      :do (progn
            ($cg! *discriminator*)
            ($cg! *generator*)
            (loop :for data :in *mnist-train-image-batches*
                  :for bidx :from 0
                  :for x = ($constant data)
                  :for z = (samplez)
                  :do (progn
                        (let* ((dr (discriminate x))
                               (g (generate z))
                               (df (discriminate g)))
                          (bced dr df)
                          ($adgd! *discriminator*)
                          ($cg! *discriminator*)
                          ($cg! *generator*))
                        (let* ((g (generate z))
                               (df (discriminate g)))
                          (bceg df)
                          ($adgd! *generator*)
                          ($cg! *discriminator*)
                          ($cg! *generator*))))
            (let* ((x ($constant (car *mnist-train-image-batches*)))
                   (dr (discriminate x))
                   (z (samplez))
                   (g (generate z))
                   (df (discriminate g))
                   (l ($+ ($bce dr ($constant ($* 0.9 ($one dr))))
                          ($bce df ($constant ($zero df))))))
              ($cg! *discriminator*)
              ($cg! *generator*)
              (prn "DL:" epoch ($data l))
              (let ((fname (format nil "/Users/Sungjin/Desktop/img~A.png" epoch)))
                (outpng ($index ($data g) 0 0) fname)))
            (gcf)))

(loop :for epoch :from 1 :to *epoch*
      :do (loop :for data :in *mnist-train-image-batches*
                :for bidx :from 0
                :for x = ($constant data)
                :do (progn
                      (when (zerop (rem bidx 10)) (prn epoch "=>" bidx))
                      ($cg! *discriminator*)
                      ($cg! *generator*)
                      (let* ((z (samplez))
                             (g (generate z))
                             (dr (discriminate x))
                             (df (discriminate g))
                             (l ($+ ($bce dr ($constant ($* 0.9 (apply #'ones ($size ($data dr))))))
                                    ($bce df ($constant (apply #'zeros ($size ($data df))))))))
                        ($adgd! *discriminator*)
                        ($cg! *discriminator*)
                        ($cg! *generator*)
                        (when (zerop (rem bidx 100))
                          (prn "  LD:" ($data l))
                          (prn "  PR:" ($mean ($data dr)))
                          (prn "  PG:" ($mean ($data df)))))
                      (let* ((z (samplez))
                             (g (generate z))
                             (df (discriminate g))
                             (l ($bce df ($constant ($* 0.9 (apply #'ones ($size ($data df))))))))
                        ($adgd! *generator*)
                        ($cg! *discriminator*)
                        ($cg! *generator*)
                        (when (zerop (rem bidx 100))
                          (prn "  LG:" ($data l))
                          (prn "  GF:" ($mean ($data df)))
                          (let ((fname (format nil
                                               "/Users/Sungjin/Desktop/img~A.png"
                                               bidx)))
                            (outpng ($index ($data g) 0 0) fname))))
                      (gcf))))

(loop :for epoch :from 1 :to *epoch*
      :do (loop :for data :in *mnist-train-image-batches*
                :for bidx :from 0
                :for x = ($constant data)
                :do (progn
                      (when (zerop (rem bidx 10)) (prn epoch "=>" bidx))
                      ($cg! *discriminator*)
                      ($cg! *generator*)
                      (loop :for k :from 1 :to *k*
                            :for z = (samplez)
                            :for g = (generate z)
                            :for dr = (discriminate x)
                            :for df = (discriminate g)
                            :for ldr = ($log ($+ dr *eps*))
                            :for ldf = ($log ($+ ($- 1 df) *eps*))
                            :for l = ($neg ($mean ($+ ldr ldf)))
                            :do (progn
                                  ($adgd! *discriminator*)
                                  ($cg! *discriminator*)
                                  ($cg! *generator*)
                                  (when (and (zerop (rem bidx 100)) (eq k *k*))
                                    (prn "  LD:" ($data l))
                                    (prn "  PR:" ($mean ($data dr)))
                                    (prn "  PG:" ($mean ($data df))))))
                      (loop :for j :from 1 :to *j*
                            :for z = (samplez)
                            :for g = (generate z)
                            :for df = (discriminate g)
                            :for l = ($neg ($mean ($log ($+ df *eps*))))
                            :do (progn
                                  ($adgd! *generator*)
                                  ($cg! *discriminator*)
                                  ($cg! *generator*)
                                  (when (and (zerop (rem bidx 100)) (eq j *j*))
                                    (prn "  LG:" ($data l))
                                    (prn "  GF:" ($mean ($data df)))
                                    (let ((fname (format nil
                                                         "/Users/Sungjin/Desktop/img~A.png"
                                                         bidx)))
                                      (outpng ($index ($data g) 0 0) fname)))))
                      (gcf))))

;; train D for data sitribution
(loop :for data :in *mnist-train-image-batches*
      :for x = ($constant data)
      :for d = (discriminate x)
      :for ld = ($log ($+ d *eps*))
      :for l = ($neg ($mean ld))
      :do (progn
            (prn ($data l))
            ($adgd! *discriminator*)
            ($cg! *discriminator*)
            ($cg! *generator*)))

;; train G for fake generator
(loop :for i :from 1 :to 2
      :for z = (samplez)
      :for g = (generate z)
      :for d = (discriminate g)
      :for ld = ($log ($+ d *eps*))
      :for l = ($neg ($mean ld))
      :do (progn
            (prn ($data l))
            ($adgd! *generator*)
            ($cg! *discriminator*)
            ($cg! *generator*)))

;; train
(loop :for epoch :from 1 :to *epoch*
      :do (loop :for data :in *mnist-train-image-batches*
                :for bidx :from 0
                :for x = ($constant data)
                :do (progn
                      (when (zerop (rem bidx 10))
                        (prn epoch "=>" bidx))
                      ($cg! *discriminator*)
                      ($cg! *generator*)
                      (loop :for k :from 1 :to *k*
                            :do (let* ((z (samplez))
                                       (g (generate z))
                                       (dr (discriminate x))
                                       (df (discriminate g))
                                       (ldr ($log ($+ dr *eps*)))
                                       (ldf ($log ($+ ($- 1 df) *eps*)))
                                       (l ($neg ($mean ($+ ldr ldf)))))
                                  ($adgd! *discriminator*)
                                  ;;($adgd! *generator*)
                                  ($cg! *discriminator*)
                                  ($cg! *generator*)
                                  (when (and (zerop (rem bidx 100)) (eq k *k*))
                                    (prn "  LD:" ($data l))
                                    (prn "  PR:" ($mean ($data dr)))
                                    (prn "  PG:" ($mean ($data df)))
                                    (let ((fname (format nil
                                                         "/Users/Sungjin/Desktop/img~A.png"
                                                         bidx)))
                                      (outpng ($index ($data g) 0 0) fname)))))
                      (let* ((z (samplez))
                             (g (generate z))
                             (gf (discriminate g))
                             (l ($mean ($log ($+ ($- 1 gf) *eps*)))))
                        ($adgd! *generator*)
                        ($cg! *discriminator*)
                        ($cg! *generator*)
                        (when (zerop (rem bidx 100))
                          (prn "  LG:" ($data l))
                          (prn "  GF:" ($mean ($data gf)))))
                      (gcf))))

(loop :for epoch :from 1 :to *epoch*
      :do (loop :for input :in *mnist-train-image-batches*
                :for bidx :from 0
                :for x = ($constant input)
                :do (progn
                      (when (zerop (rem bidx 10))
                        (prn epoch "=>" bidx))
                      ($cg! *discriminator*)
                      ($cg! *generator*)
                      (loop :for k :from 1 :to *k*
                            :do (let* ((z (samplez))
                                       (g (generate z))
                                       (dr (discriminate x))
                                       (df (discriminate g))
                                       (ldr ($log ($+ dr *eps*)))
                                       (ldf ($log ($+ ($- ($constant 1) df) *eps*)))
                                       (l ($neg ($mean ($+ ldr ldf)))))
                                  ($adgd! *discriminator*)
                                  ;;($adgd! *generator*)
                                  ($cg! *discriminator*)
                                  ($cg! *generator*)
                                  (when (and (zerop (rem bidx 100)) (eq k *k*))
                                    (prn "  LD:" ($data l))
                                    (prn "  PR:" ($mean ($data dr)))
                                    (prn "  PG:" ($mean ($data df)))
                                    (let ((fname (format nil
                                                         "/Users/Sungjin/Desktop/img~A.png"
                                                         bidx)))
                                      (outpng ($index ($data g) 0 0) fname)))))
                      (loop :for j :from 1 :to *j*
                            :do (let* ((z (samplez))
                                       (g (generate z))
                                       (gf (discriminate g))
                                       (l ($neg ($mean ($log ($+ gf *eps*))))))
                                  ($adgd! *generator*)
                                  ($cg! *discriminator*)
                                  ($cg! *generator*)
                                  (when (and (zerop (rem bidx 100)) (eq j *j*))
                                    (prn "  LG:" ($data l))
                                    (prn "  GF:" ($mean ($data gf))))))
                      (gcf))))

(-> (samplez)
    (generate)
    (discriminate)
    ($mean)
    (prn))

(-> ($ *mnist-train-image-batches* (random ($count *mnist-train-image-batches*)))
    ($constant)
    (discriminate)
    ($mean)
    (prn))

(-> (samplez)
    (generate)
    ($data)
    ($mean)
    (prn))

(-> ($ *mnist-train-image-batches* (random ($count *mnist-train-image-batches*)))
    ($mean)
    (prn))

(defparameter *samples* (-> (samplez)
                            (generate)
                            ($data)))
(defparameter *data* ($ *mnist-train-image-batches*
                        (random ($count *mnist-train-image-batches*))))

(let ((datag ($reshape ($index *samples* 0 (random *batch-size*)) 28 28))
      (datad ($reshape ($index *data* 0 (random *batch-size*)) 28 28)))
  (prn datag)
  (prn datad)
  (outpng datag "/Users/Sungjin/Desktop/testg.png")
  (outpng datad "/Users/Sungjin/Desktop/testd.png")
  ($cg! *discriminator*)
  ($cg! *generator*))
