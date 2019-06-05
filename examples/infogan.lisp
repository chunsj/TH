;; from
;; https://wiseodd.github.io/techblog/2017/01/29/infogan/
;; https://github.com/wiseodd/generative-models/blob/master/GAN/infogan/infogan_pytorch.py

(defpackage :infogan
  (:use #:common-lisp
        #:mu
        #:th
        #:th.image
        #:th.db.mnist))

(in-package :infogan)

;; load mnist data, takes ~22 secs in macbook 2017
(defparameter *mnist* (read-mnist-data))

;; mnist data has following dataset
;; train-images, train-labels and test-images, test-labels
(prn *mnist*)

(defparameter *output* (format nil "~A/Desktop" (user-homedir-pathname)))

(defparameter *eps* 1E-8)

(defun lossd (dr df)
  (let ((dr ($+ dr *eps*))
        (df ($+ df *eps*)))
    ($neg ($mean ($+ ($log dr) ($log ($- 1 df)))))))
(defun lossg (df)
  (let ((df ($+ df *eps*)))
    ($neg ($mean ($log df)))))
(defun lossq (c qc)
  (let ((qc ($+ qc *eps*)))
    ($mean ($neg ($sum ($* c ($log qc)) 1)))))

;; XXX cannot figure out why adam works best (adadelta does not work well)
(defun optm (params) ($amgd! params 1E-3))

(defun outpng (data fname &optional (w 28) (h 28))
  (let ((img (opticl:make-8-bit-gray-image w h))
        (d ($reshape data w h)))
    (loop :for i :from 0 :below h
          :do (loop :for j :from 0 :below w
                    :do (progn
                          (setf (aref img i j) (round (* 255 ($ d i j)))))))
    (opticl:write-png-file fname img)))

;; training data - uses batches for performance, 30, 60 works well
(defparameter *batch-size* 30)
(defparameter *batch-count* (/ 60000 *batch-size*))

(defparameter *mnist-train-image-batches*
  (loop :for i :from 0 :below *batch-count*
        :for range = (loop :for k :from (* i *batch-size*) :below (* (1+ i) *batch-size*)
                           :collect k)
        :collect ($contiguous! ($index ($ *mnist* :train-images) 0 range))))

(defparameter *discriminator* (parameters))
(defparameter *generator* (parameters))
(defparameter *qnet* (parameters))

(defparameter *gen-size* 100)
(defparameter *hidden-size* 128)
(defparameter *img-size* (* 28 28))
(defparameter *lbl-size* 10)

(defun xinit (size) ($* (apply #'rndn size) (/ 1 (sqrt (/ ($ size 0) 2)))))

(defparameter *os* (ones *batch-size*))

;; generator network
(defparameter *gw1* ($push *generator* (xinit (list (+ *gen-size* *lbl-size*) *hidden-size*))))
(defparameter *gb1* ($push *generator* (zeros *hidden-size*)))
(defparameter *gw2* ($push *generator* (xinit (list *hidden-size* *img-size*))))
(defparameter *gb2* ($push *generator* (zeros *img-size*)))

;; you can apply leaky relu, $lrelu
(defun generate (z c)
  (-> ($cat z c 1)
      ($affine *gw1* *gb1* *os*)
      ($relu)
      ($affine *gw2* *gb2* *os*)
      ($sigmoid)))

;; discriminator network
(defparameter *dw1* ($push *discriminator* (xinit (list *img-size* *hidden-size*))))
(defparameter *db1* ($push *discriminator* (zeros *hidden-size*)))
(defparameter *dw2* ($push *discriminator* (xinit (list *hidden-size* 1))))
(defparameter *db2* ($push *discriminator* (zeros 1)))

;; you can apply leaky relu, $lrelu
(defun discriminate (x)
  (-> x
      ($affine *dw1* *db1* *os*)
      ($relu)
      ($affine *dw2* *db2* *os*)
      ($sigmoid)))

;; q(c|X) network
(defparameter *qw1* ($push *qnet* (xinit (list *img-size* *hidden-size*))))
(defparameter *qb1* ($push *qnet* (zeros *hidden-size*)))
(defparameter *qw2* ($push *qnet* (xinit (list *hidden-size* *lbl-size*))))
(defparameter *qb2* ($push *qnet* (zeros *lbl-size*)))

(defun qnet (x)
  (-> x
      ($affine *qw1* *qb1* *os*)
      ($relu)
      ($affine *qw2* *qb2* *os*)
      ($softmax)))

(defun rones (nrows cprobs)
  (let* ((indices ($multinomial cprobs nrows))
         (res (zeros ($size indices 0) ($size cprobs 0))))
    (loop :for i :from 0 :below nrows
          :for j = ($ indices i)
          :do (setf ($ res i j) 1))
    res))

(defparameter *cprobs* ($/ (ones 10) *lbl-size*))

(defun samplez () (rndn *batch-size* *gen-size*))
(defun samplec () (rones *batch-size* *cprobs*))

(defparameter *epoch* 100)
(defparameter *k* 1)

($cg! *discriminator*)
($cg! *generator*)
($cg! *qnet*)

(defparameter *train-data-batches* (subseq *mnist-train-image-batches* 0))
(defparameter *train-count* ($count *train-data-batches*))

(gcf)

(time
 (loop :for epoch :from 1 :to *epoch*
       :for dloss = 0
       :for gloss = 0
       :for qloss = 0
       :do (progn
             ($cg! *discriminator*)
             ($cg! *generator*)
             (prn "*****")
             (prn "EPOCH:" epoch)
             (loop :for x :in *train-data-batches*
                   :for bidx :from 0
                   :for c = (samplec)
                   :for z = (samplez)
                   :do (let ((dlv nil)
                             (dgv nil)
                             (dqv nil))
                         ;; discriminator
                         (dotimes (k *k*)
                           (let* ((dr (discriminate x))
                                  (df (discriminate (generate z c)))
                                  (l ($data (lossd dr df))))
                             (incf dloss l)
                             (setf dlv l)
                             (optm *discriminator*)
                             ($cg! *discriminator*)
                             ($cg! *generator*)
                             ($cg! *qnet*)))
                         ;; generator
                         (let* ((df (discriminate (generate z c)))
                                (l ($data (lossg df))))
                           (incf gloss l)
                           (setf dgv l)
                           (optm *generator*)
                           ($cg! *discriminator*)
                           ($cg! *generator*)
                           ($cg! *qnet*))
                         ;; q network
                         (let* ((g-sample (generate z c))
                                (qc (qnet g-sample))
                                (l ($data (lossq c qc))))
                           (incf qloss l)
                           (setf dqv l)
                           (optm *generator*)
                           (optm *qnet*)
                           ($cg! *discriminator*)
                           ($cg! *generator*)
                           ($cg! *qnet*))
                         (when (zerop (rem bidx 200))
                           (prn "  D/L/Q:" bidx dlv dgv dqv))))
             (when (zerop (rem epoch 1))
               (let ((g (generate (samplez) (samplec))))
                 ($cg! *discriminator*)
                 ($cg! *generator*)
                 ($cg! *qnet*)
                 (loop :for i :from 1 :to 1
                       :for s = (random *batch-size*)
                       :for fname = (format nil "~A/i~A-~A.png" *output* epoch i)
                       :do (outpng ($index ($data g) 0 s) fname))))
             (prn " LOSS:" epoch (/ dloss *train-count*) (/ gloss *train-count*)
                  (/ qloss *train-count*)))))

(defun outpngs25 (data81 fname &optional (w 28) (h 28))
  (let* ((n 5)
         (img (opticl:make-8-bit-gray-image (* n w) (* n h)))
         (datas (mapcar (lambda (data) ($reshape data w h)) data81)))
    (loop :for i :from 0 :below n
          :do (loop :for j :from 0 :below n
                    :for sx = (* j w)
                    :for sy = (* i h)
                    :for d = ($ datas (+ (* j n) i))
                    :do (loop :for i :from 0 :below h
                              :do (loop :for j :from 0 :below w
                                        :do (progn
                                              (setf (aref img (+ sx i) (+ sy j))
                                                    (round (* 255 ($ d i j)))))))))
    (opticl:write-png-file fname img)))

;; generate samples
(let* ((c (let ((c (zeros *batch-size* *lbl-size*)))
            (loop :for i :from 0 :below *batch-size*
                  :do (if (< i 10)
                          (setf ($ c i i) 1)
                          (setf ($ c i (rem i 10)) 1)))
            c))
       (generated (generate (samplez) c)))
  (outpngs25 (loop :for i :from 0 :below 25
                   :collect ($index ($data generated) 0 i))
             (format nil "~A/G9.png" *output*))
  ($cg! *discriminator*)
  ($cg! *generator*)
  ($cg! *qnet*))

(setf *mnist* nil
      *mnist-train-image-batches* nil
      *train-data-batches* nil)

(gcf)
