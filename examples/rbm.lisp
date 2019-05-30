;; from
;; https://github.com/odie2630463/Restricted-Boltzmann-Machines-in-pytorch

(defpackage :rbm
  (:use #:common-lisp
        #:mu
        #:th
        #:th.image
        #:th.db.mnist))

(in-package :rbm)

;; load mnist data, takes ~22 secs in macbook 2017
(defparameter *mnist* (read-mnist-data))

;; mnist data has following dataset
;; train-images, train-labels and test-images, test-labels
(prn *mnist*)

(defparameter *batch-size* 60)
(defparameter *batch-count* (/ 60000 *batch-size*))

(defparameter *mnist-train-image-batches*
  (loop :for i :from 0 :below *batch-count*
        :for range = (loop :for k :from (* i *batch-size*) :below (* (1+ i) *batch-size*)
                           :collect k)
        :collect ($contiguous! ($index ($ *mnist* :train-images) 0 range))))

(defparameter *n-vis* 784)
(defparameter *n-hin* 500)

(defparameter *rbm* (parameters))

(defparameter *w* ($push *rbm* ($* 1E-2 (rndn *n-vis* *n-hin*))))
(defparameter *vb* ($push *rbm* (zeros 1 *n-vis*)))
(defparameter *hb* ($push *rbm* (zeros 1 *n-hin*)))
(defparameter *k* 1)

(defun sample-from-p (p)
  ($relu ($sign ($- p (apply #'rnd ($size p))))))

(defun v-to-h (v)
  (let* ((nrow ($size v 0))
         (os (ones nrow 1))
         (ah ($+ ($@ v *w*) ($@ os *hb*)))
         (ph ($sigmoid ah)))
    ph))

(defun h-to-v (h)
  (let* ((nrow ($size h 0))
         (os (ones nrow 1))
         (av ($+ ($@ h ($transpose *w*)) ($@ os *vb*)))
         (pv ($sigmoid av)))
    pv))

(defun run (v)
  (let* ((vh (v-to-h v))
         (h (sample-from-p vh))
         (vr nil))
    (loop :for i :from 0 :below *k*
          :for pv = (h-to-v h)
          :for cv = (sample-from-p pv)
          :for ph = (v-to-h cv)
          :for ch = (sample-from-p ph)
          :do (setf h ch
                    vr cv))
    vr))

(defun free-energy (v)
  (let* ((vbias ($@ v ($transpose *vb*)))
         (nrow ($size v 0))
         (os (ones nrow 1))
         (wxb ($+ ($@ v *w*) ($@ os *hb*)))
         (hidden ($sum ($log ($+ ($exp wxb) 1)) 1)))
    ($mean ($- ($neg hidden) vbias))))

(defun opt! () ($gd! *rbm* 0.1))

(defun mean (vs) (* 1D0 (/ (reduce #'+ vs) ($count vs))))

(defparameter *epoch* 10)

($cg! *rbm*)

(loop :for epoch :from 1 :to *epoch*
      :for loss = nil
      :do (progn
            ($cg! *rbm*)
            (loop :for input :in *mnist-train-image-batches*
                  :for sample = ($bernoulli input input)
                  :for v = sample
                  :for v1 = (run v)
                  :for l = ($- (free-energy v) (free-energy v1))
                  :do (progn
                        (push ($data l) loss)
                        (opt!)))
            (prn epoch (mean loss))))

(defparameter *output* (format nil "~A/Desktop" (user-homedir-pathname)))

(defun outpng (data fname &optional (w 28) (h 28))
  (let ((img (opticl:make-8-bit-gray-image w h))
        (d ($reshape data w h)))
    (loop :for i :from 0 :below h
          :do (loop :for j :from 0 :below w
                    :do (progn
                          (setf (aref img i j) (round (* 255 ($ d i j)))))))
    (opticl:write-png-file (format nil "~A/~A" *output* fname) img)))

(let* ((input (car *mnist-train-image-batches*))
       (sample ($bernoulli input input))
       (v sample)
       (v1 (run v))
       (idx (random ($size input 0))))
  (outpng ($index input 0 idx) "input.png")
  (outpng ($index v 0 idx) "outv.png")
  (outpng ($index ($data v1) 0 idx) "outv1.png"))

;; clear resources
(setf *mnist* nil)
(setf *mnist-train-image-batches* nil)
(gcf)
