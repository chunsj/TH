(ql:quickload :cl-ppcre)
(ql:quickload :parse-number)

(defpackage :vgg16-weight-proc
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :vgg16-weight-proc)

(defun convert-vgg16-weights (filename)
  (with-open-file (in filename :direction :input)
    (write-conv-weights in '(64 3 3 3) "vgg16-k1.txt" "vgg16-b1.txt")
    (write-conv-weights in '(64 64 3 3) "vgg16-k2.txt" "vgg16-b2.txt")
    (write-conv-weights in '(128 64 3 3) "vgg16-k3.txt" "vgg16-b3.txt")
    (write-conv-weights in '(128 128 3 3) "vgg16-k4.txt" "vgg16-b4.txt")
    (write-conv-weights in '(256 128 3 3) "vgg16-k5.txt" "vgg16-b5.txt")
    (write-conv-weights in '(256 256 3 3) "vgg16-k6.txt" "vgg16-b6.txt")
    (write-conv-weights in '(256 256 3 3) "vgg16-k7.txt" "vgg16-b7.txt")
    (write-conv-weights in '(512 256 3 3) "vgg16-k8.txt" "vgg16-b8.txt")
    (write-conv-weights in '(512 512 3 3) "vgg16-k9.txt" "vgg16-b9.txt")
    (write-conv-weights in '(512 512 3 3) "vgg16-k10.txt" "vgg16-b10.txt")
    (write-conv-weights in '(512 512 3 3) "vgg16-k11.txt" "vgg16-b11.txt")
    (write-conv-weights in '(512 512 3 3) "vgg16-k12.txt" "vgg16-b12.txt")
    (write-conv-weights in '(512 512 3 3) "vgg16-k13.txt" "vgg16-b13.txt")
    (write-affine-weights in '(25088 4096) "vgg16-w14.txt" "vgg16-b14.txt")
    (write-affine-weights in '(4096 4096) "vgg16-w15.txt" "vgg16-b15.txt")
    (write-affine-weights in '(4096 1000) "vgg16-w16.txt" "vgg16-b16.txt")))

;; reading performance
(with-open-file (in "/Users/Sungjin/Documents/Lisp/zoo/vgg16/k27" :direction :input)
  (gcf)
  ;;(time (mapcar #'parse-number:parse-number (cl-ppcre:split "\\s+" (read-line in nil))))
  (time (read-line in nil))
  (gcf))

(let* ((f (file.disk "w16h.txt" "w"))
       (tx (tensor 4096 1000)))
  ($fwrite tx f)
  ($fclose f))

(let* ((f (file.disk "b16h.txt" "w"))
       (tx (tensor 1 1000)))
  ($fwrite tx f)
  ($fclose f))

(loop :for i :from 1 :to 13
      :for kdim :in '((64 3 3 3)
                      (64 64 3 3)
                      (128 64 3 3)
                      (128 128 3 3)
                      (256 128 3 3)
                      (256 256 3 3)
                      (256 256 3 3)
                      (512 256 3 3)
                      (512 512 3 3)
                      (512 512 3 3)
                      (512 512 3 3)
                      (512 512 3 3)
                      (512 512 3 3))
      :for bdim = (car kdim)
      :do (progn
            (let ((f (file.disk (format nil "k~Ah.txt" i) "w"))
                  (tx (apply #'tensor kdim)))
              ($fwrite tx f)
              ($fclose f))
            (let ((f (file.disk (format nil "b~Ah.txt" i) "w"))
                  (tx (tensor bdim)))
              ($fwrite tx f)
              ($fclose f))))

(gcf)

(let* ((f (file.disk "vgg16-b16.txt" "r"))
       (tx (tensor)))
  ($fread tx f)
  (prn tx)
  ($fclose f))

(let* ((f (file.disk "cath.txt" "w"))
       (tx (tensor 3 224 224)))
  ($fwrite tx f)
  ($fclose f))

(let ((imgdata (opticl:make-8-bit-rgb-image 224 224)))
  (with-open-file (in "/Users/Sungjin/Desktop/cat.txt" :direction :input)
    (loop :for l :from 0 :below 224
          :for vals = (coerce (mapcar #'parse-number:parse-number
                                      (cl-ppcre:split "\\s+" (read-line in nil)))
                              'vector)
          :do (loop :for m :from 0 :below 224
                    :for i = (+ (* 3 m) 2)
                    :for j = (+ (* 3 m) 1)
                    :for k = (+ (* 3 m))
                    :do (progn
                          (setf (aref imgdata l m 0) (aref vals i))
                          (setf (aref imgdata l m 1) (aref vals j))
                          (setf (aref imgdata l m 2) (aref vals k))))))
  (prn imgdata)
  (opticl:write-png-file "/Users/Sungjin/Desktop/cat.png" imgdata))

(defparameter *vgg16-weights* (th.m.vgg16:read-vgg16-weights))
(defparameter *vgg16* (th.m.vgg16:vgg16 :all *vgg16-weights*))

(let* ((rgb (th.image:tensor-from-png-file "data/cat.vgg16.png" :normalize nil))
       (bgr (th.m.vgg16:convert-to-vgg16-input rgb)))
  (prn bgr)
  (let* ((r1000 (funcall *vgg16* bgr))
         (category-vi ($max r1000 1))
         (category-val ($ (car category-vi) 0 0))
         (category-idx ($ (cadr category-vi) 0 0)))
    (prn r1000)
    (prn category-val)
    (prn category-idx)
    (prn ($ (th.m.vgg16:vgg16-categories) category-idx))))

(let ((f (file.disk "cat.txt" "r"))
      (tx (tensor)))
  ($fread tx f)
  ($fclose f)
  (th.image:write-tensor-png-file tx "/Users/Sungjin/Desktop/cat.png" :denormalize nil))
