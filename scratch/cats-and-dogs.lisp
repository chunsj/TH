(ql:quickload :opticl)

(defpackage :cats-and-dogs
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :cats-and-dogs)

(defun read-jpeg-file (filename &optional (imgh 64) (imgw 64))
  (let* ((imgdataraw (opticl:read-jpeg-file filename))
         (imgdata (opticl:resize-image imgdataraw imgh imgw :interpolate :bilinear))
         (dimensions (array-dimensions imgdata))
         (height (nth 0 dimensions))
         (width (nth 1 dimensions))
         (channels (nth 2 dimensions))
         (tensor (tensor channels height width)))
    (loop :for h :from 0 :below height
          :do (loop :for w :from 0 :below width
                    :do (let ((r (aref imgdata h w 0))
                              (g (aref imgdata h w 1))
                              (b (aref imgdata h w 2)))
                          (setf ($ tensor 0 h w) (/ r 255.0)
                                ($ tensor 1 h w) (/ g 255.0)
                                ($ tensor 2 h w) (/ b 255.0)))))
    tensor))

(defparameter *data-directory* "/Users/Sungjin/CatsDogs")

(defun read-train-cat-file (number &optional (h 64) (w 64))
  (when (and (>= number 0) (<= number 10000))
    (let ((filename (format nil "~A/train/cat.~A.jpg" *data-directory* number)))
      (read-jpeg-file filename h w))))

(defun read-train-dog-file (number &optional (h 64) (w 64))
  (when (and (>= number 0) (<= number 10000))
    (let ((filename (format nil "~A/train/dog.~A.jpg" *data-directory* number)))
      (read-jpeg-file filename h w))))

(defparameter *output-directory* "/Users/Sungjin/Desktop")

(defun write-rgb-png-file (tensor filename)
  (let* ((height ($size tensor 1))
         (width ($size tensor 2))
         (img (opticl:make-8-bit-rgb-image height width)))
    (loop :for h :from 0 :below height
          :do (loop :for w :from 0 :below width
                    :do (setf (aref img h w 0) (round (* 255 ($ tensor 0 h w)))
                              (aref img h w 1) (round (* 255 ($ tensor 1 h w)))
                              (aref img h w 2) (round (* 255 ($ tensor 2 h w))))))
    (opticl:write-png-file (format nil "~A/~A" *output-directory* filename) img)))

(defun write-gray-png-file (tensor filename &optional (channel 0))
  (let* ((height ($size tensor 1))
         (width ($size tensor 2))
         (img (opticl:make-8-bit-gray-image height width)))
    (loop :for h :from 0 :below height
          :do (loop :for w :from 0 :below width
                    :do (setf (aref img h w) (round (* 255 ($ tensor channel h w))))))
    (opticl:write-png-file (format nil "~A/~A" *output-directory* filename) img)))


(write-rgb-png-file (read-train-cat-file 10 64 64) "cat10.png")
(write-gray-png-file (read-train-dog-file 20 64 64) "dog20.png")
