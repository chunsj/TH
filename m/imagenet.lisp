(defpackage th.m.imagenet
  (:use #:common-lisp
        #:mu
        #:th)
  (:export #:imagenet-categories
           #:imagenet-input
           #:imagenet-top5-matches))

(in-package th.m.imagenet)

(defparameter +model-location+ ($concat (namestring (user-homedir-pathname)) ".th.models"))

(defun imagenet-categories ()
  (with-open-file (in (format nil "~A/imagenet/categories.txt" +model-location+) :direction :input)
    (coerce (loop :for i :from 0 :below 1000
                  :for line = (read-line in nil)
                  :for catn = (subseq line 0 9)
                  :for desc = (subseq line 10)
                  :collect (list catn desc))
            'vector)))

;; to use torch vision weight
;; All pre-trained models expect input images normalized in the same way,
;; i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
;; where H and W are expected to be at least 224.
;; The images have to be loaded in to a range of [0, 1] and then normalized using
;; normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
;;                                  std=[0.229, 0.224, 0.225])
(defun imagenet-input (rgb-8bit-3channel-tensor)
  (let ((x rgb-8bit-3channel-tensor)
        (c 3)
        (w 224)
        (h 224))
    (when (and x (eq 3 ($ndim x)) (eq c ($size x 0)) (eq h ($size x 1)) (eq w ($size x 2)))
      (let ((input ($resize! ($empty x) (list c h w))))
        (setf ($ input 0) ($/ ($- ($ x 0) 0.485) 0.229))
        (setf ($ input 1) ($/ ($- ($ x 1) 0.456) 0.224))
        (setf ($ input 2) ($/ ($- ($ x 2) 0.406) 0.225))
        input))))

(defun imagenet-top5-matches (result)
  (let* ((sorted-val-idx ($sort result 1 t))
         (vals ($subview (car sorted-val-idx) 0 1 0 5))
         (indices ($subview (cadr sorted-val-idx) 0 1 0 5)))
    (loop :for i :from 0 :below 5
          :for val = ($ vals 0 i)
          :for idx = ($ indices 0 i)
          :collect (cons ($ (imagenet-categories) idx) val))))
