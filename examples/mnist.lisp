(defpackage :mnist-example
  (:use #:common-lisp
        #:mu
        #:th
        #:th.db.mnist))

(in-package :mnist-example)

;; load mnist data, takes ~22 secs in macbook 2017
(defparameter *mnist* (read-mnist-data))

;; mnist data has following dataset
;; train-images, train-labels and test-images, test-labels
(prn *mnist*)

;; utility functions for easier, systematic building convolution data
(defun mkfilter (fn nc kw kh) (tensor fn nc kw kh))
(defun mkfbias (fn) (tensor fn))

;; network parameters
(defparameter *filter-number* 30)
(defparameter *channel-number* 1)
(defparameter *filter-width* 5)
(defparameter *filter-height* 5)
(defparameter *pool-width* 2)
(defparameter *pool-height* 2)
(defparameter *pool-stride-width* 2)
(defparameter *pool-stride-height* 2)
(defparameter *pool-out-width* 12)
(defparameter *pool-out-height* 12)
(defparameter *l2-output* 100)
(defparameter *l3-output* 10)
(defparameter *k* (-> (mkfilter *filter-number* *channel-number*
                                *filter-width* *filter-height*)
                      ($uniform! 0 0.01)
                      ($parameter)))
(defparameter *kb* (-> (mkfbias *filter-number*)
                       ($zero!)
                       ($parameter)))
(defparameter *w2* (-> (rnd (* *filter-number* *pool-out-width* *pool-out-height*)
                            *l2-output*)
                       ($mul! 0.01)
                       ($parameter)))
(defparameter *b2* (-> (zeros *l2-output*)
                       ($parameter)))
(defparameter *w3* (-> (rnd *l2-output* *l3-output*)
                       ($mul! 0.01)
                       ($parameter)))
(defparameter *b3* (-> (zeros *l3-output*)
                       ($parameter)))

;; reading/writing network weights - this example comes from dlfs follow-ups
(defun mnist-write-weight-to (w fname)
  (let ((f (file.disk fname "w")))
    ($fwrite ($data w) f)
    ($fclose f)))

(defun mnist-cnn-write-weights ()
  (mnist-write-weight-to *k* "examples/mnist-cnn-weights/mnist-cnn-k.dat")
  (mnist-write-weight-to *kb* "examples/mnist-cnn-weights/mnist-cnn-kb.dat")
  (mnist-write-weight-to *w2* "examples/mnist-cnn-weights/mnist-cnn-w2.dat")
  (mnist-write-weight-to *b2* "examples/mnist-cnn-weights/mnist-cnn-b2.dat")
  (mnist-write-weight-to *w3* "examples/mnist-cnn-weights/mnist-cnn-w3.dat")
  (mnist-write-weight-to *b3* "examples/mnist-cnn-weights/mnist-cnn-b3.dat"))

(defun mnist-read-weight-from (w fname)
  (let ((f (file.disk fname "r")))
    ($fread ($data w) f)
    ($fclose f)))

(defun mnist-cnn-read-weights ()
  (mnist-read-weight-from *k* "examples/mnist-cnn-weights/mnist-cnn-k.dat")
  (mnist-read-weight-from *kb* "examples/mnist-cnn-weights/mnist-cnn-kb.dat")
  (mnist-read-weight-from *w2* "examples/mnist-cnn-weights/mnist-cnn-w2.dat")
  (mnist-read-weight-from *b2* "examples/mnist-cnn-weights/mnist-cnn-b2.dat")
  (mnist-read-weight-from *w3* "examples/mnist-cnn-weights/mnist-cnn-w3.dat")
  (mnist-read-weight-from *b3* "examples/mnist-cnn-weights/mnist-cnn-b3.dat"))

;; x should have been reshaped before entering
(defun mnist-predict (x)
  (-> x
      ($conv2d *k* *kb*)
      ($relu)
      ($maxpool2d *pool-width* *pool-height*
                  *pool-stride-width* *pool-stride-height*)
      ($reshape ($size x 0) (* *filter-number* *pool-out-width* *pool-out-height*))
      ($xwpb *w2* *b2*)
      ($relu)
      ($xwpb *w3* *b3*)
      ($softmax)))

(defparameter *batch-size* 600)
(defparameter *batch-count* (/ ($size ($ *mnist* :train-images) 0) *batch-size*))

;; training data - uses batches for performance
(defparameter *mnist-train-image-batches*
  (loop :for i :from 0 :below *batch-count*
        :for rng = (loop :for k :from (* i *batch-size*) :below (* (1+ i) *batch-size*)
                         :collect k)
        :collect ($contiguous! ($index ($ *mnist* :train-images) 0 rng))))

(defparameter *mnist-train-label-batches*
  (loop :for i :from 0 :below *batch-count*
        :for rng = (loop :for k :from (* i *batch-size*) :below (* (1+ i) *batch-size*)
                         :collect k)
        :collect ($contiguous! ($index ($ *mnist* :train-labels) 0 rng))))

(loop :for p :in (list *k* *kb* *w2* *b2* *w3* *b3*)
      :do (th::$cg! p))

(defparameter *epoch* 25)

(gcf)

;; the actual training
(time
 (loop :for epoch :from 1 :to *epoch*
       :do (loop :for i :from 0 :below *batch-count*
                 :for xi = ($ *mnist-train-image-batches* i)
                 :for x = (-> xi
                              ($reshape ($size xi 0) *channel-number* 28 28))
                 :for y = (-> ($ *mnist-train-label-batches* i))
                 :for y* = (mnist-predict x)
                 :for loss = ($cee y* y)
                 :do (progn
                       (prn (format nil "[~A|~A]: ~A" (1+ i) epoch ($data loss)))
                       ($adgd! (list *k* *kb* *w2* *b2* *w3* *b3*))))))

(defun mnist-predict-eval (x)
  (-> x
      ($conv2d ($data *k*) ($data *kb*))
      ($relu)
      ($maxpool2d *pool-width* *pool-height*
                  *pool-stride-width* *pool-stride-height*)
      ($reshape ($size x 0) (* *filter-number* *pool-out-width* *pool-out-height*))
      ($xwpb ($data *w2*) ($data *b2*))
      ($relu)
      ($xwpb ($data *w3*) ($data *b3*))
      ($softmax)))

;; test stats
(defun mnist-test-stat (&optional verbose)
  (let ((xt ($ *mnist* :test-images))
        (yt ($ *mnist* :test-labels)))
    ($count (loop :for i :from 0 :below ($size xt 0)
                  :for xi = ($index xt 0 (list i))
                  :for yi = ($index yt 0 (list i))
                  :for yi* = (mnist-predict-eval ($reshape xi ($size xi 0) 1 28 28))
                  :for err = (let ((e ($sum ($abs ($sub ($round yi*) yi)))))
                               (when (and verbose (> e 0)) (prn (list i e)))
                               e)
                  :when (> err 0)
                    :collect i))))

;; prn test stats after training
(prn (mnist-test-stat))

;; writing/reading
(mnist-cnn-write-weights)
(mnist-cnn-read-weights)
