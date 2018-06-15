(defpackage :gdl-ch11-2
  (:use #:common-lisp
        #:mu
        #:th
        #:th.db.imdb))

(in-package :gdl-ch11-2)

(defun process-review (review)
  (remove-duplicates (->> (remove-duplicates (split #\space review) :test #'equal)
                          (mapcar (lambda (w)
                                    (cl-ppcre:regex-replace-all
                                     "[^a-z0-9A-Z]"
                                     (string-downcase w)
                                     "")))
                          (remove-if-not (lambda (w) (> ($count w) 0))))
                     :test #'equal))

(defparameter *imdb* (read-imdb-data2))
(defparameter *reviews* (->> ($ *imdb* :reviews)
                             (mapcar #'process-review)))
(defparameter *labels* (->> ($ *imdb* :labels)
                            (mapcar (lambda (s) (if (equal s "positive") 1 0)))))

(defparameter *words* (remove-duplicates (apply #'$concat *reviews*) :test #'equal))
(defparameter *w2i* (let ((h (make-hash-table :test 'equal :size ($count *words*))))
                      (loop :for i :from 0 :below ($count *words*)
                            :for w = ($ *words* i)
                            :do (setf ($ h w) i))
                      h))

(defun review-to-indices (review-words)
  (sort (remove-duplicates (->> review-words
                                (mapcar (lambda (w) ($ *w2i* w)))
                                (remove-if (lambda (w) (null w))))
                           :test #'equal)
        #'<))

;; train dataset
(defparameter *train-dataset* (mapcar #'review-to-indices
                                      (subseq *reviews* 0 (- ($count *reviews*) 1000))))
(defparameter *train-targets* (tensor (subseq *labels* 0 (- ($count *labels*) 1000))))

;; test dataset
(defparameter *test-dataset* (mapcar #'review-to-indices
                                     (subseq *reviews* (- ($count *reviews*) 1000))))
(defparameter *test-targets* (tensor (subseq *labels* (- ($count *labels*) 1000))))

;; neural network
(defparameter *alpha* 0.01)
(defparameter *iterations* 4)
(defparameter *hidden-size* 100)

(defparameter *w01* ($- ($* 0.2 (rnd ($count *words*) *hidden-size*)) 0.1))
(defparameter *w12* ($- ($* 0.2 (rnd *hidden-size* 1)) 0.1))

;; prediction utility function
(defun predict-sentiment (x)
  (let* ((w01 ($index *w01* 0 x))
         (l1 (-> ($sum w01 0)
                 ($sigmoid!)))
         (l2 (-> ($dot l1 *w12*)
                 ($sigmoid!))))
    l2))

;; print test stats
(defun print-test-perf ()
  (let ((total 0)
        (correct 0))
    (loop :for i :from 0 :below (min 1000 ($count *test-dataset*))
          :for x = ($ *test-dataset* i)
          :for y = ($ *test-targets* i)
          :do (let ((s (predict-sentiment x)))
                (incf total)
                (when (< (abs (- s y)) 0.5)
                  (incf correct))))
    (prn "=>" total correct)))

(defun train (&optional (niter *iterations*))
  (loop :for iter :from 1 :to niter
        :do (let ((total 0)
                  (correct 0))
              (loop :for i :from 0 :below ($count *train-dataset*)
                    :for x = ($ *train-dataset* i)
                    :for y = ($ *train-targets* i)
                    :for w01 = ($index *w01* 0 x)
                    :for l1 = (-> ($sum w01 0)
                                  ($sigmoid))
                    :for l2 = (-> ($dot l1 *w12*)
                                  ($sigmoid))
                    :for dl2 = ($sub l2 y)
                    :for dl1 = ($* dl2 ($transpose *w12*))
                    :do (let ((d1 ($mul! dl1 *alpha*))
                              (d2 ($mul! l1 (* dl2 *alpha*))))
                          (setf ($index *w01* 0 x)
                                ($sub! w01 ($expand! d1 ($size w01))))
                          ($sub! *w12* d2)
                          (incf total)
                          (when (< (abs dl2) 0.5)
                            (incf correct))))
              (when (zerop (rem iter 1))
                (prn iter total correct)
                (print-test-perf)
                (gcf)))))

;; execute training
(train)

;; personal test to check the network really works
(let* ((my-review "this so called franchise movie of avengers is great master piece. i've enjoyed it very much and my kids love this one as well. though my wife generally does not like this kind of genre, she said this one is better than others.")
       (review (process-review my-review))
       (x (review-to-indices review)))
  (print x)
  (print (predict-sentiment x)))

(let* ((my-review "this movie is just a political propaganda, it has neither entertainment or message. i just regret my spending of precious time on this one.")
       (review (process-review my-review))
       (x (review-to-indices review)))
  (print x)
  (print (predict-sentiment x)))

;; what hidden layer learns
(defun similar (word)
  (let ((target-index ($ *w2i* word)))
    (when target-index
      (let ((weight-target ($ *w01* target-index))
            (scores nil))
        (loop :for w :in *words*
              :for weight = ($ *w01* ($ *w2i* w))
              :for difference = ($sub weight weight-target)
              :for wdiff = ($dot difference difference)
              :do (let ((score (sqrt wdiff)))
                    (push (cons w score) scores)))
        (subseq (sort scores (lambda (a b) (< (cdr a) (cdr b)))) 0 (min 10 ($count scores)))))))

(print (similar "beautiful"))
(print (similar "terrible"))
