(ql:quickload :cl-ppcre)

(defpackage :gdl-ch11-2
  (:use #:common-lisp
        #:mu
        #:th
        #:th.db.imdb))

(in-package :gdl-ch11-2)

(defun process-review (review)
  (remove-duplicates (->> (split #\space review)
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
(defparameter *iterations* 2)
(defparameter *hidden-size* 100)

(defparameter *w01* ($- ($* 0.2 (rnd ($count *words*) *hidden-size*)) 0.1))
(defparameter *w12* ($- ($* 0.2 (rnd *hidden-size* 1)) 0.1))

(defun reset-weights ()
  (setf *w01* ($- ($* 0.2 (rnd ($count *words*) *hidden-size*)) 0.1))
  (setf *w12* ($- ($* 0.2 (rnd *hidden-size* 1)) 0.1)))

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
(reset-weights)
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

(defun tokenize-review (review)
  (->> (split #\space review)
       (mapcar (lambda (w)
                 (cl-ppcre:regex-replace-all
                  "[^a-z0-9A-Z]"
                  (string-downcase w)
                  "")))
       (remove-if-not (lambda (w) (> ($count w) 0)))))

(defparameter *review-tokens* (mapcar #'tokenize-review ($ *imdb* :reviews)))
(defparameter *vocab* (let ((counts #{}))
                        (loop :for sentence :in *review-tokens*
                              :do (loop :for word :in sentence
                                        :do (let ((pcnt ($ counts word 0)))
                                              (setf ($ counts word) (1+ pcnt)))))
                        (let ((cnts (loop :for w :in (hash-table-keys counts)
                                          :collect (cons w ($ counts w)))))
                          (coerce (->> (sort cnts (lambda (p1 p2) (> (cdr p1) (cdr p2))))
                                       (mapcar #'car))
                                  'vector))))

(defparameter *w2i* #{})
(loop :for i :from 0 :below ($count *vocab*) :do (setf ($ *w2i* ($ *vocab* i)) i))

(defparameter *concatenated* nil)
(defparameter *input-dataset* nil)
(loop :for sentence :in *review-tokens*
      :do (let ((sentence-indices nil))
            (loop :for word :in sentence
                  :do (let ((wi ($ *w2i* word)))
                        (push wi sentence-indices)
                        (push wi *concatenated*)))
            (push (reverse sentence-indices) *input-dataset*)))
(setf *concatenated* (coerce (reverse *concatenated*) 'vector))
(setf *input-dataset* (reverse *input-dataset*))

;; shuffle input-dataset
(let ((indices (tensor.int (rndperm ($count *input-dataset*)))))
  (setf *input-dataset* (loop :for i :in ($list indices)
                              :collect ($ *input-dataset* i))))
(setf *input-dataset* (coerce *input-dataset* 'vector))

(defparameter *alpha* 0.05)
(defparameter *iterations* 2)

(defparameter *hidden-size* 50)
(defparameter *window* 2)
(defparameter *negative* 5)

(defparameter *w01* ($* ($- (rnd ($count *vocab*) *hidden-size*) 0.5) 0.2))
(defparameter *w12* ($- ($* 0.2 (rnd ($count *vocab*) *hidden-size*)) 0.1))

(defun reset-weights ()
  (setf *w01* ($* ($- (rnd ($count *vocab*) *hidden-size*) 0.5) 0.2))
  (setf *w12* ($- ($* 0.2 (rnd ($count *vocab*) *hidden-size*)) 0.1)))

(defparameter *layer-2-target* (zeros (1+ *negative*)))
(setf ($ *layer-2-target* 0) 1)

(defun similar (word)
  (let ((target-index ($ *w2i* word)))
    (when target-index
      (let ((weight-target ($ *w01* target-index))
            (scores nil))
        (loop :for i :from 0 :below ($count *vocab*)
              :for w = ($ *vocab* i)
              :for index = ($ *w2i* w)
              :for weight = ($ *w01* index)
              :for difference = ($sub weight weight-target)
              :for wdiff = ($dot difference difference)
              :do (let ((score (sqrt wdiff)))
                    (push (cons w score) scores)))
        (subseq (sort scores (lambda (a b) (< (cdr a) (cdr b)))) 0 (min 10 ($count scores)))))))

(defun negsample (target &optional (negative *negative*))
  (let ((rns (loop :for k :from 0 :below negative :collect (random ($count *concatenated*)))))
    (append (list target) (mapcar (lambda (i) ($ *concatenated* i)) rns))))

(defun mkctx (review i)
  (let ((left (subseq review (max 0 (- i *window*)) i))
        (right (subseq review (1+ i) (min ($count review) (+ 1 i *window*)))))
    (append left right)))

(defun train (&optional (iterations *iterations*))
  (loop :for niter :from 0 :below iterations
        :for nci = ($count *input-dataset*)
        :do (loop :for nreview :from 0 :below nci
                  :for review = ($ *input-dataset* nreview)
                  :for nr = ($count review)
                  :do (progn
                        (loop :for i :from 0 :below nr
                              :for targetw = ($ review i)
                              :for x = (mkctx review i)
                              :for sample = (negsample targetw)
                              :for w01 = ($index *w01* 0 x)
                              :for l1 = ($resize! ($mean w01 0) (list 1 *hidden-size*))
                              :for w2s = ($index *w12* 0 sample)
                              :for l2 = ($sigmoid ($mm l1 ($transpose w2s)))
                              :for dl2 = ($sub l2 *layer-2-target*)
                              :for dl1 = ($mm dl2 w2s)
                              :do (let ((dw1 ($mul! dl1 *alpha*))
                                        (dw2 ($mul! ($vv ($resize! dl2 (list (1+ *negative*)))
                                                         ($resize! l1 (list *hidden-size*)))
                                                    *alpha*)))
                                    (setf ($index *w01* 0 x) ($sub w01 ($expand! dw1 ($size w01))))
                                    (setf ($index *w12* 0 sample) ($sub w2s dw2))))
                        (when (zerop (rem nreview 200))
                          (prn niter nreview (similar "terrible")))))))

(reset-weights)
(train)

(prn (similar "terrible"))
(prn (similar "king"))
(prn (similar "queen"))

(defun analogy (positives negatives)
  (let* ((norms (-> ($sum ($mul *w01* *w01*) 1)
                    ($resize! (list ($size *w01* 0) 1))))
         (normed-weights ($mul ($expand! norms ($size *w01*)) *w01*))
         (query-vector (zeros ($size *w01* 1)))
         (scores nil))
    (loop :for word :in positives
          :do ($add! query-vector ($ normed-weights ($ *w2i* word))))
    (loop :for word :in negatives
          :do ($sub! query-vector ($ normed-weights ($ *w2i* word))))
    (loop :for i :from 0 :below ($count *vocab*)
          :for w = ($ *vocab* i)
          :for index = ($ *w2i* w)
          :for weight = ($ *w01* index)
          :for difference = ($sub weight query-vector)
          :for wdiff = ($dot difference difference)
          :for score = (sqrt wdiff)
          :do (push (cons w score) scores))
    (-> (sort scores (lambda (a b) (< (cdr a) (cdr b))))
        (subseq  1 (min 10 ($count scores))))))

(print (analogy '("terrible" "good") '("bad")))
(print (analogy '("elizabeth" "he") '("she")))
(print (analogy '("king" "woman") '("man")))
