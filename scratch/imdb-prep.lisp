(defpackage :imdb-prep
  (:use #:common-lisp
        #:mu))

(in-package :imdb-prep)

(with-open-file (stream "/Users/me/IMDB/imdb_master.csv" :direction :input)
  (with-open-file (frv "/Users/me/IMDB/reviews.txt" :direction :output
                                                         :if-exists :supersede)
    (with-open-file (flbl "/Users/me/IMDB/labels.txt" :direction :output
                                                           :if-exists :supersede)
      (loop :for i :from 0 :to 100000
            :for line = (read-line stream nil)
            :do (when (and line (> i 0))
                  (let* ((scmps (split #\, line))
                         (n ($count scmps))
                         (type ($ scmps 1))
                         (lbl ($ scmps (- n 2)))
                         (review (cl-ppcre:regex-replace-all
                                  "\\<br /\\>"
                                  (->> (apply #'strcat (subseq scmps 2 (- n 2)))
                                       (remove #\"))
                                  "")))
                    (format frv "~A~%" review)
                    (format flbl "~A~%" lbl)))))))

;; make a smaller dataset
(defparameter *small-size* 2000)
(with-open-file (stream "/Users/me/IMDB/reviews.txt" :direction :input)
  (with-open-file (frvs "/Users/me/IMDB/reviews-small.txt" :direction :output
                                                                :if-exists :supersede)
    (loop :for i :from 0 :to (+ 12500 *small-size*)
          :for line = (read-line stream nil)
          :do (when (and line (or (< i *small-size*) (> i 12500)))
                (format frvs "~A~%" line)))))

(with-open-file (stream "/Users/me/IMDB/labels.txt" :direction :input)
  (with-open-file (frvs "/Users/me/IMDB/labels-small.txt" :direction :output
                                                               :if-exists :supersede)
    (loop :for i :from 0 :to (+ 12500 *small-size*)
          :for line = (read-line stream nil)
          :do (when (and line (or (< i *small-size*) (> i 12500)))
                (format frvs "~A~%" line)))))

;; a smaller test dataset as well
(defparameter *small-test-size* 500)
(with-open-file (stream "/Users/me/IMDB/reviews.txt" :direction :input)
  (with-open-file (frvs "/Users/me/IMDB/reviews-small-test.txt" :direction :output
                                                                     :if-exists :supersede)
    (loop :for i :from 0 :to (+ 12500 *small-size* *small-test-size*)
          :for line = (read-line stream nil)
          :do (when (and line (or (and (>= i *small-size*) (< i (+ *small-size*
                                                                  *small-test-size*)))
                                  (and (> i (+ 12500 *small-size*)))))
                (format frvs "~A~%" line)))))

(with-open-file (stream "/Users/me/IMDB/labels.txt" :direction :input)
  (with-open-file (frvs "/Users/me/IMDB/labels-small-test.txt" :direction :output
                                                                    :if-exists :supersede)
    (loop :for i :from 0 :to (+ 12500 *small-size* *small-test-size*)
          :for line = (read-line stream nil)
          :do (when (and line (or (and (>= i *small-size*) (< i (+ *small-size*
                                                                  *small-test-size*)))
                                  (and (> i (+ 12500 *small-size*)))))
                (format frvs "~A~%" line)))))
