(defpackage :imdb-prep
  (:use #:common-lisp
        #:mu))

(in-package :imdb-prep)

(with-open-file (stream "/Users/Sungjin/IMDB/imdb_master.csv" :direction :input)
  (with-open-file (frv "/Users/Sungjin/IMDB/reviews.txt" :direction :output
                                                         :if-exists :supersede)
    (with-open-file (flbl "/Users/Sungjin/IMDB/labels.txt" :direction :output
                                                           :if-exists :supersede)
      (loop :for i :from 0 :below 100000
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
