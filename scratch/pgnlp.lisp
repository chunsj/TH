(ql:quickload :drakma)
(ql:quickload :plump)
(ql:quickload :lquery)

(defpackage :practical-nlp
  (:use #:common-lisp
        #:mu
        #:th)
  (:import-from #:drakma)
  (:import-from #:plump)
  (:import-from #:lquery))

;; from https://towardsdatascience.com/a-practitioners-guide-to-natural-language-processing-part-i-processing-understanding-text-9f4abfd13e72
;; needs more library and/or code
;; needs lisp nlp library.

(in-package :practical-nlp)

(defparameter *seed-urls* '("https://inshorts.com/en/read/technology"
                            "https://inshorts.com/en/read/sports"
                            "https://inshorts.com/en/read/world"))
(defparameter *news-data* nil)

(defun rmnonch (string)
  (cl-ppcre:regex-replace-all "[^a-z0-9A-Z ]" (string-downcase string) ""))

;; prepare *news-data*
(loop :for url :in *seed-urls*
      :for category = ($last (split #\/ url))
      :for content = (plump:parse (drakma:http-request url))
      :for headlines = (->> (lquery:$ content "div.news-card span" (combine (attr :itemprop) (text)))
                            (remove-if-not (lambda (e) (equal "headline" ($0 e))))
                            (map 'list (lambda (e) (subseq e 1))))
      :for articles = (->> (lquery:$ content "div.news-card div" (combine (attr :itemprop) (text)))
                           (remove-if-not (lambda (e) (equal "articleBody" ($0 e))))
                           (map 'list (lambda (e) (subseq e 1))))
      :do (loop :for i :from 0 :below ($count headlines)
                :for headline = ($ headlines i)
                :for article = ($ articles i)
                :do (push (list category
                                (rmnonch (format nil "~{~A~^ ~}" headline))
                                (rmnonch (format nil "~{~A~^ ~}" article)))
                          *news-data*)))
(setf *news-data* (reverse *news-data*))
