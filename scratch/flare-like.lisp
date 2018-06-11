(defpackage :flare-like
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :flare-like)

(let* ((x ($constant '(1 1)))
       (M ($constant '((1 2) (3 4) (5 6))))
       (z ($mv M x)))
  (print z)
  (print ($data z))
  (print ($size z)))

;; flare - https://github.com/aria42/flare
;; flare is written in clojure and which is the thing i'd like to do in common lisp, it seems.
;; should i refer flare and follow its design if it's better? XXX
;; i think with CLOS, common lisp can build a better flare/torch.
