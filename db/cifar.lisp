(declaim (optimize (speed 3) (debug 0) (satefy 0)))

(defpackage :th.db.cifar
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :th.db.cifar)

;; XXX CIFAR 10 and 100 support
