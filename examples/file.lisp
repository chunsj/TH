(defpackage th.file-examples
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :th.file-examples)

(let ((f (file.disk "thfile.dat" "w"))
      (s (storage.double '(1 2 3 4))))
  ($fwrite s f)
  ($fclose f))

(let ((f (file.disk "thfile.dat" "r"))
      (s (storage.double)))
  ($fread s f)
  ($fclose f)
  (print s))

(let ((f (file.disk "thfile.dat" "w"))
      (s (storage.double '(1 2 3 4))))
  (setf ($fbinaryp f) t)
  ($fwrite s f)
  ($fclose f))

(let ((f (file.disk "thfile.dat" "r"))
      (s (storage.double)))
  (setf ($fbinaryp f) t)
  ($fread s f)
  ($fclose f)
  (print s))

(let ((f (file.disk "thfile.dat" "w"))
      (x (tensor.double '((1 2) (3 4)))))
  ($fwrite x f)
  ($fclose f))

(let ((f (file.disk "thfile.dat" "r"))
      (x (tensor.double)))
  ($fread x f)
  ($fclose f)
  (print x))

(let ((f (file.disk "thfile.dat" "w"))
      (x (tensor.double '((1 2) (3 4)))))
  (setf ($fbinaryp f) t)
  ($fwrite x f)
  ($fclose f))

(let ((f (file.disk "thfile.dat" "r"))
      (x (tensor.double)))
  (setf ($fbinaryp f) t)
  ($fread x f)
  ($fclose f)
  (print x))

(let ((f (file.disk "thfile.dat" "w"))
      (x (tensor)))
  ($fwrite x f)
  ($fclose f))

(let ((f (file.disk "thfile.dat" "r"))
      (x (tensor)))
  ($fread x f)
  ($fclose f)
  (print x))
