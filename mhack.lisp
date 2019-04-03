(in-package :th)

;; XXX this should go with Allocator in TH

(defvar *mhack-foreign-memory-size* nil)
(defvar *mhack-threshold* 16)

(defun hack-gc ()
  (when (and *mhack-foreign-memory-size*
             (>= *mhack-foreign-memory-size* (* *mhack-threshold* 1024 1024)))
    (format t "***** HACK GC! ~A *****~%" *mhack*)
    (setf *mhack-foreign-memory-size* 0)
    (gc)))

(defun mhack (sz)
  (when *mhack-foreign-memory-size*
    (incf *mhack-foreign-memory-size* sz)
    (hack-gc)))

(defun dimsz (dimensions)
  (if dimensions
      (reduce #'* dimensions)
      0))

(defmacro with-foreign-hack (size-mb &body body)
  `(let ((*mhack-foreign-memory-size* 0)
         (*mhack-threshold* ,size-mb))
     (gcf)
     ,@body))
