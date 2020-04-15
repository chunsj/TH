(declaim (optimize (speed 3) (debug 0) (safety 0)))

(in-package :th)

(cffi:defcfun ("THSetNumThreads" th-set-num-threads) :void (n :int))
(cffi:defcfun ("THGetNumThreads" th-get-num-threads) :int)

;; macbook 12 2017 - default is 4 but 1 shows better performance
(th-set-num-threads 1)
