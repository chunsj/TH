(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defun load-native-library ()
  (let ((stype (software-type)))
    (cond ((string-equal "Darwin" stype) (cffi:load-foreign-library "libATen.dylib"))
          ((string-equal "Linux" stype) (cffi:load-foreign-library "/usr/local/lib/libATen.so")))))

(load-native-library)
