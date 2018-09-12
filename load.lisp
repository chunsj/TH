(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defun load-native-library ()
  (let ((stype (software-type)))
    (cond ((string-equal "Darwin" stype)
           (progn
             (cffi:load-foreign-library "libTH.dylib")
             (cffi:load-foreign-library "libTHNN.dylib")))
          ((string-equal "Linux" stype)
           (progn
             (cffi:load-foreign-library "/usr/local/lib/libTH.so")
             (cffi:load-foreign-library "/usr/local/lib/libTHNN.so"))))))

(load-native-library)
