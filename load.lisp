(declaim (optimize (speed 3) (debug 1) (safety 0)))

(in-package :th)

(defun load-native-library ()
  (let ((stype (software-type)))
    (cond ((string-equal "Darwin" stype)
           (progn
             (cffi:load-foreign-library "libTHTensor.dylib")
             (cffi:load-foreign-library "libTHNeural.dylib")))
          ((string-equal "Linux" stype)
           (progn
             (cffi:load-foreign-library "/usr/local/lib/libTHTensor.so")
             (cffi:load-foreign-library "/usr/local/lib/libTHNNeural.so"))))))

(load-native-library)
