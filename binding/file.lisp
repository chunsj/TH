(in-package :th)

(defmethod $openedp ((file file)) (eq 1 (th-file-is-opened ($handle file))))
(defmethod $quietp ((file file)) (eq 1 (th-file-is-quiet ($handle file))))
(defmethod $pendanticp ((file file)) (eq 0 (th-file-is-quiet ($handle file))))
(defmethod $readablep ((file file)) (eq 1 (th-file-is-readable ($handle file))))
(defmethod $writablep ((file file)) (eq 1 (th-file-is-writable ($handle file))))
(defmethod $binaryp ((file file)) (eq 1 (th-file-is-binary ($handle file))))
(defmethod $asciip ((file file)) (eq 0 (th-file-is-binary ($handle file))))
(defmethod $autospacingp ((file file)) (eq 1 (th-file-is-auto-spacing ($handle file))))
(defmethod $noautospacingp ((file file)) (eq 0 (th-file-is-auto-spacing ($handle file))))
(defmethod $errorp ((file file)) (eq 1 (th-file-has-error ($handle file))))

(defmethod (setf $binaryp) (value (file file))
  (if value
      (th-file-binary ($handle file))
      (th-file-ascii ($handle file)))
  file)

(defmethod (setf $asciip) (value (file file))
  (if value
      (th-file-ascii ($handle file))
      (th-file-binary ($handle file)))
  file)

(defmethod (setf $autospacingp) (value (file file))
  (if value
      (th-file-auto-spacing ($handle file))
      (th-file-no-auto-spacing ($handle file)))
  file)

(defmethod (setf $noautospacingp) (value (file file))
  (if value
      (th-file-no-auto-spacing ($handle file))
      (th-file-auto-spacing ($handle file)))
  file)

(defmethod (setf $quietp) (value (file file))
  (if value
      (th-file-quiet ($handle file))
      (th-file-pedantic ($handle file)))
  file)

(defmethod (setf $pendanticp) (value (file file))
  (if value
      (th-file-pedantic ($handle file))
      (th-file-quiet ($handle file)))
  file)

(defmethod (setf $errorp) (value (file file))
  (when value (th-file-clear-error ($handle file)))
  file)

(defmethod $readbyte ((file file)) (th-file-read-byte-scalar ($handle file)))
(defmethod $readchar ((file file)) (th-file-read-char-scalar ($handle file)))
(defmethod $readshort ((file file)) (th-file-read-short-scalar ($handle file)))
(defmethod $readint ((file file)) (th-file-read-int-scalar ($handle file)))
(defmethod $readlong ((file file)) (th-file-read-long-scalar ($handle file)))
(defmethod $readfloat ((file file)) (th-file-read-float-scalar ($handle file)))
(defmethod $readdouble ((file file)) (th-file-read-double-scalar ($handle file)))

(defmethod $readbyte! ((storage storage.byte) (file file))
  (th-file-read-byte ($handle storage) ($handle file))
  storage)
(defmethod $readchar! ((storage storage.char) (file file))
  (th-file-read-char ($handle storage) ($handle file))
  storage)
