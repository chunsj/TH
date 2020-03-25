;; from
;; http://karpathy.github.io/2015/05/21/rnn-effectiveness/

(defpackage :genchars-basic
  (:use #:common-lisp
        #:mu
        #:th
        #:th.layers
        #:th.ex.data))

(in-package :genchars-basic)

(defparameter *data-lines* (remove-if (lambda (line) (< ($count line) 1)) (text-lines :pg)))
(defparameter *data* (format nil "~{~A~^~%~}" *data-lines*))
(defparameter *chars* ($array (remove-duplicates (coerce *data* 'list))))
(defparameter *data-size* ($count *data*))
(defparameter *vocab-size* ($count *chars*))

(defparameter *char-to-idx* (let ((ht #{}))
                              (loop :for i :from 0 :below *vocab-size*
                                    :for ch = ($ *chars* i)
                                    :do (setf ($ ht ch) i))
                              ht))
(defparameter *idx-to-char* *chars*)

(defun choose (probs)
  "select one of the index by their given probabilities"
  (let ((probs ($div probs ($sum probs))))
    ($ ($reshape! ($multinomial probs 1) ($count probs)) 0)))

(defun outps (h wy by &optional (temperature 1) ones)
  (-> ($affine h wy by ones)
      ($/ temperature)
      ($softmax)))

(defun next-idx (h wy by &optional (temperature 1) ones)
  (choose (outps h wy by temperature ones)))

;;
;; building rnn with cells and layers
;;

(defparameter *hidden-size* 100)
(defparameter *sequence-length* 50)

(defclass affine-cell (th.layers::layer)
  ((wx :initform nil)
   (wh :initform nil)
   (a :initform nil)
   (bh :initform nil)
   (ph :initform nil :accessor $cell-state)
   (os :initform #{})
   (wi :initform nil)))

(defun affine-cell (input-size output-size
                    &key (activation :tanh) (weight-initializer :he-normal)
                      weight-initialization (biasp t))
  (let ((n (make-instance 'affine-cell)))
    (with-slots (wx wh bh ph wi a) n
      (setf wi weight-initialization)
      (setf a (th.layers::afn activation))
      (when biasp (setf bh ($parameter (zeros output-size))))
      (setf wx (th.layers::wif weight-initializer (list input-size output-size)
                               weight-initialization))
      (setf wh (th.layers::wif weight-initializer (list output-size output-size)
                               weight-initialization)))

    n))

(defmethod $train-parameters ((l affine-cell))
  (with-slots (wx wh bh) l
    (if bh (list wx wh bh) (list wx wh))))

(defmethod $parameters ((l affine-cell))
  (with-slots (wx wh bh) l
    (if bh (list wx wh bh) (list wx wh))))

(defun affine-ones (l x)
  (when (eq 2 ($ndim x))
    (with-slots (os) l
      (let* ((n ($size x 0))
             (o ($ os n)))
        (unless o
          (setf ($ os n) (ones n))
          (setf o ($ os n)))
        o))))

(defmethod $execute ((l affine-cell) x &key (trainp t))
  (with-slots (wx wh bh ph a) l
    (let ((ones (affine-ones l x))
          (ph0 (if ph ph (zeros ($size x 0) *hidden-size*)))
          (bh0 (when bh ($data bh))))
      (let ((ph1 (if a
                     (if trainp
                         (funcall a ($affine2 x wx ph0 wh bh ones))
                         (funcall a ($affine2 x ($data wx) ph0 ($data wh) bh0 ones)))
                     (if trainp
                         ($affine2 x wx ph0 wh bh ones)
                         ($affine2 x ($data wx) ph0 ($data wh) bh0 ones)))))
        (setf ph ph1)))))

(defclass rnn-layer (th.layers::layer)
  ((stateful :initform nil :accessor $recurrent-statefule-p)
   (cell :initform nil)))

(defun rnn-layer (input-size output-size
                  &key (cellfn #'affine-cell) (activation :tanh) (weight-initializer :he-normal)
                    weight-initialization (biasp t) statefulp)
  (let ((n (make-instance 'rnn-layer)))
    (with-slots (stateful cell) n
      (setf stateful statefulp)
      (setf cell (funcall cellfn
                          input-size output-size
                          :activation activation
                          :weight-initializer weight-initializer
                          :weight-initialization weight-initialization
                          :biasp biasp)))
    n))

(defmethod $train-parameters ((l rnn-layer))
  (with-slots (cell) l
    ($train-parameters cell)))

(defmethod $parameters ((l rnn-layer))
  (with-slots (cell) l
    ($parameters cell)))

(defmethod $execute ((l rnn-layer) xs &key (trainp t))
  (with-slots (cell stateful) l
    (unless stateful (setf ($cell-state cell) nil))
    (loop :for x :in xs
          :collect ($execute cell x :trainp trainp))))

(defclass character-encoder ()
  ((char-to-idx :initform #{})
   (idx-to-char :initform nil)))

(defun character-encoder () (make-instance 'character-encoder))

(defun build-encoder (encoder data)
  (with-slots (char-to-idx idx-to-char) encoder
    (setf idx-to-char ($array (remove-duplicates (coerce data 'list))))
    (let ((vocab-size ($count idx-to-char)))
      (loop :for i :from 0 :below vocab-size
            :for ch = ($ idx-to-char i)
            :do (setf ($ char-to-idx ch) i))))
  encoder)

(defun encoder-vocab-size (encoder)
  (with-slots (idx-to-char) encoder
    ($count idx-to-char)))

(defun encoder-encode-strings-1-of-k (encoder strings)
  "encode strings as 1-of-K encodings"
  (let ((ntime ($count ($0 strings)))
        (nbatch ($count strings))
        (vocab-size (encoder-vocab-size encoder)))
    (with-slots (char-to-idx) encoder
      (loop :for time :from 0 :below ntime
            :collect (let ((m (zeros nbatch vocab-size)))
                       (loop :for str :in strings
                             :for b :from 0
                             :for ch = ($ str time)
                             :do (setf ($ m b ($ char-to-idx ch)) 1))
                       m)))))

(defun encoder-decode-matrices-1-of-k (encoder matrics)
  "decode matrics as strings")

(let ((encoder (character-encoder)))
  (build-encoder encoder *data*)
  (prn (encoder-encode-strings-1-of-k encoder '("hello, world"))))

;; XXX need to build encoding/decoding helper
;; first, with character encoder/decoder
;; input: strings
;; output: encoder/decoder instance
;; contains: char-to-idx, idx-to-char information
;; offers indices encoding/decoding
;; offers 1-of-K encoding/decoding
;; represents sequence as a list
;; takes a list input of indices/1-of-K matrices
;; supports batch encoding/decoding

(prn (affine-cell *vocab-size* *hidden-size*))

(prn (to-1-of-k "hello, world."))
(prn (to-1-of-ks '("hello, world." "hello, world.")))

(let ((rnn (rnn-layer *vocab-size* *hidden-size*))
      (xs (to-1-of-ks '("hello, world." "hello, world."))))
  (prn ($execute rnn xs)))

;;
;; vanilla rnn
;;

(defparameter *hidden-size* 100)
(defparameter *sequence-length* 50)

(defparameter *rnn* (parameters))
(defparameter *wx* ($push *rnn* ($* 0.01 (rndn *vocab-size* *hidden-size*))))
(defparameter *wh* ($push *rnn* ($* 0.01 (rndn *hidden-size* *hidden-size*))))
(defparameter *bh* ($push *rnn* (zeros *hidden-size*)))
(defparameter *wy* ($push *rnn* ($* 0.01 (rndn *hidden-size* *vocab-size*))))
(defparameter *by* ($push *rnn* (zeros *vocab-size*)))

(defun rnn-write-weight-to (w fname)
  (let ((f (file.disk fname "w")))
    (setf ($fbinaryp f) t)
    ($fwrite ($data w) f)
    ($fclose f)))

(defun rnn-read-weight-from (w fname)
  (let ((f (file.disk fname "r")))
    (setf ($fbinaryp f) t)
    ($fread ($data w) f)
    ($fclose f)))

(defun rnn-write-weights ()
  (rnn-write-weight-to *wx* "examples/weights/genchar/rnn-wx.dat")
  (rnn-write-weight-to *wh* "examples/weights/genchar/rnn-wh.dat")
  (rnn-write-weight-to *wy* "examples/weights/genchar/rnn-wy.dat")
  (rnn-write-weight-to *bh* "examples/weights/genchar/rnn-bh.dat")
  (rnn-write-weight-to *by* "examples/weights/genchar/rnn-by.dat"))

(defun rnn-read-weights ()
  (rnn-read-weight-from *wx* "examples/weights/genchar/rnn-wx.dat")
  (rnn-read-weight-from *wh* "examples/weights/genchar/rnn-wh.dat")
  (rnn-read-weight-from *wy* "examples/weights/genchar/rnn-wy.dat")
  (rnn-read-weight-from *bh* "examples/weights/genchar/rnn-bh.dat")
  (rnn-read-weight-from *by* "examples/weights/genchar/rnn-by.dat"))

(defun to-indices (str)
  "string to indices"
  (loop :for i :from 0 :below ($count str)
        :for ch = ($ str i)
        :collect ($ *char-to-idx* ch)))

(defun to-1-of-k (str)
  "string to 1-of-K encoded matrix"
  (let ((m (zeros ($count str) *vocab-size*)))
    (loop :for i :from 0 :below ($count str)
          :for ch = ($ str i)
          :do (setf ($ m i ($ *char-to-idx* ch)) 1))
    m))

(defun to-string (indices) (coerce (mapcar (lambda (i) ($ *idx-to-char* i)) indices) 'string))

(prn (to-indices "hello, world."))

(defclass rnncell-layer (th.layers::layer)
  ((wx :initform nil)
   (wh :initform nil)
   (a :initform nil)
   (bh :initform nil)
   (ph :initform nil :reader rnncell-state)
   (os :initform #{})
   (wi :initform nil)))

(defun rnncell-layer (input-size output-size
                      &key (activation :tanh) (weight-initializer :he-normal)
                        weight-initialization (biasp t))
  (let ((n (make-instance 'rnncell-layer)))
    (with-slots (wx wh bh ph wi a) n
      (setf wi weight-initialization)
      (setf a (th.layers::afn activation))
      (when biasp (setf bh ($parameter (zeros output-size))))
      (setf wx (th.layers::wif weight-initializer (list input-size output-size)
                               weight-initialization)))
    n))

(defmethod $train-parameters ((l rnncell-layer))
  (with-slots (wx wh bh) l
    (if bh (list wx wh bh) (list wx wh))))

(defmethod $parameters ((l rnncell-layer))
  (with-slots (wx wh bh) l
    (if bh (list wx wh bh) (list wx wh))))

(defmethod $execute ((l rnn-layer) x &key (trainp t))
  (with-slots (wx wh bh ph a) l
    (let ((ones (th.layers::affine-ones l x))
          (ph0 (if ph ph (zeros ($size x 0) *hidden-size*)))
          (bh0 (when bh ($data bh))))
      (let ((ph1 (if a
                     (if trainp
                         (funcall a ($affine2 x wx ph0 wh bh ones))
                         (funcall a ($affine2 x ($data wx) ph0 ($data wh) bh0 ones)))
                     (if trainp
                         ($affine2 x wx ph0 wh bh ones)
                         ($affine2 x ($data wx) ph0 ($data wh) bh0 ones)))))
        (setf ph ph1)))))

(prn (rnncell-layer *vocab-size* *hidden-size*))

(prn *wx*)
(prn ($wimb (to-indices "hello, world.") ($data *wx*)))
(prn ($sum ($affine (to-1-of-k "hello, world.") ($data *wx*) nil) 0))

;; XXX
;; create recurrent-layer which can use cell
;; axes for indices are in order of sequence-batch-feature.
;;

(defun rnni (xi ph wx wh b &optional ones)
  (let ((xp ($index wx 0 xi))
        (hp ($affine ph wh b ones)))
    ($tanh ($+ xp hp))))

(defun rnne (x ph wx wh b &optional ones)
  ($tanh ($affine2 x wx ph wh b ones)))

(let* ((str "hello, world.")
       (xi (to-indices str))
       (xe (to-1-of-k str))
       (ph ($* 0.001 (ones 1 *hidden-size*)))
       (wx ($data *wx*))
       (wh ($data *wh*))
       (bh ($data *bh*)))
  (prn (rnni xi ph wx wh bh))
  ;;(prn (rnne xe ph wx wh bh))
  )

(prn ($index ($data *wx*) 0 '(8 8 8)))

(defun seedhi (str &optional (temperature 1))
  (let ((input (to-indices str))
        (ph (zeros 1 *hidden-size*))
        (wx ($data *wx*))
        (wh ($data *wh*))
        (bh ($data *bh*))
        (wy ($data *wy*))
        (by ($data *by*))
        (ncidx 0))
    (loop :for xti :in input
          :for ht = (rnni xti ph wx wh bh)
          :for nidx = (next-idx ht wy by temperature)
          :do (setf ph ht
                    ncidx nidx))
    (cons ncidx ph)))

(prn (seedhi "hello"))

(defun sample (str n &optional (temperature 1))
  (let ((x (zeros 1 *vocab-size*))
        (indices nil)
        (sh (when str (seedhi str temperature)))
        (ph nil))
    (if sh
        (let ((idx0 (car sh))
              (h (cdr sh)))
          (setf ($ x 0 idx0) 1)
          (setf ph h)
          (push idx0 indices))
        (let ((idx0 (random *vocab-size*))
              (h (zeros 1 *hidden-size*)))
          (setf ($ x 0 idx0) 1)
          (setf ph h)
          (push idx0 indices)))
    (loop :for i :from 0 :below n
          :for ht = ($rnn x ph wx wh bh)
          :for nidx = (next-idx ht wy by temperature)
          :do (progn
                (setf ph ht)
                (push nidx indices)
                ($zero! x)
                (setf ($ x 0 nidx) 1)))
    (prn (to-string (reverse indices)))
    (concatenate 'string str (to-string (reverse indices)))))

(defun seedh (str &optional (temperature 1))
  (let ((input (to-1-of-k str))
        (ph (zeros 1 *hidden-size*))
        (wx ($data *wx*))
        (wh ($data *wh*))
        (bh ($data *bh*))
        (wy ($data *wy*))
        (by ($data *by*))
        (ncidx 0))
    (loop :for i :from 0 :below ($size input 0)
          :for xt = ($index input 0 i)
          :for ht = ($rnn xt ph wx wh bh)
          :for nidx = (next-idx ht wy by temperature)
          :do (setf ph ht
                    ncidx nidx))
    (cons ncidx ph)))

(defun sample (str n &optional (temperature 1))
  (let ((x (zeros 1 *vocab-size*))
        (indices nil)
        (sh (when str (seedh str temperature)))
        (wx ($data *wx*))
        (wh ($data *wh*))
        (bh ($data *bh*))
        (wy ($data *wy*))
        (by ($data *by*))
        (ph nil))
    (if sh
        (let ((idx0 (car sh))
              (h (cdr sh)))
          (setf ($ x 0 idx0) 1)
          (setf ph h)
          (push idx0 indices))
        (let ((idx0 (random *vocab-size*))
              (h (zeros 1 *hidden-size*)))
          (setf ($ x 0 idx0) 1)
          (setf ph h)
          (push idx0 indices)))
    (loop :for i :from 0 :below n
          :for ht = ($rnn x ph wx wh bh)
          :for nidx = (next-idx ht wy by temperature)
          :do (progn
                (setf ph ht)
                (push nidx indices)
                ($zero! x)
                (setf ($ x 0 nidx) 1)))
    (prn (to-string (reverse indices)))
    (concatenate 'string str (to-string (reverse indices)))))

(defparameter *upto* (- *data-size* *sequence-length* 1))

;; XXX of course, we need better strategy for building data
;; for example, breaking at the word level will be better one.
(defparameter *inputs* (loop :for p :from 0 :below *upto* :by *sequence-length*
                             :for input-str = (subseq *data* p (+ p *sequence-length*))
                             :collect (let ((m (zeros *sequence-length* *vocab-size*)))
                                        (loop :for i :from 0 :below *sequence-length*
                                              :for ch = ($ input-str i)
                                              :do (setf ($ m i ($ *char-to-idx* ch)) 1))
                                        m)))
(defparameter *targets* (loop :for p :from 0 :below *upto* :by *sequence-length*
                              :for target-str = (subseq *data* (1+ p) (+ p *sequence-length* 1))
                              :collect (let ((m (zeros *sequence-length* *vocab-size*)))
                                         (loop :for i :from 0 :below *sequence-length*
                                               :for ch = ($ target-str i)
                                               :do (setf ($ m i ($ *char-to-idx* ch)) 1))
                                         m)))

(defparameter *mloss* (* (- (log (/ 1 *vocab-size*))) *sequence-length*))
(defparameter *min-mloss* *mloss*)

(defparameter *niters* 10)

($cg! *rnn*)
(gcf)

(setf *niters* 1)

(time
 (with-foreign-memory-limit ()
   (loop :for iter :from 1 :to *niters*
         :for n = 0
         :for maxloss = 0
         :for maxloss-pos = -1
         :for max-mloss = 0
         :do (progn
               (loop :for input :in *inputs*
                     :for target :in *targets*
                     :for bidx :from 0 :to 0
                     :do (let ((ph (zeros 1 *hidden-size*))
                               (tloss 0))
                           (loop :for i :from 0 :below ($size input 0)
                                 :for xt = ($index input 0 i)
                                 :for ht = ($rnn xt ph *wx* *wh* *bh*)
                                 :for ps = (outps ht *wy* *by*)
                                 :for y = ($index target 0 i)
                                 :for l = ($cee ps y)
                                 :do (progn
                                       (setf ph ht)
                                       (incf tloss ($data l))))
                           (when (> tloss maxloss)
                             (setf maxloss-pos n)
                             (setf maxloss tloss))
                           ($rmgd! *rnn*)
                           (setf *mloss* (+ (* 0.999 *mloss*) (* 0.001 tloss)))
                           (when (> *mloss* max-mloss) (setf max-mloss *mloss*))
                           (when (zerop (rem n 200))
                             (prn "[ITER]" iter n *mloss* maxloss maxloss-pos))
                           (incf n)))
               (when (< max-mloss *min-mloss*)
                 (prn "*** BETTER MLOSS - WRITE WEIGHTS: FROM" *min-mloss* "TO" max-mloss)
                 (setf *min-mloss* max-mloss)
                 (rnn-write-weights))))))

(prn (sample "This is not correct." 200 0.5))
(prn (sample "I" 200 0.5))

;;(rnn-write-weights)
(rnn-read-weights)

;; rmgd 0.002 0.99 -  1.31868 - 1.61637
;; adgd - 1.551497 - 1.841827
;; amgd 0.002 - 1.3747485 - 1.70623

(loop :for p :from 0 :below *upto* :by *sequence-length*
      :for n :from 0
      :for input-str = (subseq *data* p (+ p *sequence-length*))
      :do (when (member n '(16260 11637 7640 7615 5552 1076 290 232 192 162 87))
            (prn (format nil "~6,d" n) input-str)))
