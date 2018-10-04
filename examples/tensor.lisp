(defpackage th.tensor-examples
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :th.tensor-examples)

;; creates an empty tensor of default tensor class (float)
(prn (tensor))

;; creates a tensor of default tensor class type with specified sizes; elements are not initialized.
(prn (tensor 2 2))

;; creates a tensor with contents
(prn (tensor '(1 2 3 4)))
(prn (tensor '((1 2 3 4))))

;; creates a tensor with multidimensional contents
(prn (tensor '((1 2) (3 4) (5 6))))

;; creates tensor with other tensor, content storage is shared.
(let ((x (tensor '(4 3 2 1))))
  (prn "X = (4 3 2 1)")
  (prn x)
  (prn "SAME CONTENT AS X")
  (prn (tensor x)))

;; create tensor with sizes and strides, elements are not initialized.
(prn (tensor '(2 2) '(2 1)))

;; type specific construction functions; same as above.
(prn (tensor.byte '(1 2 3 4)))
(prn (tensor.char '(1 2 3 4)))
(prn (tensor.short '(1 2 3 4)))
(prn (tensor.int '(1 2 3 4)))
(prn (tensor.long '(1 2 3 4)))
(prn (tensor.float '(1 2 3 4)))
(prn (tensor.double '(1 2 3 4)))

;; clone a tensor creates a new tensor with independent storage.
(let ((x (tensor '((1 2) (3 4)))))
  (prn "'((1 2) (3 4))")
  (prn ($clone x)))

;; make a tensor with contiguously allocated memory if it is not allocated contiguously.
(prn ($contiguous! (tensor.float '((1 2 3) (4 5 6)))))

;; tensor types can be changed to each other
(prn (tensor.byte (tensor.double '((1.234 2.345) (3.456 4.567)))))
(prn (tensor.double (tensor.int '((1 2 3) (4 5 6)))))
(prn (tensor.long (tensor.float '(1.2 3.4 5.6 7.8))))

;; check whether it is tensor or not
(prn ($tensorp 0))
(prn ($tensorp (tensor)))

;; query number of dimensions
(prn ($ndim (tensor)))
(prn ($ndim (tensor 2 3 4)))

;; query tensor size
(prn ($size (tensor 2 3 4)))

;; query tensor size along given dimension
(prn ($size (tensor 2 3 4) 2))

;; size of scalar value is nil
(prn ($size 1))

;; stride of a tensor
(prn ($stride (tensor 2 3 4)))

;; stride of a tensor along dimension
(prn ($stride (tensor 2 3 4) 2))

;; stride of a nil object
(prn ($stride nil))

;; storage of a tensor; a tensor is a specific view on the storage
(let* ((x (tensor 4 5))
       (s ($storage x)))
  (loop :for i :from 0 :below ($count s)
        :do (setf ($ s i) i))
  (prn "'((0 1 2 3 4) ... (15 16 17 18 19))")
  (prn x))

;; contiguous or not
(prn ($contiguousp (tensor)))
(prn ($contiguousp (tensor 2 2 2)))

;; size comparison using size value
(prn (equal ($size (tensor 2 3)) ($size (tensor 2 3))))
(prn (equal ($size (tensor)) ($size (tensor 2 3))))
(prn (equal ($size (tensor 2 2)) '(2 2)))

;; size comparison
(prn ($sizep (tensor 2 2) (tensor 2 2)))
(prn ($sizep (tensor 2 3) (tensor)))

;; number of elements in a tensor
(prn ($count (tensor 3 3 4)))

;; query the element at the index location
(let ((x (tensor '((1 2 3) (4 5 6)))))
  (prn "'((1 2 3) (4 5 6))")
  (prn x)
  (prn "1")
  (prn ($ x 0 0)))

;; creates a new view on the same storage of the given tensor
(let ((x (tensor))
      (y (tensor '((1 2) (3 4)))))
  (prn "((1 2) (3 4))")
  (prn ($set! x y))
  (prn "T")
  (prn ($setp x y)))

;; copy elements from other tensor; both should have the same number of elements.
(let ((x (tensor '(1 2 3 4)))
      (y (tensor '((5 4) (3 2)))))
  ;; now the elements of x replaced with those of y
  ($copy! x y)
  (prn "(5 4 3 2)")
  (prn x)
  (prn "((5 4) (3 2))")
  (prn y)
  ;; new int tensor from x, then copies elements from list
  ;; new int tensor will have same size of x
  (prn "(123 234 345 456)")
  (prn ($copy! (tensor.int x) '(123 234 345 456)))
  (prn "(5 4 3 2)")
  ;; storage is not shared
  (prn x))

;; fill values
(let ((x (tensor 3 3)))
  ($fill x 123)
  (prn "((123 123 123) ... (123 123 123))")
  (prn x)
  (prn "((0 0 0) ... (0 0 0))")
  ;; mutable method
  (prn ($zero! x))
  (prn "((0 0 0) ... (0 0 0))")
  (prn x)
  (prn "((1 1 1) ... (1 1 1))")
  ;; immutable method
  (prn ($one x))
  (prn "((0 0 0) ... (0 0 0))")
  (prn x))

;; resizing a tensor allocates if more memory storage is required
;; note that $view only changes sizes or shape but without changing allocated memory.
(let ((x (tensor '((1 2 3) (3 4 5))))
      (y (tensor 3 3)))
  (prn "((1 2 3 3) (4 5 ? ?) ... (? ? ? ?)))")
  (prn ($resize! x '(4 4)))
  (prn "((1 2) (3 3))")
  (prn ($resize! x '(2 2)))
  ;; resize as y
  (prn "((1 2 3) (3 4 5) (? ? ?))")
  (prn ($resize! x y)))

;; select - choose sub tensor at index along dimension
(let ((x (tensor '((1 2 3) (4 5 6) (7 8 9)))))
  (prn "(4 5 6) - 2nd(1) row along 1st(0) dimension")
  (prn ($select x 0 1))
  (prn "(3 6 9) - 3rd(2) column along 2nd(1) dimension")
  (prn ($select x 1 2))
  (setf ($select x 0 1) '(-11 -22 -33))
  (prn "x with 2nd row changed as (-11 -22 -33)")
  (prn x))

;; subdimensional select using $
(let ((x (tensor '((1 2 3) (4 5 6) (7 8 9)))))
  (prn "1st row - (1 2 3)")
  (prn ($ x 0))
  (setf ($ x 1) '(6 5 4))
  (prn "2nd row changes as (6 5 4)")
  (prn x))

;; narrow - start and size along dimension
;; returns a new tensor or view built with narrowing from start selected as size along dimension
(let ((x (tensor 5 6)))
  ($zero! x)
  (prn "5x5 matrix filled with zeros.")
  (prn x)
  ;; along 1st(0) dimension, from 2nd row, select 3 rows, then fill it as one.
  (-> ($narrow x 0 1 3)
      ($fill! 1))
  (prn "along 1st(0) dimension, from 2nd(1) row, total 3 rows are filled with one.")
  (prn x)
  (-> ($narrow x 1 1 4)
      ($fill! 2))
  (prn "along 2nd(1) dimension, from 2nd(1) column, total 4 columns are filled with two.")
  (prn x)
  (setf ($narrow x 1 0 2) '(0 11 22 33 44 55 66 77 88 99))
  (prn "along 2nd(1) dimension, from 1st(0) column, total 2 columns are copied from list.")
  (prn x))

;; subview - multiple start and size per dimension, kind of multiple narrows
;; each pair of start and size along each dimensions
(let ((x (tensor '((1 2 3 4 5 6)
                   (2 3 4 5 6 7)
                   (3 4 5 6 7 8)
                   (4 5 6 7 8 9)
                   (0 0 0 0 0 0)))))
  (prn "5x6 matrix.")
  (prn x)
  (prn "((4 5 6) (5 6 7) (6 7 8)) - from 2nd row, 3 rows, from 3rd column, 3 columns.")
  (prn ($subview x 1 3 2 3))
  (setf ($subview x 1 3 2 3) '(11 12 13 14 15 16 17 18 19))
  (prn "((11 12 13) (14 15 16) (17 18 19))")
  (prn ($subview x 1 3 2 3))
  (prn "matrix changes.")
  (prn x))

;; orderly subview using $
(let ((x (tensor '((1 2 3 4 5 6)
                   (2 3 4 5 6 7)
                   (3 4 5 6 7 8)
                   (4 5 6 7 8 9)
                   (0 0 0 0 0 0)))))
  (prn "((4 5 6) (5 6 7) (6 7 8)) - from 2nd row, 3 rows, from 3rd column, 3 columns.")
  (prn ($ x '(1 3) '(2 3)))
  (setf ($ x '(1 3) '(2 3)) '(11 12 13 14 15 16 17 18 19))
  (prn "((11 12 13) (14 15 16) (17 18 19)) in x")
  (prn x))

;; query with dimension index size pairs or subview using $
(let ((x (tensor '((1 2 3 4 5 6)
                   (2 3 4 5 6 7)
                   (3 4 5 6 7 8)
                   (4 5 6 7 8 9)
                   (0 0 0 0 0 0)))))
  (prn "original x")
  (prn x)
  (prn "subview of size 3x3, from 2nd row and 3rd column")
  (prn ($ x '(1 3) '(2 3)))
  (setf ($ x '(1 3) '(2 3)) '(11 12 13 14 15 16 17 18 19))
  (prn "3x3 subview changed as 11 to 19")
  (prn ($ x '(1 3) '(2 3)))
  (prn "changed x")
  (prn x))

;; general selection using $
(let ((x (zeros 5 6)))
  (prn "5x6 zero matrix.")
  (prn x)
  (setf ($ x 0 2) 1)
  (prn "1 at 1st row, 3rd column")
  (prn x)
  (setf ($ x 4) 9)
  (prn "5th row as 9")
  (prn x)
  (setf ($ x '(:all (5 1))) 8)
  (prn "6th column as 8")
  (prn x)
  (setf ($ x '((1 1) (1 3))) 2)
  (prn "1x3 from 2nd row and 2nd column, filled with 2.")
  (prn x)
  (setf ($ x '((0 5) (3 1))) -1)
  (prn "5x1 from 1st row and 4th column, filled with -1.")
  (prn x)
  (setf ($ x '((0 5) (1 1))) (range 1 5))
  (prn "5x1 from 1st row and 2nd column, copied from (1 ... 5)")
  (prn x)
  (setf ($ x ($lt x 0)) 5)
  (prn "element as 5 if oringinal one is less than 0.")
  (prn x))

;; index-select - creates a new storage, not sharing
;; collects subtensor along dimension at indices
(let ((x (tensor '((1 2 3) (4 5 6) (7 8 9)))))
  (prn "original x")
  (prn x)
  (prn "1st(0) and 2nd(1) along 1st(0) dimension")
  (prn ($index x 0 '(0 1)))
  (let ((y ($index x 1 '(1 2))))
    (prn "2nd(1) and 3rd(2) along 2nd(1) dimension")
    (prn y)
    ($fill! y 0)
    (prn "zero filled")
    (prn y)
    (prn "x unchanged")
    (prn x)))

;; index-copy - set copies elements into selected index location
(let ((x (tensor '((1 2 3 4) (2 3 4 5) (3 4 5 6) (4 5 6 7) (5 6 7 8))))
      (y (tensor 5 2)))
  ($fill! ($select y 1 0) -1)
  ($fill! ($select y 1 1) -2)
  (prn "original x")
  (prn x)
  (prn "original y")
  (prn y)
  (setf ($index x 1 '(3 0)) y)
  (prn "4th(3) and 1st(0) columns along 2nd(1) dimension copied from y.")
  (prn x))

;; index-fill
(let ((x (tensor '((1 2 3 4) (2 3 4 5) (3 4 5 6) (4 5 6 7) (5 6 7 8)))))
  (prn "original x")
  (prn x)
  (setf ($index x 1 '(0 3)) 123)
  (prn "1st(0) and 4th(3) columns along 2nd(1) dimension set as 123")
  (prn x))

;; gather
(let ((x (tensor 5 5)))
  (loop :for i :from 0 :below ($count x)
        :do (setf ($ ($storage x) i) i))
  (prn "original 5x5 matrix")
  (prn x)
  (prn "by incrementing index, collect nth 1st(0) dimensional values.")
  (prn "1st row will be (0 0), (1 1), (2 2), (3 3), (4 4)")
  (prn "2nd row will be (1 0), (2 1), (3 2), (4 3), (0 4)")
  (prn ($gather x 0 '((0 1 2 3 4) (1 2 3 4 0))))
  (prn "by incrementing index, collect nth 2nd(1) dimensional values.")
  (prn "1st column will be (0 0), (1 1), (2 2), (3 3), (4 4)")
  (prn "2nd column will be (1 0), (2 1), (3 2), (4 3), (0 4)")
  (prn ($gather x 1 '((0 1) (1 2) (2 3) (3 4) (4 0)))))

;; scatter
(let ((x (tensor 5 5))
      (y (tensor '((1 2 3 4 5) (-5 -4 -3 -2 -1)))))
  ($zero! x)
  (prn "5x5 zeros")
  (prn x)
  ($scatter! x 0 '((0 1 2 3 4) (1 2 3 4 0)) y)
  (prn "as in gather, but set from y")
  (prn x)
  ($scatter! x 1 '((0 1) (1 2) (2 3) (3 4) (4 0)) 9)
  (prn "as in gather, but fill a value")
  (prn x))

;; masked-select
(let ((x (tensor 3 4))
      (mask '((1 0 1 0 0 0) (1 1 0 0 0 1)))
      (z (tensor)))
  (loop :for i :from 0 :below ($count x)
        :do (setf ($ ($storage x) i) (1+ i)))
  (prn "original x")
  (prn x)
  (prn "only at value 1(true)")
  (prn ($masked x mask))
  ($set! z ($masked x mask))
  (prn "same as above.")
  (prn z)
  ($fill! z -99)
  (prn "z value changed as -99")
  (prn z)
  (prn "x unchanged.")
  (prn x))

;; masked-copy
(let ((x (tensor 3 4))
      (mask '((1 0 1 0 0 0) (1 1 0 0 0 1)))
      (z (tensor '(1 2 3 4 5))))
  ($zero! x)
  (prn "x matrix, 3x4")
  (prn x)
  (setf ($masked x mask) z)
  (prn "set by z")
  (prn x))

;; masked-fill
(let ((x (tensor 3 4))
      (mask '((1 0 1 0 0 0) (1 1 0 0 0 1))))
  ($zero! x)
  (prn "original 3x4 matrix")
  (prn x)
  (setf ($masked x mask) 5)
  (prn "filled as 5")
  (prn x))

;; nonzero - returns locations of non zero elements
(prn ($nonzero (tensor '((1 2 0 3 4) (0 0 1 0 0)))))

;; repeat - repeat content as given times
(prn ($repeat (tensor '(1 2)) 3 2))
(prn ($repeat (tensor '((1 2) (3 4) (5 6))) 2 3))

;; squeeze - removes singletone dimensions
(let ((x (tensor 2 1 2 1 2)))
  (prn "original size")
  (prn ($size x))
  (prn "no 1s")
  (prn ($size ($squeeze x)))
  (prn "no 1 in 2nd(1) dimension")
  (prn ($size ($squeeze x 1))))

;; unsqueeze - add a singleton dimension
(let ((x (tensor '(1 2 3 4))))
  (prn "vector")
  (prn x)
  (prn "along 1st(0) dimension")
  (prn ($unsqueeze x 0))
  (prn "along 2nd(1) dimension")
  (prn ($unsqueeze x 1)))

;; view - different from resize(allocation), reshape(new storage), just a view
(let ((x (tensor '(0 0 0 0))))
  (prn "original vector")
  (prn x)
  (prn "2x2")
  (prn ($view x 2 2))
  (prn "as the size of other 2x2 tensor")
  (prn ($view x (tensor 2 2))))

;; transpose
(let ((x (tensor '((1 2 3) (4 5 6)))))
  (prn "original 2x3")
  (prn x)
  (prn "transposed 3x2")
  (prn ($transpose x)))

;; transpose - shares storage but different view(tensor)
(let ((x (tensor 3 4)))
  ($zero! x)
  ($fill! ($select x 1 2) 7)
  (prn "original x")
  (prn x)
  (let ((y ($transpose x)))
    ($fill! ($select y 1 2) 8)
    (prn "modified transposed x or y")
    (prn y)
    (prn "original x")
    (prn x)))

;; permute - multidimensional transposing
(let ((x (tensor 3 4 2 5)))
  (prn "original size")
  (prn ($size x))
  (prn "permute size with 2nd, 3rd, 1st and 4th dimensions - 4,2,3,5")
  (prn ($size ($permute x 1 2 0 3))))

;; unfold - slice with size by step along dimension
(let ((x (tensor 7)))
  (loop :for i :from 1 :to 7 :do (setf ($ x (1- i)) i))
  (prn "vector, 1 to 7")
  (prn x)
  (prn "slice along 1st(0) dimension, size of 2, by step 1")
  (prn ($unfold x 0 2 1))
  (prn "slice along 1st(0) dimension, size of 2, by step 2")
  (prn ($unfold x 0 2 2)))

;; fmap - just elementwise function application
(let ((x (zeros 3 3))
      (n 0))
  ($fmap (lambda (v) (+ v (* 0.5 pi (incf n)))) x)
  (prn x)
  ($fmap (lambda (v) (round (sin v))) x)
  (prn x))

;; fmap - more, shape is irrelevant when they have same count.
(let ((x (tensor 3 3))
      (y (tensor 9))
      (z (tensor '(0 1 2 3 4 5 6 7 8))))
  (loop :for i :from 1 :to 9 :do (setf ($ ($storage x) (1- i)) i
                                       ($ ($storage y) (1- i)) i))
  (prn x)
  (prn y)
  (prn z)
  (prn "1*1, 2*2, 3*3, ...")
  (prn ($fmap (lambda (vx vy) (* vx vy)) x y))
  (prn "1 + 1 + 0, 2 + 2 + 1, 3 + 3 + 2, ...")
  (prn ($fmap (lambda (xx yy zz) (+ xx yy zz)) x y z))
  (prn x))

;; split - split tenor with size along dimension
(let ((x (zeros 3 4 5)))
  (prn "by 2 along 0 - 2x4x5, 1x4x5")
  (prn ($split x 2 0))
  (prn "by 2 along 1 - 3x2x5, 3x2x5")
  (prn ($split x 2 1))
  (prn "by 2 along 2 - 3x4x2, 3x4x2, 3x4x1")
  (prn ($split x 2 2)))

;; chunk - n parition of approaximately same size along dimension
(let ((x (ones 3 4 5)))
  (prn "2 partitions along 0 - 2x4x5, 1x4x5")
  (prn ($chunk x 2 0))
  (prn "2 partitions along 1 - 3x2x5, 3x2x5")
  (prn ($chunk x 2 1))
  (prn "2 paritions along 2 - 3x4x3, 3x4x2")
  (prn ($chunk x 2 2)))

;; concat tensors
(prn ($cat (ones 3) (zeros 3)))
(prn ($cat (ones 1 3) (zeros 1 3) 1))
(prn ($cat (ones 1 3) (zeros 1 3) 0))
(prn ($cat (ones 3 3) (zeros 1 3)))
(prn ($cat (ones 3 4) (zeros 3 2) 1))

;; diagonal matrix
(prn ($diag (tensor '(1 2 3 4))))
(prn ($diag (ones 3 3)))

;; identity matrix
(prn (eye 2))
(prn (eye 3 4))
(prn ($eye (tensor.byte) 3))
(prn ($eye (tensor.byte 10 20) 3 4))

;; linspace
(prn (linspace 1 2))
(prn (linspace 1 2 11))

;; logspace
(prn (logspace 1 2))
(prn (logspace 1 2 11))

;; uniform random
(prn (rnd 3 3))

;; normal random
(prn (rndn 2 4))

;; range
(prn (range 2 5))
(prn (range 2 5 1.2))

;; arange
(prn (arange 2 5))
(prn (range 2 5))

;; randperm
(prn (rndperm 10))
(prn (rndperm 5))

;; reshape
(let ((x (ones 2 3))
      (y nil))
  (prn x)
  (setf y ($reshape x 3 2))
  (prn y)
  ($fill! y 2)
  (prn y)
  (prn x))

;; tril and triu
(let ((x (ones 4 4)))
  (prn ($tril x))
  (prn ($tril x -1))
  (prn ($triu x))
  (prn ($triu x 1)))

;; abs
(let ((x (tensor '((-1 2) (3 -4)))))
  (prn x)
  (prn ($abs x))
  (prn x)
  (prn ($abs! x))
  (prn x))

;; sign
(let ((x (tensor '((-1 2) (3 -4)))))
  (prn x)
  (prn ($sign x))
  (prn x)
  (prn ($sign! x))
  (prn x))

;; acos
(let ((x (tensor '((-1 1) (1 -1)))))
  (prn x)
  (prn ($acos x))
  (prn x)
  (prn ($acos! x))
  (prn x))

;; asin
(let ((x (tensor '((-1 1) (1 -1)))))
  (prn x)
  (prn ($asin x))
  (prn x)
  (prn ($asin! x))
  (prn x))

;; atan
(let ((y (tensor '((-11 1) (1 -11)))))
  (prn y)
  (prn ($atan y))
  (prn y)
  (prn ($atan! y))
  (prn y))

;; atan2
(let ((y (tensor '((-11 1) (1 -11))))
      (x (tensor '((1 1) (1 1)))))
  (prn y)
  (prn ($atan2 y x))
  (prn y)
  (prn ($atan2! y x))
  (prn y))

;; ceil
(let ((x (tensor '((1 1.1 1.7) (-0.8 -1.1 -2.3)))))
  (prn x)
  (prn ($ceil x))
  (prn x)
  (prn ($ceil! x))
  (prn x))

;; cos
(let ((x (tensor '((-3.14 0) (3.14 0)))))
  (prn x)
  (prn ($cos x))
  (prn x)
  (prn ($cos! x))
  (prn x))

;; cosh
(let ((x (tensor '((-3.14 0) (3.14 0)))))
  (prn x)
  (prn ($cosh x))
  (prn x)
  (prn ($cosh! x))
  (prn x))

;; exp
(let ((x (tensor '((0 1 2) (-1 -2 -3)))))
  (prn x)
  (prn ($exp x))
  (prn x)
  (prn ($exp! x))
  (prn x))

;; floor
(let ((x (tensor '((1 1.1 1.7) (-0.8 -1.1 -2.3)))))
  (prn x)
  (prn ($floor x))
  (prn x)
  (prn ($floor! x))
  (prn x))

;; log
(let ((x ($exp (tensor '((0 1 2) (-1 -2 -3))))))
  (prn x)
  (prn ($log x))
  (prn x)
  (prn ($log! x))
  (prn x))

;; log1p
(let ((x ($exp (tensor '((0 1 2) (-1 -2 -3))))))
  (prn x)
  (prn ($log1p x))
  (prn x)
  (prn ($log1p! x))
  (prn x))

;; neg
(let ((x (tensor '((0 1 2) (-1 -2 -3)))))
  (prn x)
  (prn ($neg x))
  (prn x)
  (prn ($neg! x))
  (prn x))

;; cinv
(let ((x (tensor '((3 2 1) (-1 -2 -3)))))
  (prn x)
  (prn ($cinv x))
  (prn x)
  (prn ($cinv! x))
  (prn x))

;; expt
(let ((x (tensor '((2 3) (1 2))))
      (y (tensor '((2 2) (3 3))))
      (n 2))
  (prn x)
  (prn ($expt x n))
  (prn ($expt n x))
  (prn ($expt x y))
  (prn x)
  (prn ($expt! x n))
  (prn x)
  (prn ($expt! n x))
  (prn x))

;; round
(let ((x (tensor '((1.1 1.8) (-1.1 -1.8)))))
  (prn x)
  (prn ($round x))
  (prn x)
  (prn ($round! x))
  (prn x))

;; sin
(let ((x (tensor '((-3.14 0) (3.14 0)))))
  (prn x)
  (prn ($sin x))
  (prn x)
  (prn ($sin! x))
  (prn x))

;; sinh
(let ((x (tensor '((-3.14 0) (3.14 0)))))
  (prn x)
  (prn ($sinh x))
  (prn x)
  (prn ($sinh! x))
  (prn x))

;; sqrt
(let ((x (tensor '((1 2 4) (3 5 9)))))
  (prn x)
  (prn ($sqrt x))
  (prn x)
  (prn ($sqrt! x))
  (prn x))

;; rsqrt
(let ((x (tensor '((1 2 4) (3 5 9)))))
  (prn x)
  (prn ($rsqrt x))
  (prn x)
  (prn ($rsqrt! x))
  (prn x))

;; tan
(let ((x (tensor '((1 2) (3 4)))))
  (prn x)
  (prn ($tan x))
  (prn x)
  (prn ($tan! x))
  (prn x))

;; tanh
(let ((x (tensor '((1 2) (-3 -4)))))
  (prn x)
  (prn ($tanh x))
  (prn x)
  (prn ($tanh! x))
  (prn x))

;; sigmoid
(let ((x (tensor '((-2 -1) (1 2)))))
  (prn x)
  (prn ($sigmoid x))
  (prn x)
  (prn ($sigmoid! x))
  (prn x))

;; equal
(let ((x (tensor '(1 2 3)))
      (y (tensor '(1 2 3))))
  (prn ($equal x y)))

;; add, sub, and mul
(let ((x (tensor '((1 2) (3 4))))
      (y (tensor '((2 3) (4 5))))
      (a 10))
  (prn ($add x a))
  (prn ($add x y))
  (prn ($sub x a))
  (prn ($sub x y))
  (prn ($mul x a))
  (prn ($mul x y)))

;; clamp
(let ((x (tensor '((1 2 3 4 5) (2 3 4 5 6) (3 4 5 6 7))))
      (min 2)
      (max 5))
  (prn x)
  (prn ($clamp x min max))
  (prn x)
  (prn ($clamp! x min max))
  (prn x))

;; add-cmul
(let ((x (tensor 2 2))
      (y (tensor 4))
      (z (tensor 2 2)))
  ($fill! x 1)
  ($fill! y 3)
  ($fill! z 5)
  (prn ($addmul x y z 2)))

;; div
(let ((x (ones 2 2))
      (y (range 1 4)))
  (prn ($div x y)))

;; add-cdiv
(let ((x (-> (tensor 2 2) ($fill! 1)))
      (y (range 1 4))
      (z (-> (tensor 2 2) ($fill! 5))))
  (prn ($adddiv x y z 2)))

;; fmod, remainder
(let ((x (tensor '(-3 3))))
  (prn ($fmod x 2))
  (prn ($fmod x -2))
  (prn ($rem x 2))
  (prn ($rem x -2))
  (prn ($fmod (tensor '((3 3) (-3 -3))) (tensor '((2 -2) (2 -2)))))
  (prn ($rem (tensor '((3 3) (-3 -3))) (tensor '((2 -2) (2 -2))))))

;; dot
(let ((x (-> (tensor 2 2) ($fill! 3)))
      (y (-> (tensor 4) ($fill! 2))))
  (prn ($dot x y)))

;; add-mv
(let ((y (ones 3))
      (m (-> (tensor 3 2) ($fill! 3)))
      (x (-> (tensor 2) ($fill! 2))))
  (prn ($mv m x))
  (prn ($addmv y m x)))

;; add-r
(let ((x (range 1 3))
      (y (range 1 2))
      (m (ones 3 2)))
  (prn ($addr m x y))
  (prn ($addr! m x y 2 1)))

;; add-mm
(let ((c (ones 4 4))
      (a (-> (range 1 12) ($resize! '(4 3))))
      (b (-> (range 1 12) ($resize! '(3 4)))))
  (prn ($addmm c a b)))

;; add-bmm
(let ((c (ones 4 4))
      (ba (-> (range 1 24) ($resize! '(2 4 3))))
      (bb (-> (range 1 24) ($resize! '(2 3 4)))))
  (prn ($addbmm c ba bb)))

;; badd-bmm
(let ((bc (ones 2 4 4))
      (ba (-> (range 1 24) ($resize! '(2 4 3))))
      (bb (-> (range 1 24) ($resize! '(2 3 4)))))
  (prn ($baddbmm bc ba bb)))

;; operators
(prn ($+ 5 (rnd 3)))
(let ((x (-> (tensor 2 2) ($fill! 2)))
      (y (-> (tensor 4) ($fill! 3))))
  (prn ($+ x y))
  (prn ($- y x))
  (prn ($+ x 3))
  (prn ($- x)))
(let ((m (-> (tensor 2 2) ($fill! 2)))
      (n (-> (tensor 2 4) ($fill! 3)))
      (x (-> (tensor 2) ($fill! 4)))
      (y (-> (tensor 2) ($fill! 5))))
  (prn ($* x y))
  (prn ($@ m x))
  (prn ($@ m n)))
(prn ($/ (ones 2 2) 3))

;; cross
(let ((x (rndn 4 3))
      (y (rndn 4 3))
      (z (tensor)))
  (prn x)
  (prn y)
  (prn ($xx x y))
  (prn ($xx! z x y 1))
  (prn z))

;; cumulative product
(let ((x (range 1 5))
      (m (tensor.long '((1 4 7) (2 5 8) (3 6 9)))))
  (prn x)
  (prn ($cumprd x))
  (prn m)
  (prn ($cumprd m))
  (prn ($cumprd m 0)))

;; cumulative sum
(let ((x (range 1 5))
      (m (tensor.long '((1 4 7) (2 5 8) (3 6 9)))))
  (prn x)
  (prn ($cumsum x))
  (prn m)
  (prn ($cumsum m))
  (prn ($cumsum m 1)))

;; max and min
(let ((x (rndn 4 4))
      (vals (tensor))
      (indices (tensor.long)))
  (prn x)
  (prn ($max x))
  (prn ($min x))
  (prn ($max! vals indices x))
  (prn ($max! vals indices x 1)))

;; mean
(let ((x (rndn 3 4)))
  (prn x)
  (prn ($mean x))
  (prn ($mean x 0))
  (prn ($mean x 1)))

;; cmax
(let ((a (tensor '(1 2 3)))
      (b (tensor '(3 2 1))))
  (prn ($cmax a b))
  (prn ($cmax a b 2 3)))

;; cmin
(let ((a (tensor '(1 2 3)))
      (b (tensor '(3 2 1))))
  (prn ($cmin a b))
  (prn ($cmin a b 2 3)))

;; median
(let ((x (rndn 3 4))
      (vals (tensor))
      (indices (tensor.long)))
  (prn x)
  (prn ($median x))
  (prn ($median! vals indices x))
  (prn ($median! vals indices x 1)))

;; product
(let ((a (tensor '(((1 2) (3 4)) ((5 6) (7 8))))))
  (prn a)
  (prn ($prd a))
  (prn ($prd a 0))
  (prn ($prd a 1)))

;; sort
(let ((x (rndn 3 3))
      (vals (tensor))
      (indices (tensor.long)))
  (prn x)
  (prn ($sort! vals indices x)))

;; conv2
(let ((x (rnd 100 100))
      (k (rnd 10 10)))
  (prn ($size ($conv2 x k)))
  (prn ($size ($conv2 x k :full))))
(let ((x (rnd 500 100 100))
      (k (rnd 500 10 10)))
  (prn ($size ($conv2 x k)))
  (prn ($size ($conv2 x k :full))))

;; conv3 - slow, in this laptop, it takes ~6secs
(let ((x (rnd 100 100 100))
      (k (rnd 10 10 10)))
  (prn ($size ($conv3 x k)))
  (prn ($size ($conv3 x k :full))))

;; gesv
(let ((a (-> (tensor '((6.80 -2.11  5.66  5.97  8.23)
                       (-6.05 -3.30  5.36 -4.44  1.08)
                       (-0.45  2.58 -2.70  0.27  9.04)
                       (8.32  2.71  4.35  -7.17  2.14)
                       (-9.67 -5.14 -7.26  6.08 -6.87)))
             ($transpose)))
      (b (-> (tensor '((4.02  6.19 -8.22 -7.57 -3.03)
                       (-1.56  4.00 -8.67  1.75  2.86)
                       (9.81 -4.09 -4.57 -8.61  8.99)))
             ($transpose)))
      (x (tensor))
      (lu (tensor)))
  (prn a)
  (prn b)
  (prn ($gesv! x lu b a))
  (prn x)
  (prn ($@ a x))
  (prn ($dist b ($@ a x))))

;; trtrs
(let ((a (-> (tensor '((6.80 -2.11  5.66  5.97  8.23)
                       (0 -3.30  5.36 -4.44  1.08)
                       (0  0 -2.70  0.27  9.04)
                       (0  0  0  -7.17  2.14)
                       (0  0  0  0 -6.87)))))
      (b (-> (tensor '((4.02  6.19 -8.22 -7.57 -3.03)
                       (-1.56  4.00 -8.67  1.75  2.86)
                       (9.81 -4.09 -4.57 -8.61  8.99)))
             ($transpose)))
      (x (tensor)))
  (prn a)
  (prn b)
  (prn ($trtrs! x b a))
  (prn x)
  (prn ($@ a x))
  (prn ($dist b ($@ a x))))

;; potrf
(let ((a (tensor '((1.2705  0.9971  0.4948  0.1389  0.2381)
                   (0.9971  0.9966  0.6752  0.0686  0.1196)
                   (0.4948  0.6752  1.1434  0.0314  0.0582)
                   (0.1389  0.0686  0.0314  0.0270  0.0526)
                   (0.2381  0.1196  0.0582  0.0526  0.3957))))
      (chu (tensor))
      (chl (tensor)))
  (prn ($potrf! chu a))
  (prn ($@ ($transpose chu) chu))
  (prn ($potrf! chl a nil))
  (prn ($@ chl ($transpose chl))))

;; pstrf
(let ((a (tensor '((1.2705  0.9971  0.4948  0.1389  0.2381)
                   (0.9971  0.9966  0.6752  0.0686  0.1196)
                   (0.4948  0.6752  1.1434  0.0314  0.0582)
                   (0.1389  0.0686  0.0314  0.0270  0.0526)
                   (0.2381  0.1196  0.0582  0.0526  0.3957))))
      (chu (tensor))
      (chl (tensor))
      (piv (tensor.int))
      (ap nil))
  (prn ($pstrf! chu piv a))
  (setf ap ($@ ($transpose chu) chu))
  (prn ap)
  (prn a)
  (setf ($index ap 0 piv) ($clone ap))
  (setf ($index ap 1 piv) ($clone ap))
  (prn ap)
  (prn ($norm ($- a ap)))
  (prn ($pstrf! chl piv a nil))
  (setf ap ($@ chl ($transpose chl)))
  (prn ap)
  (prn a)
  (setf ($index ap 0 piv) ($clone ap))
  (setf ($index ap 1 piv) ($clone ap))
  (prn ap)
  (prn ($norm ($- a ap))))

;; potrs
(let ((a (tensor '((1.2705  0.9971  0.4948  0.1389  0.2381)
                   (0.9971  0.9966  0.6752  0.0686  0.1196)
                   (0.4948  0.6752  1.1434  0.0314  0.0582)
                   (0.1389  0.0686  0.0314  0.0270  0.0526)
                   (0.2381  0.1196  0.0582  0.0526  0.3957))))
      (b (tensor '((0.6219  0.3439  0.0431)
                   (0.5642  0.1756  0.0153)
                   (0.2334  0.8594  0.4103)
                   (0.7556  0.1966  0.9637)
                   (0.1420  0.7185  0.7476))))
      (cholesky (tensor))
      (solve (tensor)))
  ($potrf! cholesky a)
  (prn cholesky)
  (prn a)
  (prn ($@ ($transpose cholesky) cholesky))
  (prn ($dist a ($@ ($transpose cholesky) cholesky)))
  ($potrs! solve b cholesky)
  (prn solve)
  (prn b)
  (prn ($@ a solve))
  (prn ($dist b ($@ a solve))))

;; potri
(let ((a (tensor '((1.2705  0.9971  0.4948  0.1389  0.2381)
                   (0.9971  0.9966  0.6752  0.0686  0.1196)
                   (0.4948  0.6752  1.1434  0.0314  0.0582)
                   (0.1389  0.0686  0.0314  0.0270  0.0526)
                   (0.2381  0.1196  0.0582  0.0526  0.3957))))
      (cholesky (tensor))
      (inv (tensor)))
  ($potrf! cholesky a)
  ($potri! inv cholesky)
  (prn ($@ a inv))
  (prn ($dist (eye 5 5) ($@ a inv))))

;; gels
(let ((a (-> (tensor '((1.44 -9.96 -7.55  8.34  7.08 -5.45)
                       (-7.84 -0.28  3.24  8.09  2.52 -5.70)
                       (-4.39 -3.24  6.27  5.28  0.74 -1.19)
                       (4.53  3.83 -6.64  2.06 -2.47  4.70)))
             ($transpose)))
      (b (-> (tensor '((8.58  8.26  8.48 -5.28  5.72  8.93)
                       (9.35 -4.43 -0.70 -0.26 -7.36 -2.52)))
             ($transpose)))
      (x (tensor)))
  ($gels! x b a)
  (prn x)
  (prn ($dist b ($@ a ($narrow x 0 0 4)))))

;; syev
(let ((a (-> (tensor '((1.96  0.00  0.00  0.00  0.00)
                       (-6.49  3.80  0.00  0.00  0.00)
                       (-0.47 -6.39  4.17  0.00  0.00)
                       (-7.20  1.50 -1.51  5.70  0.00)
                       (-0.65 -6.34  2.67  1.80 -7.10)))
             ($transpose)))
      (e (tensor))
      (v (tensor)))
  (prn a)
  ($syev! e v a)
  (prn e)
  ($syev! e v a t)
  (prn e)
  (prn v)
  (prn ($@ v ($diag e) ($transpose v)))
  (prn ($dist a ($triu ($@ v ($diag e) ($transpose v))))))

;; ev - XXX ERROR, overflow with float32
(let ((a (-> (tensor '((1.96  0.00  0.00  0.00  0.00)
                       (-6.49  3.80  0.00  0.00  0.00)
                       (-0.47 -6.39  4.17  0.00  0.00)
                       (-7.20  1.50 -1.51  5.70  0.00)
                       (-0.65 -6.34  2.67  1.80 -7.10)))
             ($transpose)))
      (b nil)
      (e (tensor))
      (v (tensor)))
  (setf b ($+ a ($transpose ($triu a 1))))
  (prn b)
  ($ev! e v b)
  (prn e)
  ($ev! e v b t)
  (prn e)
  (prn v)
  (prn ($@ v ($diag ($select e 1 0)) ($transpose v)))
  (prn ($dist b ($@ v ($diag ($select e 1 0)) ($transpose v)))))

;; svd
(let ((a (-> (tensor '((8.79  6.11 -9.15  9.57 -3.49  9.84)
                       (9.93  6.91 -7.93  1.64  4.02  0.15)
                       (9.83  5.04  4.86  8.83  9.80 -8.99)
                       (5.45 -0.27  4.85  0.74 10.00 -6.02)
                       (3.16  7.98  3.01  5.80  4.27 -5.31)))
             ($transpose)))
      (u (tensor))
      (s (tensor))
      (v (tensor)))
  (prn a)
  ($svd! u s v a)
  (prn u)
  (prn s)
  (prn v)
  (prn ($@ u ($diag s) ($transpose v)))
  (prn ($dist a ($@ u ($diag s) ($transpose v)))))

;; inverse
(let ((a (rnd 10 10)))
  (prn a)
  (prn ($@ a ($inverse a)))
  (prn ($dist (eye 10 10) ($@ a ($inverse a)))))

;; qr
(let ((a (tensor '((12 -51 4) (6 167 -68) (-4 24 -41))))
      (q (tensor))
      (r (tensor)))
  (prn a)
  ($qr! q r a)
  (prn q)
  (prn r)
  (prn ($round ($@ q r)))
  (prn ($@ ($transpose q) q)))

;; lt
(prn ($lt (tensor '((1 2) (3 4))) (tensor '((2 1) (4 3)))))
(let ((a (rnd 10))
      (b (rnd 10)))
  (prn a)
  (prn b)
  (prn ($lt a b))
  (prn ($ a ($gt a b)))
  (setf ($ a ($gt a b)) 123)
  (prn a))
