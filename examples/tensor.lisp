(defpackage th.tensor-examples
  (:use #:common-lisp
        #:mu
        #:th))

(in-package :th.tensor-examples)

;; creates an empty tensor of default tensor class (double)
(print (tensor))

;; createa a tensor of default tensor class type with sizes; elements are not set and .
(print (tensor 2 2))

;; creates a tensor with contents
(print (tensor '(1 2 3 4)))

;; creates a tensor with multidimensional contents
(print (tensor '((1 2) (3 4) (5 6))))

;; creates tensor with other tensor, content storage is shared.
(let ((x (tensor '(4 3 2 1))))
  (print "X = (4 3 2 1)")
  (print x)
  (print "SAME CONTENT AS X")
  (print (tensor x)))

;; create tensor with sizes and strides
(print (tensor '(2 2) '(2 1)))

;; type specific construction functions; same as above.
(print (tensor.byte '(1 2 3 4)))
(print (tensor.char '(1 2 3 4)))
(print (tensor.short '(1 2 3 4)))
(print (tensor.int '(1 2 3 4)))
(print (tensor.long '(1 2 3 4)))
(print (tensor.float '(1 2 3 4)))
(print (tensor.double '(1 2 3 4)))

;; clone a tensor creates a new tensor with independent storage.
(let ((x (tensor '((1 2) (3 4)))))
  (print "'((1 2) (3 4))")
  (print ($clone x)))

;; make a tensor with contiguously allocated memory if it is not allocated contiguously.
(print ($contiguous (tensor.float '((1 2 3) (4 5 6)))))

;; tensor types can be changed to each other
(print (tensor.byte (tensor.double '((1.234 2.345) (3.456 4.567)))))
(print (tensor.double (tensor.int '((1 2 3) (4 5 6)))))
(print (tensor.long (tensor.float '(1.2 3.4 5.6 7.8))))

;; check whether it is tensor or not
(print ($tensorp 0))
(print ($tensorp (tensor)))

;; query number of dimensions
(print ($ndim (tensor)))
(print ($ndim (tensor 2 3 4)))

;; query tensor size
(print ($size (tensor 2 3 4)))

;; query tensor size along given dimension
(print ($size (tensor 2 3 4) 2))

;; size of scalar value is nil
(print ($size 1))

;; stride of a tensor
(print ($stride (tensor 2 3 4)))

;; stride of a tensor along dimension
(print ($stride (tensor 2 3 4) 2))

;; stride of a nil object
(print ($stride nil))

;; storage of a tensor; a tensor is a spepcific view on the storage
(let* ((x (tensor 4 5))
       (s ($storage x)))
  (loop :for i :from 0 :below ($count s)
        :do (setf ($ s i) i))
  (print "'((0 1 2 3 4) ... (15 16 17 18 19))")
  (print x))

;; contiguous or not
(print ($contiguousp (tensor)))
(print ($contiguousp (tensor 2 2 2)))

;; size comparison using size value
(print (equal ($size (tensor 2 3)) ($size (tensor 2 3))))
(print (equal ($size (tensor)) ($size (tensor 2 3))))
(print (equal ($size (tensor 2 2)) '(2 2)))

;; size comparison
(print ($sizep (tensor 2 2) (tensor 2 2)))
(print ($sizep (tensor 2 3) (tensor)))

;; number of elements in a tensor
(print ($count (tensor 3 3 4)))

;; query the element at the index location
(let ((x (tensor '((1 2 3) (4 5 6)))))
  (print "'((1 2 3) (4 5 6))")
  (print x)
  (print "1")
  (print ($ x 0 0)))

;; creates a new view on the same storage of the given tensor
(let ((x (tensor))
      (y (tensor '((1 2) (3 4)))))
  (print "((1 2) (3 4))")
  (print ($set x y))
  (print "T")
  (print ($setp x y)))

;; copy elements from other tensor; both should have the same number of elements.
(let ((x (tensor '(1 2 3 4)))
      (y (tensor '((5 4) (3 2)))))
  ;; now the elements of x replaced with those of y
  ($copy x y)
  (print "(5 4 3 2)")
  (print x)
  (print "((5 4) (3 2))")
  (print y)
  ;; new int tensor from x, then copies elements from list
  ;; new int tensor will have same size of x
  (print "(123 234 345 456)")
  (print ($copy (tensor.int x) '(123 234 345 456)))
  (print "(5 4 3 2)")
  ;; storage is not shared
  (print x))

;; fill values
(let ((x (tensor 3 3)))
  ($fill x 123)
  (print "((123 123 123) ... (123 123 123))")
  (print x)
  (print "((0 0 0) ... (0 0 0))")
  ;; mutable method
  (print ($zero! x))
  (print "((0 0 0) ... (0 0 0))")
  (print x)
  (print "((1 1 1) ... (1 1 1))")
  ;; immutable method
  (print ($one x))
  (print "((0 0 0) ... (0 0 0))")
  (print x))

;; resizing a tensor allocates if more memory storage is required
;; note that $view only changes sizes or shape but without changing allocated memory.
(let ((x (tensor '((1 2 3) (3 4 5))))
      (y (tensor 3 3)))
  (print "((1 2 3 3) (4 5 ? ?) ... (? ? ? ?)))")
  (print ($resize x '(4 4)))
  (print "((1 2) (3 3))")
  (print ($resize x '(2 2)))
  ;; resize as y
  (print "((1 2 3) (3 4 5) (? ? ?))")
  (print ($resize x y)))

;; select - choose sub tensor at index along dimension
(let ((x (tensor '((1 2 3) (4 5 6) (7 8 9)))))
  (print "(4 5 6) - 2nd(1) row along 1st(0) dimension")
  (print ($select x 0 1))
  (print "(3 6 9) - 3rd(2) column along 2nd(1) dimension")
  (print ($select x 1 2))
  (setf ($select x 0 1) '(-11 -22 -33))
  (print "x with 2nd row changed as (-11 -22 -33)")
  (print x))

;; subdimensional select using $
(let ((x (tensor '((1 2 3) (4 5 6) (7 8 9)))))
  (print "1st row - (1 2 3)")
  (print ($ x 0))
  (setf ($ x 1) '(6 5 4))
  (print "2nd row changes as (6 5 4)")
  (print x))

;; narrow - start and size along dimension
;; returns a new tensor or view built with narrowing from start selected as size along dimension
(let ((x (tensor 5 6)))
  ($zero! x)
  (print "5x5 matrix filled with zeros.")
  (print x)
  ;; along 1st(0) dimension, from 2nd row, select 3 rows, then fill it as one.
  (-> ($narrow x 0 1 3)
      ($fill! 1))
  (print "along 1st(0) dimension, from 2nd(1) row, total 3 rows are filled with one.")
  (print x)
  (-> ($narrow x 1 1 4)
      ($fill! 2))
  (print "along 2nd(1) dimension, from 2nd(1) column, total 4 columns are filled with two.")
  (print x)
  (setf ($narrow x 1 0 2) '(0 11 22 33 44 55 66 77 88 99))
  (print "along 2nd(1) dimension, from 1st(0) column, total 2 columns are copied from list.")
  (print x))

;; subview - multiple start and size per dimension, kind of multiple narrows
;; each pair of start and size along each dimensions
(let ((x (tensor '((1 2 3 4 5 6)
                   (2 3 4 5 6 7)
                   (3 4 5 6 7 8)
                   (4 5 6 7 8 9)
                   (0 0 0 0 0 0)))))
  (print "5x6 matrix.")
  (print x)
  (print "((4 5 6) (5 6 7) (6 7 8)) - from 2nd row, 3 rows, from 3rd column, 3 columns.")
  (print ($subview x 1 3 2 3))
  (setf ($subview x 1 3 2 3) '(11 12 13 14 15 16 17 18 19))
  (print "((11 12 13) (14 15 16) (17 18 19))")
  (print ($subview x 1 3 2 3))
  (print "matrix changes.")
  (print x))

;; orderly subview using $
(let ((x (tensor '((1 2 3 4 5 6)
                   (2 3 4 5 6 7)
                   (3 4 5 6 7 8)
                   (4 5 6 7 8 9)
                   (0 0 0 0 0 0)))))
  (print "((4 5 6) (5 6 7) (6 7 8)) - from 2nd row, 3 rows, from 3rd column, 3 columns.")
  (print ($ x '(1 3) '(2 3)))
  (setf ($ x '(1 3) '(2 3)) '(11 12 13 14 15 16 17 18 19))
  (print "((11 12 13) (14 15 16) (17 18 19)) in x")
  (print x))

;; query with dimension index size pairs or subview using $
(let ((x (tensor '((1 2 3 4 5 6)
                   (2 3 4 5 6 7)
                   (3 4 5 6 7 8)
                   (4 5 6 7 8 9)
                   (0 0 0 0 0 0)))))
  (print "original x")
  (print x)
  (print "subview of size 3x3, from 2nd row and 3rd column")
  (print ($ x '(1 3) '(2 3)))
  (setf ($ x '(1 3) '(2 3)) '(11 12 13 14 15 16 17 18 19))
  (print "3x3 subview changed as 11 to 19")
  (print ($ x '(1 3) '(2 3)))
  (print "changed x")
  (print x))

;; general selection using $
(let ((x (zeros 5 6)))
  (print "5x6 zero matrix.")
  (print x)
  (setf ($ x 0 2) 1)
  (print "1 at 1st row, 3rd column")
  (print x)
  (setf ($ x 4) 9)
  (print "5th row as 9")
  (print x)
  (setf ($ x '(:all (5 1))) 8)
  (print "6th column as 8")
  (print x)
  (setf ($ x '((1 1) (1 3))) 2)
  (print "1x3 from 2nd row and 2nd column, filled with 2.")
  (print x)
  (setf ($ x '((0 5) (3 1))) -1)
  (print "5x1 from 1st row and 4th column, filled with -1.")
  (print x)
  (setf ($ x '((0 5) (1 1))) (range 1 5))
  (print "5x1 from 1st row and 2nd column, copied from (1 ... 5)")
  (print x)
  (setf ($ x ($lt x 0)) 5)
  (print "element as 5 if oringinal one is less than 0.")
  (print x))

;; index-select - creates a new storage, not sharing
;; collects subtensor along dimension at indices
(let ((x (tensor '((1 2 3) (4 5 6) (7 8 9)))))
  (print "original x")
  (print x)
  (print "1st(0) and 2nd(1) along 1st(0) dimension")
  (print ($index x 0 '(0 1)))
  (let ((y ($index x 1 '(1 2))))
    (print "2nd(1) and 3rd(2) along 2nd(1) dimension")
    (print y)
    ($fill! y 0)
    (print "zero filled")
    (print y)
    (print "x unchanged")
    (print x)))

;; index-copy - set copies elements into selected index location
(let ((x (tensor '((1 2 3 4) (2 3 4 5) (3 4 5 6) (4 5 6 7) (5 6 7 8))))
      (y (tensor 5 2)))
  ($fill! ($select y 1 0) -1)
  ($fill! ($select y 1 1) -2)
  (print "original x")
  (print x)
  (print "original y")
  (print y)
  (setf ($index x 1 '(3 0)) y)
  (print "4th(3) and 1st(0) columns along 2nd(1) dimension copied from y.")
  (print x))

;; index-fill
(let ((x (tensor '((1 2 3 4) (2 3 4 5) (3 4 5 6) (4 5 6 7) (5 6 7 8)))))
  (print "original x")
  (print x)
  (setf ($index x 1 '(0 3)) 123)
  (print "1st(0) and 4th(3) columns along 2nd(1) dimension set as 123")
  (print x))

;; gather
(let ((x (tensor 5 5)))
  (loop :for i :from 0 :below ($count x)
        :do (setf ($ ($storage x) i) i))
  (print "original 5x5 matrix")
  (print x)
  (print "by incrementing index, collect nth 1st(0) dimensional values.")
  (print "1st row will be (0 0), (1 1), (2 2), (3 3), (4 4)")
  (print "2nd row will be (1 0), (2 1), (3 2), (4 3), (0 4)")
  (print ($gather x 0 '((0 1 2 3 4) (1 2 3 4 0))))
  (print "by incrementing index, collect nth 2nd(1) dimensional values.")
  (print "1st column will be (0 0), (1 1), (2 2), (3 3), (4 4)")
  (print "2nd column will be (1 0), (2 1), (3 2), (4 3), (0 4)")
  (print ($gather x 1 '((0 1) (1 2) (2 3) (3 4) (4 0)))))

;; scatter
(let ((x (tensor 5 5))
      (y (tensor '((1 2 3 4 5) (-5 -4 -3 -2 -1)))))
  ($zero! x)
  (print "5x5 zeros")
  (print x)
  ($scatter x 0 '((0 1 2 3 4) (1 2 3 4 0)) y)
  (print "as in gather, but set from y")
  (print x)
  ($scatter x 1 '((0 1) (1 2) (2 3) (3 4) (4 0)) 9)
  (print "as in gather, but fill a value")
  (print x))

;; masked-select
(let ((x (tensor 3 4))
      (mask '((1 0 1 0 0 0) (1 1 0 0 0 1)))
      (z (tensor)))
  (loop :for i :from 0 :below ($count x)
        :do (setf ($ ($storage x) i) (1+ i)))
  (print "original x")
  (print x)
  (print "only at value 1(true)")
  (print ($masked x mask))
  ($set z ($masked x mask))
  (print "same as above.")
  (print z)
  ($fill! z -99)
  (print "z value changed as -99")
  (print z)
  (print "x unchanged.")
  (print x))

;; masked-copy
(let ((x (tensor 3 4))
      (mask '((1 0 1 0 0 0) (1 1 0 0 0 1)))
      (z (tensor '(1 2 3 4 5))))
  ($zero! x)
  (print "x matrix, 3x4")
  (print x)
  (setf ($masked x mask) z)
  (print "set by z")
  (print x))

;; masked-fill
(let ((x (tensor 3 4))
      (mask '((1 0 1 0 0 0) (1 1 0 0 0 1))))
  ($zero! x)
  (print "original 3x4 matrix")
  (print x)
  (setf ($masked x mask) 5)
  (print "filled as 5")
  (print x))

;; nonzero - returns locations of non zero elements
(print ($nonzero (tensor '((1 2 0 3 4) (0 0 1 0 0)))))

;; expand - shares storage, just new view
(let ((x (tensor 10 1))
      (y (tensor 10 2)))
  (loop :for i :from 0 :below ($count x)
        :do (setf ($ ($storage x) i) (1+ i)))
  (print "original 10x1 matrix x")
  (print x)
  (print "new expanded matrix as 10x4 using x")
  (print ($expand x 10 4))
  (print "new expanded matrix as 10x4 filled with one.")
  (print ($fill! ($expand x 10 4) 1))
  (print "modified because of $fill! 1, storage is shared.")
  (print x)
  (print "as the shape/size of other tensor, 10x2")
  (print ($expand x y)))

;; repeat - repeat content as given times
(print ($repeat (tensor '(1 2)) 3 2))

;; squeeze - removes singletone dimensions
(let ((x (tensor 2 1 2 1 2)))
  (print "original size")
  (print ($size x))
  (print "no 1s")
  (print ($size ($squeeze x)))
  (print "no 1 in 2nd(1) dimension")
  (print ($size ($squeeze x 1))))

;; unsqueeze - add a singleton dimension
(let ((x (tensor '(1 2 3 4))))
  (print "vector")
  (print x)
  (print "along 1st(0) dimension")
  (print ($unsqueeze x 0))
  (print "along 2nd(1) dimension")
  (print ($unsqueeze x 1)))

;; view - different from resize(allocation), reshape(new storage), just a view
(let ((x (tensor '(0 0 0 0))))
  (print "original vector")
  (print x)
  (print "2x2")
  (print ($view x 2 2))
  (print "as the size of other 2x2 tensor")
  (print ($view x (tensor 2 2))))

;; transpose
(let ((x (tensor '((1 2 3) (4 5 6)))))
  (print "original 2x3")
  (print x)
  (print "transposed 3x2")
  (print ($transpose x)))

;; transpose - shares storage but different view(tensor)
(let ((x (tensor 3 4)))
  ($zero! x)
  ($fill! ($select x 1 2) 7)
  (print "original x")
  (print x)
  (let ((y ($transpose x)))
    ($fill! ($select y 1 2) 8)
    (print "modified transposed x or y")
    (print y)
    (print "original x")
    (print x)))

;; permute - multidimensional transposing
(let ((x (tensor 3 4 2 5)))
  (print "original size")
  (print ($size x))
  (print "permute size with 2nd, 3rd, 1st and 4th dimensions - 4,2,3,5")
  (print ($size ($permute x 1 2 0 3))))

;; unfold - slice with size by step along dimension
(let ((x (tensor 7)))
  (loop :for i :from 1 :to 7 :do (setf ($ x (1- i)) i))
  (print "vector, 1 to 7")
  (print x)
  (print "slice along 1st(0) dimension, size of 2, by step 1")
  (print ($unfold x 0 2 1))
  (print "slice along 1st(0) dimension, size of 2, by step 2")
  (print ($unfold x 0 2 2)))

;; fmap - just elementwise function application
(let ((x (zeros 3 3))
      (n 0))
  ($fmap (lambda (v) (+ v (* 0.5 pi (incf n)))) x)
  (print x)
  ($fmap (lambda (v) (round (sin v))) x)
  (print x))

;; fmap - more, shape is irrelevant when they have same count.
(let ((x (tensor 3 3))
      (y (tensor 9))
      (z (tensor '(0 1 2 3 4 5 6 7 8))))
  (loop :for i :from 1 :to 9 :do (setf ($ ($storage x) (1- i)) i
                                       ($ ($storage y) (1- i)) i))
  (print x)
  (print y)
  (print z)
  (print "1*1, 2*2, 3*3, ...")
  (print ($fmap (lambda (vx vy) (* vx vy)) x y))
  (print "1 + 1 + 0, 2 + 2 + 1, 3 + 3 + 2, ...")
  (print ($fmap (lambda (xx yy zz) (+ xx yy zz)) x y z))
  (print x))

;; split - split tenor with size along dimension
(let ((x (zeros 3 4 5)))
  (print "by 2 along 0 - 2x4x5, 1x4x5")
  (print ($split x 2 0))
  (print "by 2 along 1 - 3x2x5, 3x2x5")
  (print ($split x 2 1))
  (print "by 2 along 2 - 3x4x2, 3x4x2, 3x4x1")
  (print ($split x 2 2)))

;; chunk - n parition of approaximately same size along dimension
(let ((x (ones 3 4 5)))
  (print "2 partitions along 0 - 2x4x5, 1x4x5")
  (print ($chunk x 2 0))
  (print "2 partitions along 1 - 3x2x5, 3x2x5")
  (print ($chunk x 2 1))
  (print "2 paritions along 2 - 3x4x3, 3x4x2")
  (print ($chunk x 2 2)))

;; concat tensors
(print ($cat 0 (ones 3) (zeros 3)))
(print ($cat 1 (ones 3) (zeros 3)))
(print ($cat 0 (ones 3 3) (zeros 1 3)))
(print ($cat 1 (ones 3 4) (zeros 3 2)))

;; diagonal matrix
(print ($diag (tensor '(1 2 3 4))))
(print ($diag (ones 3 3)))

;; identity matrix
(print (eye 2))
(print (eye 3 4))
(print ($eye (tensor.byte) 3))
(print ($eye (tensor.byte 10 20) 3 4))

;; linspace
(print (linspace 1 2))
(print (linspace 1 2 11))

;; logspace
(print (logspace 1 2))
(print (logspace 1 2 11))

;; uniform random
(print (rnd 3 3))

;; normal random
(print (rndn 2 4))

;; range
(print (range 2 5))
(print (range 2 5 1.2))

;; arange
(print (arange 2 5))
(print (range 2 5))

;; randperm
(print (rndperm 10))
(print (rndperm 5))

;; reshape
(let ((x (ones 2 3))
      (y nil))
  (print x)
  (setf y ($reshape x 3 2))
  (print y)
  ($fill! y 2)
  (print y)
  (print x))

;; tril and triu
(let ((x (ones 4 4)))
  (print ($tril x))
  (print ($tril x -1))
  (print ($triu x))
  (print ($triu x 1)))

;; abs
(let ((x (tensor '((-1 2) (3 -4)))))
  (print x)
  (print ($abs x))
  (print x)
  (print ($abs! x))
  (print x))

;; sign
(let ((x (tensor '((-1 2) (3 -4)))))
  (print x)
  (print ($sign x))
  (print x)
  (print ($sign! x))
  (print x))

;; acos
(let ((x (tensor '((-1 1) (1 -1)))))
  (print x)
  (print ($acos x))
  (print x)
  (print ($acos! x))
  (print x))

;; asin
(let ((x (tensor '((-1 1) (1 -1)))))
  (print x)
  (print ($asin x))
  (print x)
  (print ($asin! x))
  (print x))

;; atan
(let ((y (tensor '((-11 1) (1 -11)))))
  (print y)
  (print ($atan y))
  (print y)
  (print ($atan! y))
  (print y))

;; atan2
(let ((y (tensor '((-11 1) (1 -11))))
      (x (tensor '((1 1) (1 1)))))
  (print y)
  (print ($atan2 y x))
  (print y)
  (print ($atan2! y x))
  (print y))

;; ceil
(let ((x (tensor '((1 1.1 1.7) (-0.8 -1.1 -2.3)))))
  (print x)
  (print ($ceil x))
  (print x)
  (print ($ceil! x))
  (print x))

;; cos
(let ((x (tensor '((-3.14 0) (3.14 0)))))
  (print x)
  (print ($cos x))
  (print x)
  (print ($cos! x))
  (print x))

;; cosh
(let ((x (tensor '((-3.14 0) (3.14 0)))))
  (print x)
  (print ($cosh x))
  (print x)
  (print ($cosh! x))
  (print x))

;; exp
(let ((x (tensor '((0 1 2) (-1 -2 -3)))))
  (print x)
  (print ($exp x))
  (print x)
  (print ($exp! x))
  (print x))

;; floor
(let ((x (tensor '((1 1.1 1.7) (-0.8 -1.1 -2.3)))))
  (print x)
  (print ($floor x))
  (print x)
  (print ($floor! x))
  (print x))

;; log
(let ((x ($exp (tensor '((0 1 2) (-1 -2 -3))))))
  (print x)
  (print ($log x))
  (print x)
  (print ($log! x))
  (print x))

;; log1p
(let ((x ($exp (tensor '((0 1 2) (-1 -2 -3))))))
  (print x)
  (print ($log1p x))
  (print x)
  (print ($log1p! x))
  (print x))

;; neg
(let ((x (tensor '((0 1 2) (-1 -2 -3)))))
  (print x)
  (print ($neg x))
  (print x)
  (print ($neg! x))
  (print x))

;; cinv
(let ((x (tensor '((3 2 1) (-1 -2 -3)))))
  (print x)
  (print ($cinv x))
  (print x)
  (print ($cinv! x))
  (print x))

;; expt
(let ((x (tensor '((2 3) (1 2))))
      (y (tensor '((2 2) (3 3))))
      (n 2))
  (print x)
  (print ($expt x n))
  (print ($expt n x))
  (print ($expt x y))
  (print x)
  (print ($expt! x n))
  (print x)
  (print ($expt! n x))
  (print x))

;; round
(let ((x (tensor '((1.1 1.8) (-1.1 -1.8)))))
  (print x)
  (print ($round x))
  (print x)
  (print ($round! x))
  (print x))

;; sin
(let ((x (tensor '((-3.14 0) (3.14 0)))))
  (print x)
  (print ($sin x))
  (print x)
  (print ($sin! x))
  (print x))

;; sinh
(let ((x (tensor '((-3.14 0) (3.14 0)))))
  (print x)
  (print ($sinh x))
  (print x)
  (print ($sinh! x))
  (print x))

;; sqrt
(let ((x (tensor '((1 2 4) (3 5 9)))))
  (print x)
  (print ($sqrt x))
  (print x)
  (print ($sqrt! x))
  (print x))

;; rsqrt
(let ((x (tensor '((1 2 4) (3 5 9)))))
  (print x)
  (print ($rsqrt x))
  (print x)
  (print ($rsqrt! x))
  (print x))

;; tan
(let ((x (tensor '((1 2) (3 4)))))
  (print x)
  (print ($tan x))
  (print x)
  (print ($tan! x))
  (print x))

;; tanh
(let ((x (tensor '((1 2) (-3 -4)))))
  (print x)
  (print ($tanh x))
  (print x)
  (print ($tanh! x))
  (print x))

;; sigmoid
(let ((x (tensor '((-2 -1) (1 2)))))
  (print x)
  (print ($sigmoid x))
  (print x)
  (print ($sigmoid! x))
  (print x))

;; equal
(let ((x (tensor '(1 2 3)))
      (y (tensor '(1 2 3))))
  (print ($equal x y)))

;; add, sub, and mul
(let ((x (tensor '((1 2) (3 4))))
      (y (tensor '((2 3) (4 5))))
      (a 10))
  (print ($add x a))
  (print ($add x y))
  (print ($sub x a))
  (print ($sub x y))
  (print ($mul x a))
  (print ($mul x y)))

;; clamp
(let ((x (tensor '((1 2 3 4 5) (2 3 4 5 6) (3 4 5 6 7))))
      (min 2)
      (max 5))
  (print x)
  (print ($clamp x min max))
  (print x)
  (print ($clamp! x min max))
  (print x))

;; add-cmul
(let ((x (tensor 2 2))
      (y (tensor 4))
      (z (tensor 2 2)))
  ($fill! x 1)
  ($fill! y 3)
  ($fill! z 5)
  (print ($addmul x y z 2)))

;; div
(let ((x (ones 2 2))
      (y (range 1 4)))
  (print ($div x y)))

;; add-cdiv
(let ((x (-> (tensor 2 2) ($fill! 1)))
      (y (range 1 4))
      (z (-> (tensor 2 2) ($fill! 5))))
  (print ($adddiv x y z 2)))

;; fmod, remainder
(let ((x (tensor '(-3 3))))
  (print ($fmod x 2))
  (print ($fmod x -2))
  (print ($rem x 2))
  (print ($rem x -2))
  (print ($fmod (tensor '((3 3) (-3 -3))) (tensor '((2 -2) (2 -2)))))
  (print ($rem (tensor '((3 3) (-3 -3))) (tensor '((2 -2) (2 -2))))))

;; dot
(let ((x (-> (tensor 2 2) ($fill! 3)))
      (y (-> (tensor 4) ($fill! 2))))
  (print ($dot x y)))

;; add-mv
(let ((y (ones 3))
      (m (-> (tensor 3 2) ($fill! 3)))
      (x (-> (tensor 2) ($fill! 2))))
  (print ($mv m x))
  (print ($addmv y m x)))

;; add-r
(let ((x (range 1 3))
      (y (range 1 2))
      (m (ones 3 2)))
  (print ($addr m x y))
  (print ($addr! m x y 2 1)))

;; add-mm
(let ((c (ones 4 4))
      (a (-> (range 1 12) ($resize '(4 3))))
      (b (-> (range 1 12) ($resize '(3 4)))))
  (print ($addmm c a b)))

;; add-bmm
(let ((c (ones 4 4))
      (ba (-> (range 1 24) ($resize '(2 4 3))))
      (bb (-> (range 1 24) ($resize '(2 3 4)))))
  (print ($addbmm c ba bb)))

;; badd-bmm
(let ((bc (ones 2 4 4))
      (ba (-> (range 1 24) ($resize '(2 4 3))))
      (bb (-> (range 1 24) ($resize '(2 3 4)))))
  (print ($baddbmm bc ba bb)))

;; operators
(print ($+ 5 (rnd 3)))
(let ((x (-> (tensor 2 2) ($fill! 2)))
      (y (-> (tensor 4) ($fill! 3))))
  (print ($+ x y))
  (print ($- y x))
  (print ($+ x 3))
  (print ($- x)))
(let ((m (-> (tensor 2 2) ($fill! 2)))
      (n (-> (tensor 2 4) ($fill! 3)))
      (x (-> (tensor 2) ($fill! 4)))
      (y (-> (tensor 2) ($fill! 5))))
  (print ($* x y))
  (print ($@ m x))
  (print ($@ m n)))
(print ($/ (ones 2 2) 3))

(let ((x (tensor '(1 2 3)))
      (y (tensor '(3 2 1))))
  (print (th::tensor-cross (tensor) x y 0)))

;; cross
(let ((x (rndn 4 3))
      (y (rndn 4 3))
      (z (tensor)))
  (print x)
  (print y)
  (print ($xx x y))
  (print ($xx! z x y 1))
  (print z))

;; cumulative product
(let ((x (range 1 5))
      (m (tensor.long '((1 4 7) (2 5 8) (3 6 9)))))
  (print x)
  (print ($cumprd x))
  (print m)
  (print ($cumprd m))
  (print ($cumprd m 0)))

;; cumulative sum
(let ((x (range 1 5))
      (m (tensor.long '((1 4 7) (2 5 8) (3 6 9)))))
  (print x)
  (print ($cumsum x))
  (print m)
  (print ($cumsum m))
  (print ($cumsum m 1)))

;; max and min
(let ((x (rndn 4 4))
      (vals (tensor))
      (indices (tensor.long)))
  (print x)
  (print ($max x))
  (print ($min x))
  (print ($max! vals indices x))
  (print ($max! vals indices x 1)))

;; mean
(let ((x (rndn 3 4)))
  (print x)
  (print ($mean x))
  (print ($mean x 0))
  (print ($mean x 1)))

;; cmax
(let ((a (tensor '(1 2 3)))
      (b (tensor '(3 2 1))))
  (print ($cmax a b))
  (print ($cmax a b 2 3)))

;; cmin
(let ((a (tensor '(1 2 3)))
      (b (tensor '(3 2 1))))
  (print ($cmin a b))
  (print ($cmin a b 2 3)))

;; median
(let ((x (rndn 3 4))
      (vals (tensor))
      (indices (tensor.long)))
  (print x)
  (print ($median x))
  (print ($median! vals indices x))
  (print ($median! vals indices x 1)))

;; product
(let ((a (tensor '(((1 2) (3 4)) ((5 6) (7 8))))))
  (print a)
  (print ($prd a))
  (print ($prd a 0))
  (print ($prd a 1)))

;; sort
(let ((x (rndn 3 3))
      (vals (tensor))
      (indices (tensor.long)))
  (print x)
  (print ($sort! vals indices x)))

;; conv2
(let ((x (rnd 100 100))
      (k (rnd 10 10)))
  (print ($size ($conv2 x k)))
  (print ($size ($conv2 x k :full))))
(let ((x (rnd 500 100 100))
      (k (rnd 500 10 10)))
  (print ($size ($conv2 x k)))
  (print ($size ($conv2 x k :full))))

;; conv3 - slow, in this laptop, it takes ~6secs
(let ((x (rnd 100 100 100))
      (k (rnd 10 10 10)))
  (print ($size ($conv3 x k)))
  (print ($size ($conv3 x k :full))))

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
  (print a)
  (print b)
  (print ($gesv! x lu b a))
  (print x)
  (print ($@ a x))
  (print ($dist b ($@ a x))))

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
  (print a)
  (print b)
  (print ($trtrs! x b a))
  (print x)
  (print ($@ a x))
  (print ($dist b ($@ a x))))

;; potrf
(let ((a (tensor '((1.2705  0.9971  0.4948  0.1389  0.2381)
                   (0.9971  0.9966  0.6752  0.0686  0.1196)
                   (0.4948  0.6752  1.1434  0.0314  0.0582)
                   (0.1389  0.0686  0.0314  0.0270  0.0526)
                   (0.2381  0.1196  0.0582  0.0526  0.3957))))
      (chu (tensor))
      (chl (tensor)))
  (print ($potrf! chu a))
  (print ($@ ($transpose chu) chu))
  (print ($potrf! chl a nil))
  (print ($@ chl ($transpose chl))))

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
  (print ($pstrf! chu piv a))
  (setf ap ($@ ($transpose chu) chu))
  (print ap)
  (print a)
  (setf ($index ap 0 piv) ($clone ap))
  (setf ($index ap 1 piv) ($clone ap))
  (print ap)
  (print ($norm ($- a ap)))
  (print ($pstrf! chl piv a nil))
  (setf ap ($@ chl ($transpose chl)))
  (print ap)
  (print a)
  (setf ($index ap 0 piv) ($clone ap))
  (setf ($index ap 1 piv) ($clone ap))
  (print ap)
  (print ($norm ($- a ap))))

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
  (print cholesky)
  (print a)
  (print ($@ ($transpose cholesky) cholesky))
  (print ($dist a ($@ ($transpose cholesky) cholesky)))
  ($potrs! solve b cholesky)
  (print solve)
  (print b)
  (print ($@ a solve))
  (print ($dist b ($@ a solve))))

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
  (print ($@ a inv))
  (print ($dist (eye 5 5) ($@ a inv))))

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
  (print x)
  (print ($dist b ($@ a ($narrow x 0 0 4)))))

;; syev
(let ((a (-> (tensor '((1.96  0.00  0.00  0.00  0.00)
                       (-6.49  3.80  0.00  0.00  0.00)
                       (-0.47 -6.39  4.17  0.00  0.00)
                       (-7.20  1.50 -1.51  5.70  0.00)
                       (-0.65 -6.34  2.67  1.80 -7.10)))
             ($transpose)))
      (e (tensor))
      (v (tensor)))
  (print a)
  ($syev! e v a)
  (print e)
  ($syev! e v a t)
  (print e)
  (print v)
  (print ($@ v ($diag e) ($transpose v)))
  (print ($dist a ($triu ($@ v ($diag e) ($transpose v))))))

;; ev
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
  (print b)
  ($ev! e v b)
  (print e)
  ($ev! e v b t)
  (print e)
  (print v)
  (print ($@ v ($diag ($select e 1 0)) ($transpose v)))
  (print ($dist b ($@ v ($diag ($select e 1 0)) ($transpose v)))))

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
  (print a)
  ($svd! u s v a)
  (print u)
  (print s)
  (print v)
  (print ($@ u ($diag s) ($transpose v)))
  (print ($dist a ($@ u ($diag s) ($transpose v)))))

;; inverse
(let ((a (rnd 10 10)))
  (print a)
  (print ($@ a ($inverse a)))
  (print ($dist (eye 10 10) ($@ a ($inverse a)))))

;; qr
(let ((a (tensor '((12 -51 4) (6 167 -68) (-4 24 -41))))
      (q (tensor))
      (r (tensor)))
  (print a)
  ($qr! q r a)
  (print q)
  (print r)
  (print ($round ($@ q r)))
  (print ($@ ($transpose q) q)))

;; lt
(print ($lt (tensor '((1 2) (3 4))) (tensor '((2 1) (4 3)))))
(let ((a (rnd 10))
      (b (rnd 10)))
  (print a)
  (print b)
  (print ($lt a b))
  (print ($ a ($gt a b)))
  (setf ($ a ($gt a b)) 123)
  (print a))
