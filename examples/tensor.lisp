(in-package :th)

;;
;; CONSTRUCTION
;;
;; default construction
(print (tensor))
(print (tensor 2 2))
(print (tensor '(1 2 3 4)))
(print (tensor '((1 2) (3 4) (5 6))))
(print (tensor (tensor '(4 3 2 1))))
(print (tensor '(2 2) '(2 1)))
;; type specific construction
(print (tensor.byte '(1 2 3 4)))
(print (tensor.char '(1 2 3 4)))
(print (tensor.short '(1 2 3 4)))
(print (tensor.int '(1 2 3 4)))
(print (tensor.long '(1 2 3 4)))
(print (tensor.float '(1 2 3 4)))
(print (tensor.double '(1 2 3 4)))

;; clone
(print ($clone (tensor '((1 2) (3 4)))))

;; contiguous
(print ($contiguous (tensor.float '((1 2 3) (4 5 6)))))

;; coerced copying
(print (tensor.byte (tensor.double '((1.234 2.345) (3.456 4.567)))))
(print (tensor.double (tensor.int '((1 2 3) (4 5 6)))))
(print (tensor.long (tensor.float '(1.2 3.4 5.6 7.8))))

;; tensor or not
(print ($tensorp 0))
(print ($tensorp (tensor)))

;; dimensions, size and stride
(print ($ndim (tensor)))
(print ($ndim (tensor 2 3 4)))
(print ($size (tensor 2 3 4)))
(print ($size (tensor 2 3 4) 2))
(print ($size 1))
(print ($stride (tensor 2 3 4)))
(print ($stride (tensor 2 3 4) 2))
(print ($stride nil))

;; storage - a tensor is a spepcific view on the storage
(let* ((x (tensor 4 5))
       (s ($storage x)))
  (loop :for i :from 0 :below ($count s)
        :do (setf ($ s i) i))
  (print x))

;; contiguous or not
(print ($contiguousp (tensor)))

;; size comparison
(print (equal ($size (tensor 2 3)) ($size (tensor 2 3))))
(print (equal ($size (tensor)) ($size (tensor 2 3))))
(print (equal ($size (tensor 2 2)) '(2 2)))

;; element count
(print ($count (tensor 3 3 4)))

;; querying elements
(let ((x (tensor '((1 2 3) (4 5 6)))))
  (print x)
  (print ($ x 0 0))
  (print ($ x 0)))

;; set another view
(let ((x (tensor))
      (y (tensor '((1 2) (3 4)))))
  (print ($set x y))
  (print ($setp x y)))

;; copy elements from other tensor
(let ((x (tensor '(1 2 3 4)))
      (y (tensor '((5 4) (3 2)))))
  (print x)
  (print y)
  ($copy x y)
  (print x)
  (print y)
  (print ($copy (tensor.int x) '(123 234 345 456)))
  (print x))

;; fill values
(let ((x (tensor 3 3)))
  ($fill x 123)
  (print x)
  (print ($zero x))
  (print x)
  (print ($one x))
  (print x))

;; resizing
(let ((x (tensor '((1 2 3) (3 4 5))))
      (y (tensor 3 3)))
  (print x)
  (print ($resize x '(4 4)))
  (print ($resize x '(2 2)))
  (print ($resize x y)))

;; narrow - start and size
(let ((x (tensor 5 6)))
  ($zero x)
  (print x)
  (-> ($narrow x 0 1 3)
      ($fill 1))
  (print x)
  (-> ($narrow x 1 1 4)
      ($fill 2))
  (print x)
  (setf ($narrow x 1 0 2) '(0 11 22 33 44 55 66 77 88 99))
  (print x))

;; subview - multiple start and size per dimension
(let ((x (tensor '((1 2 3 4 5 6)
                   (2 3 4 5 6 7)
                   (3 4 5 6 7 8)
                   (4 5 6 7 8 9)
                   (0 0 0 0 0 0)))))
  (print x)
  (print ($subview x 1 3 2 3))
  (setf ($subview x 1 3 2 3) '(11 12 13 14 15 16 17 18 19))
  (print ($subview x 1 3 2 3))
  (print x))

;; select
(let ((x (tensor '((1 2 3) (4 5 6) (7 8 9)))))
  (print x)
  (print ($select x 0 1))
  (print ($ x 1))
  (print ($select x 1 1))
  (setf ($ x 1) '(-11 -22 -33))
  (print x))

;; query with dimension index size pairs or subview
(let ((x (tensor '((1 2 3 4 5 6)
                   (2 3 4 5 6 7)
                   (3 4 5 6 7 8)
                   (4 5 6 7 8 9)
                   (0 0 0 0 0 0)))))
  (print x)
  (print ($ x '(1 3) '(2 3)))
  (setf ($ x '(1 3) '(2 3)) '(11 12 13 14 15 16 17 18 19))
  (print ($ x '(1 3) '(2 3)))
  (print x))

;; general selection
(let ((x (zeros 5 6)))
  (print x)
  (setf ($ x 0 2) 1)
  (print x)
  (setf ($ x '((1 1) (1 3))) 2)
  (print x)
  (setf ($ x '((0 5) (3 1))) -1)
  (print x)
  (setf ($ x '((0 5) (1 1))) (range 1 5))
  (print x)
  (setf ($ x ($lt x 0)) -2)
  (print x))

;; index-select - creates a new storage, not sharing
(let ((x (tensor '((1 2 3) (4 5 6) (7 8 9)))))
  (print x)
  (print ($index x 0 '(0 1)))
  (let ((y ($index x 1 '(1 2))))
    (print y)
    ($fill y 0)
    (print y)
    (print x)))

;; index-copy - set copies elements into selected index location
(let ((x (tensor '((1 2 3 4) (2 3 4 5) (3 4 5 6) (4 5 6 7) (5 6 7 8))))
      (y (tensor 5 2)))
  ($fill ($select y 1 0) -1)
  ($fill ($select y 1 1) -2)
  (print x)
  (print y)
  (setf ($index x 1 '(0 3)) y)
  (print x))

;; index-add
(let ((x ($tensor '((1 2 3 4) (2 3 4 5) (3 4 5 6) (4 5 6 7) (5 6 7 8))))
      (y ($tensor 5 2)))
  ($fill ($select y 1 0) -10)
  ($fill ($select y 1 1) -100)
  (print x)
  (print y)
  ($index-add x 1 '(0 3) y)
  (print x))

;; index-fill
(let ((x ($tensor '((1 2 3 4) (2 3 4 5) (3 4 5 6) (4 5 6 7) (5 6 7 8)))))
  (print x)
  ($index-fill x 1 '(0 3) 123)
  (print x))

;; gather
(let ((x ($tensor 5 5)))
  (loop :for i :from 0 :below ($count x)
        :do (setf ($ ($storage x) i) i))
  (print x)
  (print ($gather x 0 '((0 1 2 3 4) (1 2 3 4 0))))
  (print ($gather x 1 '((0 1) (1 2) (2 3) (3 4) (4 0)))))

;; scatter
(let ((x ($tensor 5 5))
      (y ($tensor '((11 21 31 41 51) (12 22 32 42 52)))))
  ($zero x)
  (print x)
  ($scatter x 0 '((0 1 2 3 4) (1 2 3 4 0)) y)
  (print x)
  ($scatter x 1 '((0 1) (1 2) (2 3) (3 4) (4 0)) 1234)
  (print x))

;; masked-select
(let ((x ($tensor 3 4))
      (mask '((1 0 1 0 0 0) (1 1 0 0 0 1)))
      (z ($tensor)))
  (loop :for i :from 0 :below ($count x)
        :do (setf ($ ($storage x) i) (1+ i)))
  (print x)
  (print ($masked-select x mask))
  ($set z ($masked-select x mask))
  (print z)
  ($fill z -123)
  (print z)
  (print x))

;; masked-copy
(let ((x ($tensor 3 4))
      (mask '((1 0 1 0 0 0) (1 1 0 0 0 1)))
      (z ($tensor '(101 102 103 104 105))))
  (loop :for i :from 0 :below ($count x)
        :do (setf ($ ($storage x) i) (1+ i)))
  (print x)
  ($masked-copy x mask z)
  (print x))

;; masked-fill
(let ((x ($tensor 3 4))
      (mask '((1 0 1 0 0 0) (1 1 0 0 0 1))))
  (loop :for i :from 0 :below ($count x)
        :do (setf ($ ($storage x) i) (1+ i)))
  (print x)
  ($masked-fill x mask -123)
  (print x))

;; nonzero
(print ($nonzero ($tensor '((1 2 0 3 4) (0 0 1 0 0)))))

;; expand
(let ((x ($tensor 10 1))
      (y ($tensor 10 2)))
  (loop :for i :from 0 :below ($count x)
        :do (setf ($ ($storage x) i) (1+ i)))
  (print x)
  (print ($expand x 10 4))
  (print ($fill ($expand x 10 4) 1))
  (print x)
  (print ($expand x y)))

;; repeat - not implemented yet

;; squeeze
(let ((x ($tensor 2 1 2 1 2)))
  (print ($size x))
  (print ($size ($squeeze x)))
  (print ($size ($squeeze x 1))))

;; view
(let ((x ($tensor '(0 0 0 0))))
  (print x)
  (print ($view x 2 2))
  (print ($view x ($tensor 2 2))))

;; transpose
(let ((x ($tensor '((1 2 3) (4 5 6)))))
  (print x)
  (print ($transpose x)))
(let ((x ($tensor 3 4)))
  ($zero x)
  ($fill ($select x 1 2) 7)
  (print x)
  (let ((y ($transpose x)))
    (print y)
    ($fill ($select y 1 2) 8)
    (print y)
    (print x)))

;; permute
(let ((x ($tensor 3 4 2 5)))
  (print ($size x))
  (print ($size ($permute x 1 2 0 3))))

;; unfold
(let ((x ($tensor 7)))
  (loop :for i :from 1 :to 7 :do (setf ($ x (1- i)) i))
  (print x)
  (print ($unfold x 0 2 1))
  (print ($unfold x 0 2 2)))

;; apply
(let ((x ($zeros 3 3)))
  ($apply (lambda (i v) (if (< i 5) i v)) x)
  (print x)
  ($apply (lambda (i v) (declare (ignore i)) (sin v)) x)
  (print x))

;; map
(let ((x ($tensor 3 3))
      (y ($tensor 9))
      (z ($tensor '(0 1 2 3 4 5 6 7 8))))
  (loop :for i :from 1 :to 9 :do (setf ($ ($storage x) (1- i)) i
                                       ($ ($storage y) (1- i)) i))
  (print x)
  (print y)
  (print z)
  ($map (lambda (i vx vy) (declare (ignore i)) (* vx vy)) x y)
  (print x)
  ($map (lambda (i xx yy zz) (declare (ignore i)) (+ xx yy zz)) x y z)
  (print x))

;; split
(let ((x ($tensor 3 4 5)))
  (print ($split x 2 0))
  (print ($split x 2 1))
  (print ($split x 2 2)))

;; chunk
(let ((x ($tensor 3 4 5)))
  (print ($chunk x 2 0))
  (print ($chunk x 2 1))
  (print ($chunk x 2 2)))

;; concat
(print ($concat 0 ($ones 3) ($zeros 3)))
(print ($concat 1 ($ones 3) ($zeros 3)))
(print ($concat 0 ($ones 3 3) ($zeros 1 3)))
(print ($concat 1 ($ones 3 4) ($zeros 3 2)))

;; diag
(print ($diag ($tensor '(1 2 3 4))))
(print ($diag ($ones 3 3)))

;; eye
(print ($eye 2))
(print ($eye 3 4))
(print ($eye ($bytex 3 3)))
(print ($eye ($bytex 10 10) 3 4))

;; histc
(print ($histc ($tensor '(1 2 3 4 4 5 6 10 9 2 3 4 1 2 3 4 5 6 7 8 9 10))))

;; bhistc
(print ($bhistc ($tensor '((2 4 2 2 5 4) (3 5 1 5 3 5) (3 4 2 5 5 1))) :n 5 :min 1 :max 5))

;; linspace
(print ($linspace 1 2))
(print ($linspace 1 2 :n 11))

;; logspace
(print ($logspace 1 2))
(print ($logspace 1 2 :n 11))

;; multinomial
(print ($multinomial ($tensor '(1 1 0.5 0)) 5 :replacement t))
(print ($multinomial ($tensor '((1 1 0.5 0) (0.1 0.3 0.6 0.9))) 5 :replacement t))

;; rand
(print ($rand 3 3))

;; randn
(print ($rand 2 4))

;; range
(print ($range 2 5))
(print ($range 2 5 :step 1.2))

;; randperm
(print ($randperm 10))
(print ($randperm 5))

;; reshape
(let ((x ($ones 2 3))
      (y nil))
  (print x)
  (setf y ($reshape x 3 2))
  (print y)
  ($fill y 2)
  (print y)
  (print x))

;; tril and triu
(let ((x ($ones 4 4)))
  (print ($tril x))
  (print ($tril x -1))
  (print ($triu x))
  (print ($triu x 1)))

;; abs
(let ((x ($tensor '((-1 2) (3 -4)))))
  (print x)
  (print ($abs x))
  (print x)
  (print ($abs! x))
  (print x))

;; sign
(let ((x ($tensor '((-1 2) (3 -4)))))
  (print x)
  (print ($sign x))
  (print x)
  (print ($sign! x))
  (print x))

;; acos
(let ((x ($tensor '((-1 1) (1 -1)))))
  (print x)
  (print ($acos x))
  (print x)
  (print ($acos! x))
  (print x))

;; asin
(let ((x ($tensor '((-1 1) (1 -1)))))
  (print x)
  (print ($asin x))
  (print x)
  (print ($asin! x))
  (print x))

;; atan
(let ((y ($tensor '((-11 1) (1 -11)))))
  (print y)
  (print ($atan y))
  (print y)
  (print ($atan! y))
  (print y))

;; atan2
(let ((y ($tensor '((-11 1) (1 -11))))
      (x ($tensor '((1 1) (1 1)))))
  (print y)
  (print ($atan2 y x))
  (print y)
  (print ($atan2! y x))
  (print y))

;; ceil
(let ((x ($tensor '((1 1.1 1.7) (-0.8 -1.1 -2.3)))))
  (print x)
  (print ($ceil x))
  (print x)
  (print ($ceil! x))
  (print x))

;; cos
(let ((x ($tensor '((-3.14 0) (3.14 0)))))
  (print x)
  (print ($cos x))
  (print x)
  (print ($cos! x))
  (print x))

;; cosh
(let ((x ($tensor '((-3.14 0) (3.14 0)))))
  (print x)
  (print ($cosh x))
  (print x)
  (print ($cosh! x))
  (print x))

;; exp
(let ((x ($tensor '((0 1 2) (-1 -2 -3)))))
  (print x)
  (print ($exp x))
  (print x)
  (print ($exp! x))
  (print x))

;; floor
(let ((x ($tensor '((1 1.1 1.7) (-0.8 -1.1 -2.3)))))
  (print x)
  (print ($floor x))
  (print x)
  (print ($floor! x))
  (print x))

;; log
(let ((x ($exp ($tensor '((0 1 2) (-1 -2 -3))))))
  (print x)
  (print ($log x))
  (print x)
  (print ($log! x))
  (print x))

;; log1p
(let ((x ($exp ($tensor '((0 1 2) (-1 -2 -3))))))
  (print x)
  (print ($log1p x))
  (print x)
  (print ($log1p! x))
  (print x))

;; neg
(let ((x ($tensor '((0 1 2) (-1 -2 -3)))))
  (print x)
  (print ($neg x))
  (print x)
  (print ($neg! x))
  (print x))

;; cinv
(let ((x ($tensor '((3 2 1) (-1 -2 -3)))))
  (print x)
  (print ($cinv x))
  (print x)
  (print ($cinv! x))
  (print x))

;; expt
(let ((x ($tensor '((2 3) (1 2))))
      (y ($tensor '((2 2) (3 3))))
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
(let ((x ($tensor '((1.1 1.8) (-1.1 -1.8)))))
  (print x)
  (print ($round x))
  (print x)
  (print ($round! x))
  (print x))

;; sin
(let ((x ($tensor '((-3.14 0) (3.14 0)))))
  (print x)
  (print ($sin x))
  (print x)
  (print ($sin! x))
  (print x))

;; sinh
(let ((x ($tensor '((-3.14 0) (3.14 0)))))
  (print x)
  (print ($sinh x))
  (print x)
  (print ($sinh! x))
  (print x))

;; sqrt
(let ((x ($tensor '((1 2 4) (3 5 9)))))
  (print x)
  (print ($sqrt x))
  (print x)
  (print ($sqrt! x))
  (print x))

;; rsqrt
(let ((x ($tensor '((1 2 4) (3 5 9)))))
  (print x)
  (print ($rsqrt x))
  (print x)
  (print ($rsqrt! x))
  (print x))

;; tan
(let ((x ($tensor '((1 2) (3 4)))))
  (print x)
  (print ($tan x))
  (print x)
  (print ($tan! x))
  (print x))

;; tanh
(let ((x ($tensor '((1 2) (-3 -4)))))
  (print x)
  (print ($tanh x))
  (print x)
  (print ($tanh! x))
  (print x))

;; sigmoid
(let ((x ($tensor '((-2 -1) (1 2)))))
  (print x)
  (print ($sigmoid x))
  (print x)
  (print ($sigmoid! x))
  (print x))

;; equal
(let ((x ($tensor '(1 2 3)))
      (y ($tensor '(1 2 3)))
      (z ($tensor '(1 2 3))))
  (print ($equal x y))
  (print ($= x y z)))

;; add, sub, and mul
(let ((x ($tensor '((1 2) (3 4))))
      (y ($tensor '((2 3) (4 5))))
      (a 10))
  (print ($add x a))
  (print ($add x y))
  (print ($sub x a))
  (print ($sub x y))
  (print ($mul x a))
  (print ($mul x y)))

;; clamp
(let ((x ($tensor '((1 2 3 4 5) (2 3 4 5 6) (3 4 5 6 7))))
      (min 2)
      (max 5))
  (print x)
  (print ($clamp x min max))
  (print x)
  (print ($clamp! x min max))
  (print x))

;; add-cmul
(let ((x ($tensor 2 2))
      (y ($tensor 4))
      (z ($tensor 2 2)))
  ($fill x 2)
  ($fill y 3)
  ($fill z 5)
  (print ($addcmul x 2 y z)))

;; div
(let ((x ($ones 2 2))
      (y ($range 1 4)))
  (print ($div x y)))

;; shift
(let ((x ($long ($ones 2 2)))
      (y (-> ($long ($range 1 4)) ($resize '(2 2)))))
  (print ($<< x y))
  ($set x (-> ($longx 2 2)
              ($fill 32)))
  (print ($>> x y)))

;; add-cdiv
(let ((x (-> ($tensor 2 2) ($fill 1)))
      (y ($range 1 4))
      (z (-> ($tensor 2 2) ($fill 5))))
  (print ($addcdiv x 2 y z)))

;; fmod, remainder
(let ((x ($tensor '(-3 3))))
  (print ($fmod x 2))
  (print ($fmod x -2))
  (print ($rem x 2))
  (print ($rem x -2))
  (print ($fmod ($tensor '((3 3) (-3 -3))) ($tensor '((2 -2) (2 -2)))))
  (print ($rem ($tensor '((3 3) (-3 -3))) ($tensor '((2 -2) (2 -2))))))

;; bitand
(let ((x (-> ($longx 4) ($fill 6)))
      (y ($longx '(1 2 4 8))))
  (print ($bitand x y)))

;; bitor
(let ((x (-> ($longx 4) ($fill 3)))
      (y ($longx '(1 2 4 8))))
  (print ($bitor x y)))

;; bitxor
(let ((x (-> ($longx 4) ($fill 15)))
      (y ($longx '(1 2 4 8))))
  (print ($bitxor x y)))

;; dot
(let ((x (-> ($tensor 2 2) ($fill 3)))
      (y (-> ($tensor 4) ($fill 2))))
  (print ($dot x y)))

;; add-mv
(let ((y ($zeros 3))
      (m (-> ($tensor 3 2) ($fill 3)))
      (x (-> ($tensor 2) ($fill 2))))
  (print ($addmv 1 y 1 m x)))

;; add-r
(let ((x ($range 1 3))
      (y ($range 1 2))
      (m ($zeros 3 2)))
  (print ($addr! 1 m 1 x y))
  (print ($addr! 2 m 1 x y)))

;; add-mm
(let ((c ($zeros 4 4))
      (a (-> ($range 1 12) ($resize '(4 3))))
      (b (-> ($range 1 12) ($resize '(3 4)))))
  (print ($addmm 1 c 1 a b)))

;; add-bmm
(let ((c ($zeros 4 4))
      (ba (-> ($range 1 24) ($resize '(2 4 3))))
      (bb (-> ($range 1 24) ($resize '(2 3 4)))))
  (print ($addbmm 1 c 1 ba bb)))

;; badd-bmm
(let ((bc ($zeros 2 4 4))
      (ba (-> ($range 1 24) ($resize '(2 4 3))))
      (bb (-> ($range 1 24) ($resize '(2 3 4)))))
  (print ($baddbmm 1 bc 1 ba bb)))

;; operators
(print ($+ 5 ($rand 3)))
(let ((x (-> ($tensor 2 2) ($fill 2)))
      (y (-> ($tensor 4) ($fill 3))))
  (print ($+ x y))
  (print ($- y x))
  (print ($+ x 3))
  (print ($- x)))
(let ((m (-> ($tensor 2 2) ($fill 2)))
      (n (-> ($tensor 2 4) ($fill 3)))
      (x (-> ($tensor 2) ($fill 4)))
      (y (-> ($tensor 2) ($fill 5))))
  (print ($* x y))
  (print ($* m x))
  (print ($* m n)))
(print ($/ ($ones 2 2) 3))

;; cross
(let ((x ($randn 4 3))
      (y ($randn 4 3)))
  (print x)
  (print y)
  (print ($cross x y 1))
  (print ($cross! x y 1))
  (print x))

;; cumulative product
(let ((x ($range 1 5))
      (m ($longx '((1 4 7) (2 5 8) (3 6 9)))))
  (print x)
  (print ($cumprd x))
  (print m)
  (print ($cumprd m))
  (print ($cumprd m :dimension 1)))

;; cumulative sum
(let ((x ($range 1 5))
      (m ($longx '((1 4 7) (2 5 8) (3 6 9)))))
  (print x)
  (print ($cumsum x))
  (print m)
  (print ($cumsum m))
  (print ($cumsum m :dimension 1)))

;; max and min
(let ((x ($randn 4 4))
      (indices ($longx)))
  (print x)
  (print ($max x))
  (print ($min x))
  (print ($max x :dimension 0 :indices indices))
  (print indices)
  (print ($max x :dimension 1 :indices indices))
  (print indices)
  (print ($min x :dimension 0 :indices indices))
  (print indices)
  (print ($min x :dimension 1 :indices indices))
  (print indices))

;; mean
(let ((x ($randn 3 4)))
  (print x)
  (print ($mean x))
  (print ($mean x :dimension 0))
  (print ($mean x :dimension 1)))

;; cmax
(let ((a ($tensor '(1 2 3)))
      (b ($tensor '(3 2 1))))
  (print ($cmax a b))
  (print ($cmax a b 2 3)))

;; cmin
(let ((a ($tensor '(1 2 3)))
      (b ($tensor '(3 2 1))))
  (print ($cmin a b))
  (print ($cmin a b 2 3)))

;; median
(let ((x ($randn 3 4))
      (indices ($longx)))
  (print x)
  (print ($median x))
  (print ($median x :dimension 0 :indices indices))
  (print indices)
  (print ($median x :dimension 1 :indices indices))
  (print indices))

;; product
(let ((a ($tensor '(((1 2) (3 4)) ((5 6) (7 8))))))
  (print a)
  (print ($prd a :dimension 0))
  (print ($prd a :dimension 1)))

;; sort
(let ((x ($randn 3 3))
      (indices ($longx)))
  (print x)
  (print ($sort x :indices indices))
  (print indices))

;; conv2
(let ((x ($rand 100 100))
      (k ($rand 10 10)))
  (print ($size ($conv2 x k)))
  (print ($size ($conv2 x k :vf :f))))
(let ((x ($rand 500 100 100))
      (k ($rand 500 10 10)))
  (print ($size ($conv2 x k)))
  (print ($size ($conv2 x k :vf :f))))

;; conv3 - slow, in this laptop, it takes ~6secs
(let ((x ($rand 100 100 100))
      (k ($rand 10 10 10)))
  (print ($size ($conv3 x k)))
  (print ($size ($conv3 x k :vf :f))))

;; gesv
(let ((a (-> ($tensor '((6.80 -2.11  5.66  5.97  8.23)
                        (-6.05 -3.30  5.36 -4.44  1.08)
                        (-0.45  2.58 -2.70  0.27  9.04)
                        (8.32  2.71  4.35  -7.17  2.14)
                        (-9.67 -5.14 -7.26  6.08 -6.87)))
             ($transpose)))
      (b (-> ($tensor '((4.02  6.19 -8.22 -7.57 -3.03)
                        (-1.56  4.00 -8.67  1.75  2.86)
                        (9.81 -4.09 -4.57 -8.61  8.99)))
             ($transpose)))
      (x ($tensor))
      (lu ($tensor)))
  (print a)
  (print b)
  (print ($gesv b a :x x :lu lu))
  (print x)
  (print ($* a x))
  (print ($dist b ($* a x))))

;; trtrs
(let ((a (-> ($tensor '((6.80 -2.11  5.66  5.97  8.23)
                        (0 -3.30  5.36 -4.44  1.08)
                        (0  0 -2.70  0.27  9.04)
                        (0  0  0  -7.17  2.14)
                        (0  0  0  0 -6.87)))))
      (b (-> ($tensor '((4.02  6.19 -8.22 -7.57 -3.03)
                        (-1.56  4.00 -8.67  1.75  2.86)
                        (9.81 -4.09 -4.57 -8.61  8.99)))
             ($transpose)))
      (x ($tensor))
      (ra ($tensor)))
  (print a)
  (print b)
  (print ($trtrs b a :x x :ra ra))
  (print x)
  (print ($* a x))
  (print ($dist b ($* a x))))

;; potrf
(let ((a ($tensor '((1.2705  0.9971  0.4948  0.1389  0.2381)
                    (0.9971  0.9966  0.6752  0.0686  0.1196)
                    (0.4948  0.6752  1.1434  0.0314  0.0582)
                    (0.1389  0.0686  0.0314  0.0270  0.0526)
                    (0.2381  0.1196  0.0582  0.0526  0.3957))))
      (u ($tensor))
      (l ($tensor)))
  (print ($potrf a :out u))
  (print ($* ($transpose u) u))
  (print ($potrf a :up nil :out l))
  (print ($* l ($transpose l)))
  (print u)
  (print l))

;; pstrf
(let ((a ($tensor '((1.2705  0.9971  0.4948  0.1389  0.2381)
                    (0.9971  0.9966  0.6752  0.0686  0.1196)
                    (0.4948  0.6752  1.1434  0.0314  0.0582)
                    (0.1389  0.0686  0.0314  0.0270  0.0526)
                    (0.2381  0.1196  0.0582  0.0526  0.3957))))
      (u ($tensor))
      (l ($tensor))
      (piv ($intx))
      (ap nil))
  (print ($pstrf a :out u :piv piv))
  (print u)
  (print piv)
  (setf ap ($* ($transpose u) u))
  (print ap)
  (print a)
  ($index-copy ap 0 ($long piv) ($clone ap))
  ($index-copy ap 1 ($long piv) ($clone ap))
  (print ap)
  (print ($norm ($- a ap)))
  (print ($pstrf a :out l :up nil :piv piv))
  (print l)
  (print piv)
  (setf ap ($* l ($transpose l)))
  (print ap)
  (print a)
  ($index-copy ap 0 ($long piv) ($clone ap))
  ($index-copy ap 1 ($long piv) ($clone ap))
  (print ap)
  (print ($norm ($- a ap))))

;; potrs
(let ((a ($tensor '((1.2705  0.9971  0.4948  0.1389  0.2381)
                    (0.9971  0.9966  0.6752  0.0686  0.1196)
                    (0.4948  0.6752  1.1434  0.0314  0.0582)
                    (0.1389  0.0686  0.0314  0.0270  0.0526)
                    (0.2381  0.1196  0.0582  0.0526  0.3957))))
      (b ($tensor '((0.6219  0.3439  0.0431)
                    (0.5642  0.1756  0.0153)
                    (0.2334  0.8594  0.4103)
                    (0.7556  0.1966  0.9637)
                    (0.1420  0.7185  0.7476))))
      (cholesky ($tensor))
      (solve ($tensor)))
  ($potrf a :out cholesky)
  (print cholesky)
  (print a)
  (print ($* ($transpose cholesky) cholesky))
  (print ($dist a ($* ($transpose cholesky) cholesky)))
  ($potrs b cholesky :out solve)
  (print solve)
  (print b)
  (print ($* a solve))
  (print ($dist b ($* a solve))))

;; potri
(let ((a ($tensor '((1.2705  0.9971  0.4948  0.1389  0.2381)
                    (0.9971  0.9966  0.6752  0.0686  0.1196)
                    (0.4948  0.6752  1.1434  0.0314  0.0582)
                    (0.1389  0.0686  0.0314  0.0270  0.0526)
                    (0.2381  0.1196  0.0582  0.0526  0.3957))))
      (cholesky ($tensor))
      (inv ($tensor)))
  ($potrf a :out cholesky)
  (print cholesky)
  ($potri cholesky :out inv)
  (print inv)
  (print ($* a inv))
  (print ($dist ($eye 5 5) ($* a inv))))

;; gels
(let ((a (-> ($tensor '((1.44 -9.96 -7.55  8.34  7.08 -5.45)
                        (-7.84 -0.28  3.24  8.09  2.52 -5.70)
                        (-4.39 -3.24  6.27  5.28  0.74 -1.19)
                        (4.53  3.83 -6.64  2.06 -2.47  4.70)))
             ($transpose)))
      (b (-> ($tensor '((8.58  8.26  8.48 -5.28  5.72  8.93)
                        (9.35 -4.43 -0.70 -0.26 -7.36 -2.52)))
             ($transpose)))
      (x ($tensor))
      (f ($tensor)))
  (print a)
  (print b)
  ($gels b a :out x :factor f)
  (print x)
  (print f)
  (print ($dist b ($* a x))))

;; symeig
(let ((a (-> ($tensor '((1.96  0.00  0.00  0.00  0.00)
                        (-6.49  3.80  0.00  0.00  0.00)
                        (-0.47 -6.39  4.17  0.00  0.00)
                        (-7.20  1.50 -1.51  5.70  0.00)
                        (-0.65 -6.34  2.67  1.80 -7.10)))
             ($transpose)))
      (e ($tensor))
      (v ($tensor)))
  (print a)
  ($symeig a :e e)
  (print e)
  ($symeig a :e e :v v :ev t)
  (print e)
  (print v)
  (print ($* v ($diag e) ($transpose v)))
  (print ($dist a ($triu ($* v ($diag e) ($transpose v))))))

;; eig
(let ((a (-> ($tensor '((1.96  0.00  0.00  0.00  0.00)
                        (-6.49  3.80  0.00  0.00  0.00)
                        (-0.47 -6.39  4.17  0.00  0.00)
                        (-7.20  1.50 -1.51  5.70  0.00)
                        (-0.65 -6.34  2.67  1.80 -7.10)))
             ($transpose)))
      (b nil)
      (e ($tensor))
      (v ($tensor)))
  (setf b ($+ a ($transpose ($triu a 1))))
  (print b)
  ($eig b :e e)
  (print e)
  ($eig b :e e :v v :ev t)
  (print e)
  (print v)
  (print ($* v ($diag ($select e 1 0)) ($transpose v)))
  (print ($dist b ($* v ($diag ($select e 1 0)) ($transpose v)))))

;; svd
(let ((a (-> ($tensor '((8.79  6.11 -9.15  9.57 -3.49  9.84)
                        (9.93  6.91 -7.93  1.64  4.02  0.15)
                        (9.83  5.04  4.86  8.83  9.80 -8.99)
                        (5.45 -0.27  4.85  0.74 10.00 -6.02)
                        (3.16  7.98  3.01  5.80  4.27 -5.31)))
             ($transpose)))
      (u ($tensor))
      (s ($tensor))
      (v ($tensor)))
  (print a)
  ($svd a :u u :s s :v v)
  (print u)
  (print s)
  (print v)
  (print ($* u ($diag s) ($transpose v)))
  (print ($dist a ($* u ($diag s) ($transpose v)))))

;; inverse
(let ((a ($rand 10 10)))
  (print a)
  (print ($* a ($inverse a)))
  (print ($dist ($eye 10 10) ($* a ($inverse a)))))

;; qr
(let ((a ($tensor '((12 -51 4) (6 167 -68) (-4 24 -41))))
      (q ($tensor))
      (r ($tensor)))
  (print a)
  ($qr a :q q :r r)
  (print q)
  (print r)
  (print ($round ($* q r)))
  (print ($* ($transpose q) q)))

;; lt
(print ($lt ($tensor '((1 2) (3 4))) ($tensor '((2 1) (4 3)))))
(let ((a ($rand 10))
      (b ($rand 10)))
  (print a)
  (print b)
  (print ($lt a b))
  (print ($ a ($gt a b)))
  (setf ($ a ($gt a b)) 123)
  (print a))
