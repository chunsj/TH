# Tensor in TH

Tensor is basic data structure in using TH. It is most generic form of vectors and matrices - no,
this is not strict mathematical definitions. You can think of it as generic, multi-dimensional
array containing specified numeric type such as integer, long, single-float, or double-float and
others.

## Tensor Creations

Following method creates a tensor with uninitialized contents.

```lisp
(tensor) ;; creates an empty tensor with default type (single-float or 32bit float)
(tensor 2 3) ;; creates a matrix with size of 2 rows and 3 columns
```

You can create tensors with contents.

```lisp
(tensor '(1 2 3 4)) ;; creates a row vector
(tensor '((1 2 3 4)) ;; creates a matrix with a size of 1x4
(tensor '((1 2) (3 4) (5 6))) ;; creates a matrix with a size of 3x2
```

When given a tensor, the content storage is shared - this is the same mechanism of torch.

```lisp
(let ((x (tensor '(4 3 2 1))))
  (print "X = (4 3 2 1)")
  (print x)
  (print "SAME CONTENT AS X")
  (print (tensor x)))
```

## Tensor Storage

Contents of tensors in storage are stored in flat memory area. What makes this ordinary memory
work as something special like tensor is how the stored contents are dealt with. Tensors have
their size and strides.

You can directly specify size and stride information during creation; for example, you can create
an uninitialized tensor of 2 rows and 2 columns whose strides are 2 and 1. Thinks of strides as
how flat, 1-D memory space can be viewed as 2-D matrix.

```lisp
(tensor '(2 2) '(2 1))
```

You can create tensors of specific type; default type is 32-bit float or single-float.

```lisp
(tensor.byte '(1 2 3 4))
(tensor.char '(1 2 3 4))
(tensor.short '(1 2 3 4))
(tensor.int '(1 2 3 4))
(tensor.long '(1 2 3 4))
(tensor.float '(1 2 3 4))
(tensor.double '(1 2 3 4))
```

Tensor types are convertible.
```lisp
(tensor.byte (tensor.double '((1.234 2.345) (3.456 4.567))))
(tensor.double (tensor.int '((1 2 3) (4 5 6))))
```

Cloning a tensor is creating a new tensor with independent storage with same contents.

```lisp
(let* ((x (tensor '((1 2) (3 4))))
       (x2 ($clone x))))
```

Make storage memory as contiguously allocated one if it is not.

```lisp
($contiguous! atensor)
```

Check whether given data is tensor or not.

```lisp
($tensorp 0)
($tensorp (tensor 2 3 4))
```
