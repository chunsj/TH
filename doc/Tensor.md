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
