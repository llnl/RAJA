.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _tut-matrixmultiply-label:

------------------------------------
Matrix Multiplication: RAJA::kernel
------------------------------------

The file ``RAJA/examples/tut_matrix-multiply.cpp`` contains the complete 
working code for all examples described in this section, plus others that
show a variety of ``RAJA::kernel`` execution policy types. It also contains 
raw CUDA and HIP versions of the kernel for comparison.

Key RAJA features shown in the following examples:

  * ``RAJA::kernel`` template for nested-loop execution
  * RAJA kernel execution policies
  * ``RAJA::View`` multi-dimensional data access
  * RAJA nested-loop interchange 
  * Specifying lambda arguments through statements

In this example, we present different ways to perform multiplication of two 
square matrices 'A' and 'B' of dimension N x N and store the result in matrix 
'C'. To motivate the use of the ``RAJA::View`` abstraction that we use, 
we define the following macros to access the matrix entries in the 
C-version:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_macros_start
   :end-before: _matmult_macros_end
   :language: C++

Then, a typical C-style sequential matrix multiplication operation might
look like this:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_cstyle_start
   :end-before: _matmult_cstyle_end
   :language: C++

For the RAJA variants of the matrix multiple operation presented below,
we use ``RAJA::View`` objects, which allow us to access matrix
entries in a multi-dimensional manner similar to the C-style version that
uses macros. We create a two-dimensional N x N 'view'
for each matrix:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_views_start
   :end-before: _matmult_views_end
   :language: C++

We show the most basic RAJA view usage here -- to simplify multi-dimensional
array indexing. RAJA views can be used to abstract a variety of different 
data layouts and access patterns, including stride permutations, offsets, etc. 
For more information about RAJA views, see :ref:`feat-view-label`.

We also use the following ``RAJA::TypedRangeSegment`` objects to define the 
matrix row and column and dot product iteration spaces:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_ranges_start
   :end-before: _matmult_ranges_end
   :language: C++

.. _matmultkernel-label:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Basic RAJA::kernel Variants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, we show how to cast the matrix-multiplication operation 
using the ``RAJA::kernel`` interface, which was introduced in
:ref:`loop_elements-kernel-label`. We first present a complete example, and
then describe its key elements, noting important differences between
``RAJA::kernel`` and ``RAJA::forall`` loop execution interfaces. 

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_basickernel_start
   :end-before: _matmult_basickernel_end
   :language: C++

Here, ``RAJA::kernel`` expresses the outer 'row' and 'col' loops; 
the inner 'k' loop is included in the lambda expression for the loop body. 
Note that the ``RAJA::kernel`` template takes two arguments. Similar to 
``RAJA::forall``, the first argument describes the iteration space and the 
second argument is the lambda loop body. Unlike ``RAJA::forall``, the 
iteration space for ``RAJA::kernel`` is defined as a *tuple* of ranges 
(created via the ``RAJA::make_tuple`` method), one for the 'col' loop and 
one for the 'row' loop. Also, the lambda expression takes an iteration index 
argument for each entry in the iteration space tuple. 

.. note :: The number and order of lambda arguments must match the number and
           order of the elements in the tuple for this type of ``RAJA::kernel``
           usage to be correct.

Another important difference between ``RAJA::forall`` and ``RAJA::kernel`` 
involves the execution policy template parameter. The execution policy defined 
by the ``RAJA::KernelPolicy`` type shown here specifies a policy for each level 
in the loop nest via nested ``RAJA::statement::For`` types. Here, the row and 
column loops will both execute sequentially. The integer that appears as the 
first template parameter to each 'For' statement corresponds to the position of 
a range in the iteration space tuple and also to the associated iteration 
index argument to the lambda. Here, '0' is the 'col' range and '1' is the 
'row' range because that is the order those ranges appear in the tuple. The 
innermost type ``RAJA::statement::Lambda<0>`` indicates that the first lambda
expression (the only one in this case) argument passed to the 
``RAJA::kernel`` method will be invoked inside the inner loop.

The integer arguments to the ``RAJA::statement::For`` types are needed to 
enable the desired kernel execution pattern and potential transformations,
without changing the kernel code. Since the kernel policy is a single unified 
construct, it can be used to parallelize the nested loop iterations together, 
which we show next. 

If we want to execute the row loop using OpenMP multithreaded parallelism 
and keep the column loop sequential, the policy we would use is:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_ompkernel_start
   :end-before: _matmult_ompkernel_end
   :language: C++

To swap the loop nest ordering and keep the same execution policy on each loop,
we would use the following policy, which swaps the ``RAJA::statement::For`` 
types. The inner loop is now the 'row' loop and is run in parallel; 
the outer loop is now the 'col' loop and is run sequentially:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_ompkernel_swap_start
   :end-before: _matmult_ompkernel_swap_end
   :language: C++

.. note:: It is important to emphasize that these kernel transformations, 
          and others, can be done by switching the ``RAJA::KernelPolicy`` 
          type with no changes to the loop kernel code.

In :ref:`tut-kernelnestedreorder-label`, we provide a more detailed discussion 
of the mechanics of loop nest ordering. Next, we show other variations of the 
matrix multiplication kernel that illustrate other ``RAJA::kernel`` features. 

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
More Complex RAJA::kernel Variants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The matrix multiplication kernel variations described in this section use
execution policies to express the outer row and col loops as well as the 
inner dot product loop using the RAJA kernel interface. They illustrate more 
complex policy examples and show additional RAJA kernel features.

The first example uses sequential execution for all loops:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_3lambdakernel_seq_start
   :end-before: _matmult_3lambdakernel_seq_end
   :language: C++

Note that we use a ``RAJA::kernel_param`` method to execute the kernel. It is
similar to ``RAJA::kernel`` except that it accepts a tuple as the second
argument (between the iteration space tuple and the lambda expressions). In
general, the tuple is a set of *parameters* that can be used in the lambda
expressions comprising the kernel. Here, the parameter tuple holds a single 
scalar variable for the dot product of each row-column pair. 

The remaining arguments include a sequence of lambda expressions representing 
different parts of the kernel body. We use three lambda expressions that: 
initialize the dot product variable (lambda 0), define the 'k' inner loop 
row-col dot product operation (lambda 1), and store the computed row-col dot 
product in the proper location in the result matrix (lambda 2). Note that
all lambdas take the same arguments in the same order, which is required for 
the kernel to be well-formed. In addition to the loop index variables, we 
pass the scalar dot product variable into each lambda. This enables the same 
variables to be used in all three lambda expressions. However, observe that
not all lambda expressions use all three index variables. This is the 
result of using the ``RAJA::Params`` and ``RAJA::Segs`` template parameter
types in the ``RAJA::statement::Lambda`` types for lambdas '0' and '2'.
Specifically, ``RAJA::statement::Lambda<0, RAJA::Params<0>>`` indicates that
lambda '0' will take only the scalar parameter as an argument, and 
``RAJA::statement::Lambda<2, RAJA::Segs<0, 1>, RAJA::Params<0>>`` indicates
that lambda '2' will take index values for the column and row ranges and
the scalar parameter as arguments, in that order. Since lambda '1' takes all
arguments, we do not specify them.

Alternatively, the statement to invoke lambda '1' could be augmented to 
specify the arguments it takes:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_3lambdakernel_args_seq_start
   :end-before: _matmult_3lambdakernel_args_seq_end
   :language: C++

The result is the same.

By using ``RAJA::statement::Lambda`` parameters in this way, the code 
potentially indicates more clearly which arguments are used. Of course, this 
makes the execution policy more verbose, but that is typically hidden away 
in a header file, so it need not make the code harder to read. 

.. note::: ``RAJA::Segs`` and ``RAJA::Params`` types can be used in a 
           ``RAJA::statement::Lambda`` type to  identify which segment 
           indices and params are passed as arguments to a lambda expression.

As we noted earlier, the execution policy type passed to the 
``RAJA::kernel_param`` method as a template parameter describes how the 
statements and lambda expressions are assembled to form the complete kernel. 
To illustrate this, we describe various policies that enable the kernel to 
run in different ways. In each case, the ``RAJA::kernel_param`` method call, 
including its arguments is the same. The curious reader may inspect the 
example code in the file noted above to see that this is indeed the case.

Next, we show how to collapse nested loops in an OpenMP parallel region
using a ``RAJA::statement::Collapse`` type in the execution policy. This
allows one to parallelize multiple levels in a loop nest using OpenMP 
directives. The following policy will collapse the two outer 
loops into one OpenMP parallel region:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_3lambdakernel_ompcollapse_start
   :end-before: _matmult_3lambdakernel_ompcollapse_end
   :language: C++

The ``RAJA::ArgList`` type indicates which loops in the nest are to be 
collapsed and their nesting order within the collapse region. The integers
passed to ``ArgList`` are indices of entries in the tuple of iteration spaces 
and indicate inner to outer loop levels when read from right to left. Here, 
'1, 0' indicates that the column loop is the inner loop and the row loop is 
the outer loop. For this transformation there are no ``statement::For`` types
and policies for the individual loop levels inside the OpenMP collapse region. 

Lastly, we show how to use ``RAJA::statement::CudaKernel`` and 
``RAJA::statement::HipKernel`` types to generate GPU kernels launched with
a particular thread-block decomposition.  We reiterate that although the 
policies are different, the kernels themselves are identical to the sequential 
and OpenMP variants above.

Here is a policy that will distribute the row indices across CUDA thread 
blocks and column indices across threads in the x dimension of each block:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_3lambdakernel_cuda_start
   :end-before: _matmult_3lambdakernel_cuda_end
   :language: C++

This is equivalent to defining a CUDA kernel with the lambda body inside
it and defining row and column indices as::

  int row = blockIdx.x;
  int col = threadIdx.x;  

and launching the kernel with appropriate CUDA grid and thread-block dimensions.

The HIP execution policy is similar:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_3lambdakernel_hip_start
   :end-before: _matmult_3lambdakernel_hip_end
   :language: C++


The following policy will tile row and col indices across two-dimensional
CUDA thread blocks with 'x' and 'y' dimensions defined by a 'CUDA_BLOCK_SIZE'
parameter that can be set at compile time. Within each tile, the kernel 
iterates are executed by CUDA threads.

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_3lambdakernel_cudatiled_start
   :end-before: _matmult_3lambdakernel_cudatiled_end
   :language: C++

Note that the tiling mechanism requires a ``RAJA::statement::Tile`` type, 
with a tile size and a tiling execution policy, plus a ``RAJA::statement::For``
type with an execution execution policy for each tile dimension.

The analogous HIP execution policy is:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_3lambdakernel_hiptiled_start
   :end-before: _matmult_3lambdakernel_hiptiled_end
   :language: C++

In :ref:`tut-tiledmatrixtranspose-label` and 
:ref:`tut-matrixtransposelocalarray-label`,
we discuss loop tiling in more detail.

