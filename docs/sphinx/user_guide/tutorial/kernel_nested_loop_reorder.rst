.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _tut-kernelnestedreorder-label:

-----------------------------------------------------------
Basic ``RAJA::kernel`` Mechanics and Nested Loop Ordering
-----------------------------------------------------------

This section contains an exercise file ``RAJA/exercises/kernelintro-nested-loop-reorder.cpp``
for you to work through if you wish to get some practice with RAJA. The
file ``RAJA/exercises/kernelintro-nested-loop-reorder_solution.cpp`` contains 
complete working code for the examples discussed in this section. You can use 
the solution file to check your work and for guidance if you get stuck. To build
the exercises execute ``make kernelintro-nested-loop-reorder`` and ``make kernelintro-nested-loop-reorder_solution``
from the build directory.

Key RAJA features shown in this section are:

  * ``RAJA::kernel`` loop iteration templates and execution policies
  * Nested loop reordering
  * RAJA strongly-types indices

The examples in this
section show the nested loop reordering process in more detail. 
Specifically, we describe how to reorder execution policy statements, which
is conceptually analogous to how one would reorder for-loops in a C-style loop
nest. We also introduce strongly-typed index variables that can help users 
write correct nested loop code with RAJA. The examples do not perform any 
computation; each kernel simply prints out the loop indices in the 
order that the iteration spaces are traversed. Thus, only sequential execution 
policies are used to avoid complications resulting from print statements
used in parallel programs. The mechanics shown here work the same way for 
parallel RAJA execution policies.

Before we dive into code, we reiterate important features that 
represent the main differences between nested-loop RAJA and the 
``RAJA::forall`` construct for simple, non-nested loop kernels: 

  * An index space (e.g., range segment) and lambda index argument are 
    required for each level in a loop nest. This example contains
    triply-nested loops, so there will be three ranges and three index 
    arguments.

  * The index spaces for the nested loop levels are specified in a RAJA tuple 
    object. The order of spaces in the tuple must match the order of index 
    arguments to the lambda for this to be correct in general. RAJA provides 
    strongly-typed indices to help with this, which we show below.

  * An execution policy is required for each level in a loop nest. These
    are specified as nested statements in the ``RAJA::KernelPolicy`` type.

  * The loop nest ordering is specified in the nested kernel policy --
    the first ``statement::For`` type identifies the outermost loop, the 
    second ``statement::For`` type identifies the loop nested inside the 
    outermost loop, and so on.

We begin by defining three named **strongly-typed** variables for the loop 
index variables (i, j, k):

.. literalinclude:: ../../../../exercises/kernelintro-nested-loop-reorder_solution.cpp
   :start-after: _raja_typed_indices_start
   :end-before: _raja_typed_indices_end
   :language: C++

Specifically, the 'i' index variable type is ``IIDX``, the 'j' index variable
is ``JIDX``, and the 'k' variable is ``KIDX``, which are aliases to
``int`` type.

We also define [min, max) intervals for each loop index:

.. literalinclude:: ../../../../exercises/kernelintro-nested-loop-reorder_solution.cpp
   :start-after:  _range_min_max_start
   :end-before:  _range_min_max_end
   :language: C++

and three corresponding **typed** range segments which bind the ranges to the
index variable types via template specialization:

.. literalinclude:: ../../../../exercises/kernelintro-nested-loop-reorder_solution.cpp
   :start-after: _raja_typed_index_ranges_start
   :end-before: _raja_typed_index_ranges_end
   :language: C++

When these features are used as in this example, the compiler will 
generate error messages if the lambda expression index argument ordering
and types do not match the index ordering in the tuple. This is illustrated
at the end of this section.

We begin with a C-style loop nest with 'i' in the inner loop, 'j' in the
middle loop, and 'k' in the outer loop, which prints the (i, j, k) triple 
in the inner loop body:

.. literalinclude:: ../../../../exercises/kernelintro-nested-loop-reorder_solution.cpp
   :start-after: _cstyle_kji_loops_start
   :end-before: _cstyle_kji_loops_end
   :language: C++

The ``RAJA::kernel`` version of this is:

.. literalinclude:: ../../../../exercises/kernelintro-nested-loop-reorder_solution.cpp
   :start-after: _raja_kji_loops_start
   :end-before: _raja_kji_loops_end
   :language: C++

The integer template parameters in the ``RAJA::statement::For`` types 
represent the lambda expression index argument and the range types in the
iteration space tuple argument to ``RAJA::kernel``.

Both kernels generate the same output, as expected::

  (I, J, K)
  ---------
  (0, 1, 2) 
  (1, 1, 2) 
  (0, 2, 2) 
  (1, 2, 2) 
  (0, 1, 3) 
  (1, 1, 3) 
  (0, 2, 3) 
  (1, 2, 3) 

which you can see by running the exercise code.

Here, the ``RAJA::kernel`` execution template takes two arguments: a tuple of 
ranges, one for each of the three levels in the loop nest, and the lambda 
expression loop body. Note that the lambda has an index argument for each 
range and that their order and types match. This is required for the code to
compile.

.. note:: RAJA provides mechanisms to explicitly specify which loop variables, 
          for example, and in which order they appear in a lambda expression
          argument list. Please refer to :ref:`loop_elements-kernel-label`
          for more information.

The execution policy for the loop nest is specified in the 
``RAJA::KernelPolicy`` type. The policy uses two statement types:
``RAJA::statement::For`` and ``RAJA::statement::Lambda``.

The ``RAJA::statement::Lambda`` is used to generate code that invokes the
lambda expression. The '0' template parameter refers to the index of the 
lambda expression in the ``RAJA::kernel`` argument list following the
iteration space tuple. Since there is only one lambda expression, we reference
it with the '0' identifier. Sometimes more complicated kernels require multiple
lambda expressions, so we need a way to specify where they will appear in the
generated executable code. We show examples of this in the matrix transpose
discussion later in the tutorial.

Each level in the loop nest is identified by a
``RAJA::statement::For`` type, which identifies the iteration space and
execution policy for the level. Here, each level uses a 
sequential execution policy, which is for illustration purposes.
The integer that appears as the first template argument to each 
``RAJA::statement::For`` type corresponds to the index of a range in the tuple 
and also to the associated lambda index argument; i.e., '0' for 'i', 
'1' for 'j', and '2' for 'k'. 

The integer argument to each ``RAJA::statement::For`` type is needed so 
that the levels in the loop nest can be reordered by changing the policy 
while the kernel remains the same. To illustrate, we permute the loop nest 
ordering so that the 'j' loop is the outermost, the 'i' loop is in the middle, 
and the 'k' loop is the innermost with the following policy:

.. literalinclude:: ../../../../exercises/kernelintro-nested-loop-reorder_solution.cpp
   :start-after: _raja_jik_loops_start
   :end-before: _raja_jik_loops_end
   :language: C++

This generates the following output::

  (I, J, K)
  ---------
  (0, 1, 2) 
  (0, 1, 3) 
  (1, 1, 2) 
  (1, 1, 3) 
  (0, 2, 2) 
  (0, 2, 3) 
  (1, 2, 2) 
  (1, 2, 3)

which is the same as the corresponding C-style version:

.. literalinclude:: ../../../../exercises/kernelintro-nested-loop-reorder_solution.cpp
   :start-after: _cstyle_jik_loops_start
   :end-before: _cstyle_jik_loops_end
   :language: C++

Note that we have simply reordered the nesting of the ``RAJA::statement::For``
types in the execution policy. This is analogous to reordering the for-loops 
in C-style version.

For completeness, we permute the loops again so that the 'i' loop 
is the outermost, the 'k' loop is in the middle, and the 'j' loop is the 
innermost with the following policy:

.. literalinclude:: ../../../../exercises/kernelintro-nested-loop-reorder_solution.cpp
   :start-after: _raja_ikj_loops_start
   :end-before: _raja_ikj_loops_end
   :language: C++

The analogous C-style loop nest is:

.. literalinclude:: ../../../../exercises/kernelintro-nested-loop-reorder_solution.cpp
   :start-after: _cstyle_ikj_loops_start
   :end-before: _cstyle_ikj_loops_end
   :language: C++

The output generated by these two kernels is::

  (I, J, K)
  ---------
  (0, 1, 2) 
  (0, 2, 2) 
  (0, 1, 3) 
  (0, 2, 3) 
  (1, 1, 2) 
  (1, 2, 2) 
  (1, 1, 3) 
  (1, 2, 3)

Finally, we show an example that will generate a compilation error because
there is a type mismatch in the ordering of the range segments in the tuple
and the lambda expression argument list.

.. literalinclude:: ../../../../exercises/kernelintro-nested-loop-reorder_solution.cpp
   :start-after: _raja_compile_error_start
   :end-before: _raja_compile_error_end
   :language: C++

Do you see the problem? The last kernel is included in the exercise source
file, so you can see what happens when you attempt to compile it.
