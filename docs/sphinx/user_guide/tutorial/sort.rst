.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _tut-sort-label:

--------------------------------------------------
Parallel Sort Operations
--------------------------------------------------

This section contains an exercise file ``RAJA/exercises/sort.cpp``
for you to work through if you wish to get some practice with RAJA. The
file ``RAJA/exercises/sort_solution.cpp`` contains complete
working code for the examples discussed in this section. You can use the
solution file to check your work and for guidance if you get stuck. To build
the exercises execute ``make sort`` and ``make sort_solution``
from the build directory.

Key RAJA features shown in this section are:

  * ``RAJA::sort``, ``RAJA::sort_pairs``, ``RAJA::stable_sort``, and ``RAJA::stable_sort_pairs`` operations and execution policies
  * RAJA comparators for different types of sorts; e.g., less, greater

We show examples of RAJA sort operations using multiple RAJA execution
back-ends and describe how different sort orderings can be achieved by 
passing different RAJA comparators to the RAJA sort template methods. Each 
comparator is a template type, where the template argument is the type of 
the values it compares. For a summary of available RAJA sorts, please see 
:ref:`feat-sort-label`.

.. note:: RAJA sort operations use the same execution policy types that
          ``RAJA::forall`` loop execution templates do.

.. note:: RAJA sort operations take 'span' arguments to express the sequential
          index range of array entries used in the sort. Typically, these
          span objects are created using the ``RAJA::make_span`` method
          as shown in the examples below.

Each of the examples below uses the same integer arrays for input
and output values. We set the input array and print them as follows:

.. literalinclude:: ../../../../exercises/sort_solution.cpp
   :start-after: _sort_array_init_start
   :end-before: _sort_array_init_end
   :language: C++

This produces the following sequence of values in the ``in`` array::

   6 7 2 1 0 9 4 8 5 3 4 9 6 3 7 0 1 8 2 5

and the following sequence of (key, value) pairs shown as pairs of values
in the ``in`` and ``in_vals`` arrays, respectively::

   (6,0) (7,0) (2,0) (1,0) (0,0) (9,0) (4,0) (8,0) (5,0) (3,0)
   (4,1) (9,1) (6,1) (3,1) (7,1) (0,1) (1,1) (8,1) (2,1) (5,1)

.. note:: In the following sections, we discuss *stable* and *unstable* sort 
          operations. The difference between them is that a stable sort 
          preserves the relative order of equal elements, with respect to the 
          sort comparator operation, while an unstable sort may not preserve 
          the relative order of equal elements. For the examples below that 
          use integer arrays, there is no way to tell by inspecting the 
          output whether relative ordering is preserved for unstable sorts.
          However, the preservation of relative ordering can be seen in the
          sort pairs examples below.

^^^^^^^^^^^^^^^^
Unstable Sorts
^^^^^^^^^^^^^^^^

A sequential unstable sort operation is performed by:

.. literalinclude:: ../../../../exercises/sort_solution.cpp
   :start-after: _sort_seq_start
   :end-before: _sort_seq_end
   :language: C++

Since no comparator is passed to the sort method, the default 'less' operator
``RAJA::operators::less<int>`` is applied and the result generated in the 
``out`` array is a non-decreasing sequence of values from the ``in`` array; 
i.e.,::

   0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9

We can be explicit about the operation used in the sort operation by passing 
the 'less' operator to the sort method manually:

.. literalinclude:: ../../../../exercises/sort_solution.cpp
   :start-after: _sort_seq_less_start
   :end-before: _sort_seq_less_end
   :language: C++

The result in the ``out`` array is the same as before::

   0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9

An unstable parallel sort operation using OpenMP multithreading is
accomplished similarly by replacing the execution policy type with
and OpenMP policy:

.. literalinclude:: ../../../../exercises/sort_solution.cpp
   :start-after: _sort_omp_less_start
   :end-before: _sort_omp_less_end
   :language: C++

As is common with RAJA, the only difference between this code and
the previous one is that the execution policy is different. If we want to
run the sort on a GPU using CUDA or HIP, we would use a CUDA or HIP execution 
policy. This is shown in examples that follow.

^^^^^^^^^^^^^^^^
Stable Sorts
^^^^^^^^^^^^^^^^

A sequential stable sort (less) operation is performed by:

.. literalinclude:: ../../../../exercises/sort_solution.cpp
   :start-after: _sort_stable_seq_less_start
   :end-before: _sort_stable_seq_less_end
   :language: C++

This generates the following sequence of values in the output array 
as expected based on the examples we discussed above::

   0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9

Note that the stable sort result is the same as the unstable sort in this case
because we are sorting an array of integers. We will show an example of 
sorting pairs later where this is not the case.

Running the same sort operation on a GPU using CUDA is done by:

.. literalinclude:: ../../../../exercises/sort_solution.cpp
   :start-after: _sort_stable_cuda_less_start
   :end-before: _sort_stable_cuda_less_end
   :language: C++

Note that we pass the number of threads per CUDA thread block as the template
argument to the CUDA execution policy as we do when using ``RAJA::forall``.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Other Comparators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using a different comparator operator allows sorting in a different order.
Here is a sequential stable sort that uses the 'greater' operator
``RAJA::operators::greater<int>``:

.. literalinclude:: ../../../../exercises/sort_solution.cpp
   :start-after: _sort_stable_seq_greater_start
   :end-before: _sort_stable_seq_greater_end
   :language: C++

and similarly for HIP:

.. literalinclude:: ../../../../exercises/sort_solution.cpp
   :start-after: _sort_stable_hip_greater_start
   :end-before: _sort_stable_hip_greater_end
   :language: C++

Both of these sorts generate the following sequence of values in 
non-increasing order in the output array::

   9 9 8 8 7 7 6 6 5 5 4 4 3 3 2 2 1 1 0 0

.. note:: * The only operators provided by RAJA that are valid to use in sort
            because they enforce a strict weak ordering of elements for 
            arithmetic types are 'less' and 'greater'. Users may provide other
            operators for different sorting operations. 
          * Also the RAJA CUDA sort back-end only supports RAJA operators 
            'less' and 'greater' because it uses the NVIDIA CUB library.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Sort Pairs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Sort pairs* operations generate the same results as the sort operations
we have just described. Additionally, a second array of values is
reordered using the ordering of the first sorted array so 
**two arrays are passed to sort pairs methods.**

.. note:: For ``RAJA::sort_pairs`` algorithms, two arrays are passed. The 
          first array (*keys*) will be sorted according to the given 
          comparator operator. The elements in the second array (*values*) 
          will be reordered based on the final order of the first sorted array.

Here is a sequential unstable sort pairs operation that uses the 'less' 
operator:

.. literalinclude:: ../../../../exercises/sort_solution.cpp
   :start-after: _sort_pairs_seq_less_start
   :end-before: _sort_pairs_seq_less_end
   :language: C++

This generates the following sequence in the output array::

   (0,0) (0,1) (1,0) (1,1) (2,0) (2,1) (3,0) (3,1) (4,0) (4,1)
   (5,1) (5,0) (6,1) (6,0) (7,0) (7,1) (8,0) (8,1) (9,1) (9,0)

Note that some of the pairs with equivalent *keys* stayed in the same order
that they appeared in the unsorted arrays like ``(8,0) (8,1)``, while others are
reversed like ``(9,1) (9,0)``. This illustrates that relative ordering of
equal elements may not be preserved in an unstable sort.

Here is a sequential stable sort pairs that uses the greater operator:

.. literalinclude:: ../../../../exercises/sort_solution.cpp
   :start-after: _sort_stable_pairs_seq_greater_start
   :end-before: _sort_stable_pairs_seq_greater_end
   :language: C++

This generates the following sequence in the output array::

   (9,0) (9,1) (8,0) (8,1) (7,0) (7,1) (6,0) (6,1) (5,0) (5,1)
   (4,0) (4,1) (3,0) (3,1) (2,0) (2,1) (1,0) (1,1) (0,0) (0,1)

Note that all pairs with equivalent keys stayed in the same order that they
appeared in the unsorted input arrays.

As you may expect at this point, running an stable sort pairs
operation using OpenMP is accomplished by:

.. literalinclude:: ../../../../exercises/sort_solution.cpp
   :start-after: _sort_stable_pairs_omp_greater_start
   :end-before: _sort_stable_pairs_omp_greater_end
   :language: C++

This generates the following sequence in the output array (as we saw earlier)::

   (9,0) (9,1) (8,0) (8,1) (7,0) (7,1) (6,0) (6,1) (5,0) (5,1)
   (4,0) (4,1) (3,0) (3,1) (2,0) (2,1) (1,0) (1,1) (0,0) (0,1)

and the only difference is the execution policy template parameter.

Lastly, we show a parallel unstable sort pairs operation using CUDA:

.. literalinclude:: ../../../../exercises/sort_solution.cpp
   :start-after: _sort_pairs_cuda_greater_start
   :end-before: _sort_pairs_cuda_greater_end
   :language: C++

.. note:: RAJA sorts for the HIP back-end are similar to those for CUDA. 
          The only difference is that a HIP execution policy template 
          parameter type is used.

