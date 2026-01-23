.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _tut-indexset-label:

-------------------------------------------------
Iteration Spaces: Segments and IndexSets
-------------------------------------------------

This section contains an exercise file ``RAJA/exercises/segment-indexset-basics.cpp`` for you to work through if you wish to get some practice with RAJA. The 
file ``RAJA/exercises/segment-indexset-basics_solution.cpp`` contains complete 
working code for the examples discussed in this section. You can use the 
solution file to check your work and for guidance if you get stuck. To build
the exercises execute ``make segment-indexset-basics`` and ``make segment-indexset-basics_solution``
from the build directory.

Key RAJA features shown in this example are:

  * ``RAJA::forall`` loop execution template
  * ``RAJA::TypedRangeSegment``, ``RAJA::TypedRangeStrideSegment``, and 
    ``RAJA::TypedListSegment`` iteration space constructs
  * ``RAJA::TypedIndexSet`` container and associated execution policies

The concepts of iteration spaces and associated Loop variables are central to
writing kernels in RAJA. RAJA provides basic iteration space types
that serve as flexible building blocks that can be used to form a variety
of loop iteration patterns. These types can be used to define a particular
order for loop iterates, aggregate and partition iterates, as well as other
configurations. 

The examples in this section focus on how to use RAJA index sets and iteration 
space segments, such as index ranges and lists of indices. Lists of indices 
are important for algorithms that use indirection arrays for irregular array 
accesses. Combining different segment types, such as ranges and lists in an 
index set allows a user to launch different iteration patterns in a single loop 
execution construct (i.e., one kernel). This is something that is not 
supported by other programming models and abstractions and is unique to RAJA. 
Applying these concepts judiciously can help improve performance by allowing 
compilers to optimize for specific segment types (e.g., SIMD for range 
segments) while providing the flexibility of indirection arrays for general
indexing patterns.

Although the constructs described in the section are 
useful in numerical computations and parallel execution, the examples only 
contain print statements and sequential execution. The goal is to show you 
how to use RAJA iteration space constructs. 

^^^^^^^^^^^^^^^^^^^^^
RAJA Segments
^^^^^^^^^^^^^^^^^^^^^

A RAJA *Segment* represents a set of indices that one wants to execute as a 
unit for a kernel. RAJA provides the following Segment types:

   * ``RAJA::TypedRangeSegment`` represents a stride-1 range
   * ``RAJA::TypedRangeStrideSegment`` represents a (non-unit) stride range
   * ``RAJA::TypedListSegment`` represents an arbitrary set of indices

These segment types are used in ``RAJA::forall`` and other RAJA kernel
execution mechanisms to define the iteration space for a kernel.

After we briefly introduce these types, we will present several examples using
them.

TypedRangeSegment
^^^^^^^^^^^^^^^^^^^

A ``RAJA::TypedRangeSegment`` is the fundamental type for defining a
stride-1 (i.e., contiguous) range of indices. This is illustrated in the
figure below.

.. figure:: ../figures/RangeSegment.png

   A range segment defines a stride-1 index range [beg, end).

One creates a range segment object as follows::

   // A stride-1 index range [beg, end) using type int.
   RAJA::TypedRangeSegment<int> my_range(beg, end);

Any integral type can be given as the template parameter.

.. note:: When using a RAJA range segment, no loop iterations will be run when
          begin >= end.

TypedRangeStrideSegment
^^^^^^^^^^^^^^^^^^^^^^^^^

A ``RAJA::TypedRangeStrideSegment`` defines a range with a constant stride,
including negative stride values if needed. This is illustrated in the
figure below.

.. figure:: ../figures/RangeStrideSegment.png

   A range-stride segment defines an index range with arbitrary stride [beg, end, stride). In the figure the stride is 2.

One creates a range stride segment object as follows::

   // A stride-3 index range [beg, end) using type int.
   RAJA::TypedRangeStrideSegment<int> my_stride2_range(beg, end, 3);

   // A index range with -1 stride [0, N-1) using type int
   RAJA::TypedRangeStrideSegment<int> my_neg1_range( N-1, -1, -1);

Any integral type can be given as the template parameter.

When the negative-stride segment above is passed to a ``RAJA::forall`` method,
for example, the loop will run in reverse order with iterates::

  N-1  N-2  N-3 ... 1 0

.. note:: When using a RAJA strided range, no loop iterations will be run
          under the following conditions:

            * Stride > 0 and begin > end
            * Stride < 0 and begin < end
            * Stride == 0

TypedListSegment
^^^^^^^^^^^^^^^^^^

A ``RAJA::TypedListSegment`` is used to define an arbitrary set of
indices, akin to an indirection array. This is illustrated in the figure below.

.. figure:: ../figures/ListSegment.png

   A list segment defines an arbitrary collection of indices. Here, we have a list segment with 5 irregularly-spaced indices.

One creates a list segment object by passing a container of integral values to 
a list segment constructor. For example::

   // Create a vector holding some integer index values
   std::vector<int> idx = {0, 2, 3, 4, 7, 8, 9, 53};

   // Create list segment with these indices where the indices are
   // stored in the CUDA device memory space
   camp::resources::Resource cuda_res{camp::resources::Cuda()};
   RAJA::TypedListSegment<int> idx_list( idx[0], cuda_res );

   // Alternatively
   RAJA::TypedListSegment<int> idx_list( &idx[0], idx.size(),
                                         cuda_res );

When the list segment above is passed to a ``RAJA::forall`` method,
for example, the kernel will execute with iterates::

  0 2 3 4 7 8 9 53

Note that a ``RAJA::TypedListSegment`` constructor can take a pointer to
an array of indices and an array length. If the indices are
in a container, such as ``std::vector`` that provides ``begin()``, ``end()``,
and ``size()`` methods, the container can be passed to the constructor and
the length argument is not required.

.. note:: Currently, a camp resource object must be passed to a list segment 
          constructor to copy the indices in the indices into the proper 
          memory space for a kernel to execute (as shown above). In the future,
          this will change and the user will be responsible for providing
          the indices in the proper memory space.

^^^^^^^^^^^
IndexSets
^^^^^^^^^^^

A ``RAJA::TypedIndexSet`` is a container that can hold an arbitrary collection
of segment objects. The following figure shows an index set with two contiguous
ranges and an irregularly-spaced list of indices.

.. figure:: ../figures/IndexSet.png

   An index set with two range segments and one list segment.

We can create such an index set as follows::

   // Create an index set that can hold range and list segments with
   // int index value type
   RAJA::TypedIndexSet< RAJA::TypedRangeSegment<int>, 
                        RAJA::TypedListSegment<int> > iset;

   // Add two range segments and one list segment to the index set
   iset.push_back( RAJA::TypedRangeSegment<int>( ... ) );
   iset.push_back( RAJA::TypedListSegment<int>(...) );
   iset.push_back( RAJA::TypedRangeSegment<int>( ... ) );

A ``RAJA::TypedIndexSet`` object can be passed to a RAJA kernel execution 
method, such as ``RAJA::forall`` to execute all segments in the index set
with one method call. We will show this in detail in the examples below.

.. note:: It is the responsibility of the user to ensure that segments are
          defined properly when using RAJA index sets. For example, if the
          same index appears in multiple segments, the corresponding loop
          iteration will be run multiple times.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Segment and IndexSet Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The examples in this section illustrate how the  segment types that RAJA 
provides can be used to define kernel iteration spaces. We use the following
type aliases to make the code more compact:

.. literalinclude:: ../../../../exercises/segment-indexset-basics_solution.cpp
   :start-after: _raja_segment_type_start
   :end-before: _raja_segment_type_end
   :language: C++


Stride-1 Indexing
^^^^^^^^^^^^^^^^^^^

Consider a simple C-style kernel that prints a contiguous sequence of values:

.. literalinclude:: ../../../../exercises/segment-indexset-basics_solution.cpp
   :start-after: _cstyle_range1_start
   :end-before: _cstyle_range1_end
   :language: C++

When run, the kernel prints the following sequence, as expected::

  0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19

Three RAJA variants of the kernel using a ``RAJA::TypedRangeSegment``, a
``RAJA::TypedRangeStrideSegment``, and a ``RAJA::TypedListSegment`` are:

.. literalinclude:: ../../../../exercises/segment-indexset-basics_solution.cpp
   :start-after: _raja_range1_start
   :end-before: _raja_range1_end
   :language: C++

.. literalinclude:: ../../../../exercises/segment-indexset-basics_solution.cpp
   :start-after: _raja_striderange1_start
   :end-before: _raja_striderange1_end
   :language: C++

.. literalinclude:: ../../../../exercises/segment-indexset-basics_solution.cpp
   :start-after: _raja_list1_start
   :end-before: _raja_list1_end
   :language: C++

Each of these variants prints the same integer sequence shown above. 

One interesting thing to note is that with ``RAJA::TypedListSegment`` and
``RAJA::forall``, the actual iteration value is passed to the lambda loop body.
So the indirection array concept is not visible. In contrast, in C-style code, 
one has to manually retrieve the index value from the indirection array to 
achieve the desired result. For example:

.. literalinclude:: ../../../../exercises/segment-indexset-basics_solution.cpp
   :start-after: _cstyle_list1_start
   :end-before: _cstyle_list1_end
   :language: C++

Non-unit Stride Indexing 
^^^^^^^^^^^^^^^^^^^^^^^^^

Consider the following C-style kernel that prints the integer sequence 
discussed earlier in reverse order:

.. literalinclude:: ../../../../exercises/segment-indexset-basics_solution.cpp
   :start-after: _cstyle_negstriderange1_start
   :end-before: _cstyle_negstriderange1_end
   :language: C++

We can accomplish the same result using a ``RAJA::TypedRangeStrideSegment``:

.. literalinclude:: ../../../../exercises/segment-indexset-basics_solution.cpp
   :start-after: _raja_negstriderange1_start
   :end-before: _raja_negstriderange1_end
   :language: C++

Alternatively, we can use a ``RAJA::TypedListSegment``, where we reverse the
index array we used earlier to define the appropriate list segment:

.. literalinclude:: ../../../../exercises/segment-indexset-basics_solution.cpp
   :start-after: _raja_negstridelist1_start
   :end-before: _raja_negstridelist1_end
   :language: C++

The more common use of the ``RAJA::TypedRangeStrideSegment`` type is to run
constant strided loops with a positive non-unit stride. For example:

.. literalinclude:: ../../../../exercises/segment-indexset-basics_solution.cpp
   :start-after: _raja_range2_start
   :end-before: _raja_range2_end
   :language: C++

The C-style equivalent of this is:

.. literalinclude:: ../../../../exercises/segment-indexset-basics_solution.cpp
   :start-after: _cstyle_range2_start
   :end-before: _cstyle_range2_end
   :language: C++

IndexSets: Complex Iteration Spaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We noted earlier that ``RAJA::TypedIndexSet`` objects can be used to partition
iteration spaces into disjoint parts. Among other things, this can be useful to
expose parallelism in algorithms that would otherwise require significant 
code transformation to do so. Please see :ref:`tut-vertexsum-label` for 
discussion of an example that illustrates this.

Here is an example that uses two ``RAJA::TypedRangeSegment`` objects in an
index set to represent an iteration space broken into two disjoint 
contiguous intervals:

.. literalinclude:: ../../../../exercises/segment-indexset-basics_solution.cpp
   :start-after: _raja_indexset_2ranges_start
   :end-before: _raja_indexset_2ranges_end
   :language: C++

The integer sequence that is printed is::

  0  1  2  3  4  5  6  7  8  9  15  16  17  18  19

as we expect. 

The execution policy type when using a RAJA index set is a 
*two-level* policy. The first level specifies how to iterate over the segments
in the index set, such as sequentially or in parallel using OpenMP. The second
level is the execution policy used to execute each segment.

.. note:: Iterating over the indices of all segments in a RAJA index set
          requires a two-level execution policy, with two template parameters,
          as shown above. The first parameter specifies how to iterate over
          the segments. The second parameter specifies how the kernel will 
          execute each segment over each segment. 
          See :ref:`indexsetpolicy-label` for more information about
          RAJA index set execution policies.
  
It is worth noting that a C-style version of this kernel requires either 
an indirection array to run in one loop or two for-loops. For example:

.. literalinclude:: ../../../../exercises/segment-indexset-basics_solution.cpp
   :start-after: _cstyle_2ranges_start
   :end-before: _cstyle_2ranges_end
   :language: C++

Finally, we show an example that uses an index set holding two range segments
and one list segment to partition an iteration space into three parts:

.. literalinclude:: ../../../../exercises/segment-indexset-basics_solution.cpp
   :start-after: _raja_indexset_3segs_start
   :end-before: _raja_indexset_3segs_end
   :language: C++

The integer sequence that is printed is::

  0  1  2  3  4  5  6  7  10  11  14  20  22  24  25  26  27
