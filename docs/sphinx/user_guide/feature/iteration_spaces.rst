.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _feat-index-label:

================================
Indices, Segments, and IndexSets
================================

Loop variables and their associated iteration spaces are fundamental to 
writing loop kernels in RAJA. RAJA provides some basic iteration space types 
that serve as flexible building blocks that can be used to form a variety 
of loop iteration patterns. These types can be used to define a particular 
order for loop iterates, aggregate and partition iterates, as well as other
configurations. In this section, we introduce RAJA index and iteration space 
concepts and types.

.. note:: All RAJA iteration space types described here are located in the 
          namespace ``RAJA``.

Please see the following tutorial sections for detailed examples that use
RAJA iteration space concepts:

 * :ref:`tut-indexset-label`
 * :ref:`tut-vertexsum-label`

.. _indices-label:

-------
Indices
-------

Just like traditional C and C++ for-loops, RAJA uses index variables to 
identify loop iterates. Any lambda expression that represents all or part of
a loop body passed to a ``RAJA::forall`` or ``RAJA::kernel`` method will 
take at least one loop index variable argument. RAJA iteration space types 
are templates that allow users to use any integral type for an
index variable. 

.. _segments-label:

-----------------------
Segments and IndexSets
-----------------------

A RAJA **Segment** represents a set of indices that one wants to 
execute as a unit for a kernel. RAJA provides the following Segment types:

   * ``RAJA::TypedRangeSegment`` represents a stride-1 range
   * ``RAJA::TypedRangeStrideSegment`` represents a (non-unit) stride range
   * ``RAJA::TypedListSegment`` represents an arbitrary set of indices

RAJA also provides a convenience helper ``RAJA::mask<Policy>(ctx, body)`` for
RAJA::launch kernels when one logical thread should execute setup work. It is mainly
useful for per-team initialization before ``ctx.teamSync()``, and it keeps the
intent explicit without pretending the work is a one-element segment.

A ``RAJA::TypedIndexSet`` is a container that can hold an arbitrary collection
of segments to compose iteration patterns in a single kernel invocation.

Segment and IndexSet types are used in ``RAJA::forall`` and other RAJA kernel
execution mechanisms to define the iteration space for a kernel.

.. note:: Iterating over the indices of all segments in a RAJA index set 
          requires a two-level execution policy, with two template parameters,
          as shown above. The first parameter specifies how to iterate over 
          the segments. The second parameter specifies how each segment will 
          execute. See :ref:`indexsetpolicy-label` for more information about 
          RAJA index set execution policies.

.. note:: It is the responsibility of the user to ensure that segments are
          defined properly when using RAJA index sets. For example, if the
          same index appears in multiple segments, the corresponding loop
          iteration will be run multiple times.

Please see :ref:`tut-indexset-label` for a detailed discussion of how to create
and use these segment types.

Segment Types and Iteration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is worth noting that RAJA segment types model **C++ iterable interfaces**.
In particular, each segment type defines three methods:

  * begin()
  * end()
  * size()

and two types:

  * iterator (essentially a *random access* iterator type)
  * value_type

Thus, any iterable type that defines these methods and types appropriately
can be used as a segment with RAJA kernel execution templates.

Python-like Range Helpers
^^^^^^^^^^^^^^^^^^^^^^^^^

RAJA also provides ``RAJA::range(...)`` helpers that construct the segment
type for common half-open iteration patterns. These helpers are intended to
mirror the shape of Python's ``range`` while returning RAJA segment objects
that can be passed directly to ``RAJA::forall`` and other execution
interfaces.

The supported forms are::

  RAJA::range(end)                 // [0, end)
  RAJA::range(begin, end)          // [begin, end)
  RAJA::range(begin, end, stride)  // [begin, end) with stride

The storage type can be explicitly specified if desired, and the arguments
are converted to that storage type. If an argument cannot be converted to the
chosen storage type, compilation fails. The explicit-type forms are::

  RAJA::range<IndexT>(end)                 // [0, end)
  RAJA::range<IndexT>(begin, end)          // [begin, end)
  RAJA::range<IndexT>(begin, end, stride)  // [begin, end) with stride

The return type depends on the arguments:

* ``RAJA::range(end)`` and ``RAJA::range(begin, end)`` return a
  ``RAJA::TypedRangeSegment``.
* ``RAJA::range(begin, end, stride)`` returns a
  ``RAJA::TypedRangeStrideSegment``.
* When one of the bounds is a RAJA strong index type, such as a type created
  with ``RAJA_INDEX_VALUE``, all bounds must use that same strong type. Mixed
  strong and plain integral bounds are rejected rather than narrowed. A
  strided range may still use a plain signed integral stride, such as
  ``RAJA::range(CellIndex {1}, CellIndex {N}, 2)``.
* Providing an explicit template argument, such as
  ``RAJA::range<MyIndex>(end)``, overrides the deduced storage type, but
  explicit storage must still be compatible with the argument types. For example,
  ``RAJA::range<int>(RangeStrongIndex(3), 17)`` and
  ``RAJA::range<RangeStrongIndex>(AnotherRangeStrongIndex(3), 17)`` are
  rejected, while ``RAJA::range<RangeStrongIndex>(RangeStrongIndex(3),
  RangeStrongIndex(17))`` is valid. For strided ranges, explicit strong
  storage can be used to convert plain integral values intentionally, such as
  ``RAJA::range<RangeStrongIndex>(3, 17, 1)``.

Index types created with ``RAJA_INDEX_VALUE`` wrap an integral value. The
examples below use ``*i`` to retrieve that wrapped value before indexing
ordinary C/C++ arrays; ``RAJA::stripIndexType(i)`` provides the same conversion
with a named helper.

For example::

  RAJA_INDEX_VALUE(CellIndex, "CellIndex");

  RAJA::forall<RAJA::seq_exec>(RAJA::range(N), [=](RAJA::Index_type i) {
    values[i] = i * i;
  });

  RAJA::forall<RAJA::seq_exec>(RAJA::range<CellIndex>(CellIndex {N}),
                               [=](CellIndex i) {
    typed_values[*i] = *i + 10;
  });

  RAJA::forall<RAJA::seq_exec>(RAJA::range(2, 6), [=](int i) {
    subrange_values[i] = i;
  });

  RAJA::forall<RAJA::seq_exec>(RAJA::range(CellIndex {1}, CellIndex {N},
                                            CellIndex {2}),
                               [=](CellIndex i) {
                                 strided_values[*i] = *i;
                               });

Strided ranges follow the same half-open interval convention as
``RAJA::TypedRangeStrideSegment``. The stride argument must have a signed
integral type. Positive strides move forward, and negative strides move
backward. For example, ``RAJA::range(N - 1, -1, -2)`` visits ``N - 1,
N - 3, ...`` down to the first value that remains greater than ``-1``. A zero
stride is invalid and causes RAJA to abort or throw, depending on the build
configuration.

The older ``RAJA::make_range`` and ``RAJA::make_strided_range`` helpers remain
available. Use ``RAJA::range(...)`` when the Python-like spelling improves
readability or when you want the one-argument ``[0, end)`` shorthand.

The complete example added in this branch is shown below:

.. literalinclude:: ../../../../examples/raja-ranges.cpp
   :language: c++
