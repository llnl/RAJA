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
