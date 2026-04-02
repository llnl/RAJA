.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _feat-sort-label:

================
Sort Operations
================

RAJA provides portable parallel sort operations, which are described in this 
section.

A few important notes:

.. note:: * All RAJA sort operations are in the namespace ``RAJA``.
          * Each RAJA sort operation is a template on an *execution policy*
            parameter. The same policy types used for ``RAJA::forall`` methods
            may be used for RAJA sorts. Please see :ref:`feat-policies-label` 
            for more information.
          * RAJA sort operations accept an optional *comparator* argument so
            users can perform different types of sort operations. If
            no operator is given, the default is a *less than* operation and
            the result is a sequence sorted in **non-decreasing** order.

Also:

.. note:: For sorts using the CUDA or HIP back-end, RAJA implementation uses
          the NVIDIA CUB library or AMD rocPRIM library, respectively.
          Typically, the CMake variable ``CUB_DIR`` or ``ROCPRIM_DIR`` will
          be automatically set to the location of the CUB or rocPRIM library
          for the CUDA or rocPRIM installation specified when either back-end
          is enabled. More details for configuring the CUB or rocPRIM library
          for a RAJA build can be found :ref:`getting_started_depend-label`.

Please see the following tutorial sections for detailed examples that use
RAJA scan operations:

 * :ref:`tut-sort-label`

-----------------
Sort Operations
-----------------

In general, a sort operation takes a sequence of numbers 'x' and a binary
comparison operator 'op' to form a strict weak ordering of elements in
input sequence 'x' and produce a sequence of numbers 'y' as output. The
output sequence is a permutation of the input sequence where each pair of
elements 'a' and 'b', where 'a' is before 'b' in the output sequence,
satisfies '!(b op a)'. Sorts are stable if they always preserve the order of
equivalent elements, where equivalent means '!(a op b) && !(b op a)' is true.

A **stable sort** takes an input sequence 'x' where a\ :sub:`i` appears
before a\ :sub:`j` if i < j when a\ :sub:`i` and a\ :sub:`j` are equivalent 
for any i != j.

   x = { a\ :sub:`0`\, b\ :sub:`0`\, a\ :sub:`1`\, ... }

and calculates the stably sorted output sequence 'y' that preserves the
order of equivalent elements. That is, the sorted sequence where element
a\ :sub:`i` appears before the equivalent element a\ :sub:`j` if i < j:

   y = { a\ :sub:`0`\, a\ :sub:`1`\, b\ :sub:`0`\, ... }

An **unstable sort** may not preserve the order of equivalent elements and
may produce either of the following output sequences:

   y = { a\ :sub:`0`\, a\ :sub:`1`\, b\ :sub:`0`\, ... }

   or

   y = { a\ :sub:`1`\, a\ :sub:`0`\, b\ :sub:`0`\, ... }

---------------------
RAJA Unstable Sorts
---------------------

RAJA unstable sort operations look like the following:

 * ``RAJA::sort< exec_policy >(container)``
 * ``RAJA::sort< exec_policy >(container, comparator)``

For example, sorting an integer array with this sequence of values::

   6 7 2 1 0 9 4 8 5 3 4 9 6 3 7 0 1 8 2 5

with a sequential unstable sort operation:

.. literalinclude:: ../../../../exercises/sort_solution.cpp
   :start-after: _sort_seq_start
   :end-before: _sort_seq_end
   :language: C++

produces the ``out`` array with this sequence of values::

   0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9

Note that the syntax is essentially the same as :ref:`feat-scan-label`.
Here, ``container`` is a random access range of elements. ``container`` provides
access to the input sequence and contains the output sequence at the end of
sort. The sort operation listed above will be a *non-decreasing* sort
since there is no comparator argument given; i.e., the sequences will be
reordered *in-place* using the default RAJA less-than comparator.

Equivalently, the ``RAJA::operators::less`` comparator operator could be 
passed as the second argument to the sort routine to produce the same result:

.. literalinclude:: ../../../../exercises/sort_solution.cpp
   :start-after: _sort_seq_less_start
   :end-before: _sort_seq_less_end
   :language: C++

Note that container arguments can be generated from iterators using 
``RAJA::make_span(out, N)``, where we pass the base pointer for the array 
and its length.

RAJA also provides sort operations that operate on key-value pairs stored
separately:

 * ``RAJA::sort_pairs< exec_policy >(keys_container, vals_container)``
 * ``RAJA::sort_pairs< exec_policy >(keys_container, vals_container, comparator)``

``RAJA::sort_pairs`` methods generate the same output sequence of keys in
``keys_container`` as ``RAJA::sort`` does in ``container`` and reorders the
sequence of values in ``vals_container`` by permuting the sequence of values in
the same manner as the sequence of keys; i.e. the sequence of pairs is sorted
based on comparing their keys. Detailed examples are provided in 
:ref:`tut-sort-label`.

.. note:: The comparator used in ``RAJA::sort_pairs`` only compares keys.

---------------------
RAJA Stable Sorts
---------------------

RAJA stable sort operations are used essentially the same as unstable sorts:

 * ``RAJA::stable_sort< exec_policy >(container)``
 * ``RAJA::stable_sort< exec_policy >(container, comparator)``

RAJA also provides stable sort pairs that operate on key-value pairs stored
separately:

 * ``RAJA::stable_sort_pairs< exec_policy >(keys_container, vals_container)``
 * ``RAJA::stable_sort_pairs< exec_policy >(keys_container, vals_container, comparator)``

.. _feat-sortops-label:

--------------------------
RAJA Comparison Operators
--------------------------

RAJA provides two operators that can be used to produce different ordered sorts:

  * ``RAJA::operators::less<T>``
  * ``RAJA::operators::greater<T>``

.. note:: All RAJA comparison operators are in the namespace ``RAJA::operators``.

