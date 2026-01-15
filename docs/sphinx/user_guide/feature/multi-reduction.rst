.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _feat-multi-reductions-label:

=========================
MultiReduction Operations
=========================

RAJA provides multi-reduction types that allow users to perform a runtime number
of reduction operations in kernels launched using ``RAJA::forall``, ``RAJA::kernel``,
and ``RAJA::launch`` methods in a portable, thread-safe manner. Users may
use as many multi-reduction objects in a loop kernel as they need. If a small
fixed number of reductions is required in a loop kernel then standard RAJA reduction objects can be
used. Available RAJA multi-reduction types are described in this section.

.. note:: All RAJA multi-reduction types are located in the namespace ``RAJA``.

Also

.. note:: * Each RAJA multi-reduction type is templated on a **multi-reduction policy**
            and a **reduction value type** for the multi-reduction variable. The
            **multi-reduction policy type must be compatible with the execution
            policy used by the kernel in which it is used.** For example, in
            a CUDA kernel, a CUDA multi-reduction policy must be used.
          * Each RAJA multi-reduction type accepts an **initial reduction value or
            values** at construction (see below).
          * Each RAJA multi-reduction type has a 'get' method to access reduced
            values after kernel execution completes.

Please see the following sections for a description of reducers:

 * :ref:`feat-reductions-label`.

Please see the following cook book sections for guidance on policy usage:

 * :ref:`cook-book-multi-reductions-label`.


--------------------
MultiReduction Types
--------------------

RAJA supports three common multi-reduction types:

* ``MultiReduceSum< multi_reduce_policy, data_type >`` - Sum of values.

* ``MultiReduceMin< multi_reduce_policy, data_type >`` - Min value.

* ``MultiReduceMax< multi_reduce_policy, data_type >`` - Max value.

and two less common bitwise multi-reduction types:

* ``MultiReduceBitAnd< multi_reduce_policy, data_type >`` - Bitwise 'and' of values (i.e., ``a & b``).

* ``MultiReduceBitOr< multi_reduce_policy, data_type >`` - Bitwise 'or' of values (i.e., ``a | b``).

.. note:: ``RAJA::MultiReduceBitAnd`` and ``RAJA::MultiReduceBitOr`` reduction types are designed to work on integral data types because **in C++, at the language level, there is no such thing as a bitwise operator on floating-point numbers.**

-----------------------
MultiReduction Examples
-----------------------

Next, we provide a few examples to illustrate basic usage of RAJA multi-reduction
types.

Here is a simple RAJA multi-reduction example that shows how to use a sum
multi-reduction type::

  const int N = 1000;
  const int B = 10;

  //
  // Initialize an array of length N with all ones, and another array to
  // integers between 0 and B-1
  //
  int vec[N];
  int bins[N];
  for (int i = 0; i < N; ++i) {
    vec[i] = 1;
    bins[i] = i % B;
  }

  // Create a sum multi-reduction object with a size of B, and initial
  // values of zero
  RAJA::MultiReduceSum< RAJA::omp_multi_reduce, int > vsum(B, 0);

  // Run a kernel using the multi-reduction object
  RAJA::forall<RAJA::omp_parallel_for_exec>( RAJA::RangeSegment(0, N),
    [=](RAJA::Index_type i) {

    vsum[bins[i]] += vec[i];

  });

  // After kernel is run, extract the reduced values
  int my_vsums[B];
  for (int bin = 0; bin < B; ++bin) {
    my_vsums[bin] = vsum[bin].get();
  }

The results of these operations will yield the following values:

 * my_vsums[0] == 100
 * my_vsums[1] == 100
 * my_vsums[2] == 100
 * my_vsums[3] == 100
 * my_vsums[4] == 100
 * my_vsums[5] == 100
 * my_vsums[6] == 100
 * my_vsums[7] == 100
 * my_vsums[8] == 100
 * my_vsums[9] == 100


Here is the same example but using values stored in a container::

  const int N = 1000;
  const int B = 10;

  //
  // Initialize an array of length N with all ones, and another array to
  // integers between 0 and B-1
  //
  int vec[N];
  int bins[N];
  for (int i = 0; i < N; ++i) {
    vec[i] = 1;
    bins[i] = i % B;
  }

  // Create a vector with a size of B, and initial values of zero
  std::vector<int> my_vsums(B, 0);

  // Create a multi-reducer initalized with size and values from my_vsums
  RAJA::MultiReduceSum< RAJA::omp_multi_reduce, int > vsum(my_vsums);

  // Run a kernel using the multi-reduction object
  RAJA::forall<RAJA::omp_parallel_for_exec>( RAJA::RangeSegment(0, N),
    [=](RAJA::Index_type i) {

    vsum[bins[i]] += vec[i];

  });

  // After kernel is run, extract the reduced values back into my_vsums
  vsum.get_all(my_vsums);

The results of these operations will yield the following values:

 * my_vsums[0] == 100
 * my_vsums[1] == 100
 * my_vsums[2] == 100
 * my_vsums[3] == 100
 * my_vsums[4] == 100
 * my_vsums[5] == 100
 * my_vsums[6] == 100
 * my_vsums[7] == 100
 * my_vsums[8] == 100
 * my_vsums[9] == 100





Here is an example of a bitwise-or multi-reduction::

  const int N = 128;
  const int B = 8;

  //
  // Initialize an array of length N to integers between 0 and B-1
  //
  int bins[N];
  for (int i = 0; i < N; ++i) {
    bins[i] = i % B;
  }

  // Create a bitwise-or multi-reduction object with initial value of '0'
  RAJA::MultiReduceBitOr< RAJA::omp_multi_reduce, int > vor(B, 0);

  // Run a kernel using the multi-reduction object
  RAJA::forall<RAJA::omp_parallel_for_exec>( RAJA::RangeSegment(0, N),
    [=](RAJA::Index_type i) {

    vor[bins[i]] |= i;

  });

  // After kernel is run, extract the reduced values
  int my_vors[B];
  for (int bin = 0; bin < B; ++bin) {
    my_vors[bin] = vor[bin].get();
  }

The results of these operations will yield the following values:

 * my_vors[0] == 120 == 0b1111000
 * my_vors[1] == 121 == 0b1111001
 * my_vors[2] == 122 == 0b1111010
 * my_vors[3] == 123 == 0b1111011
 * my_vors[4] == 124 == 0b1111100
 * my_vors[5] == 125 == 0b1111101
 * my_vors[6] == 126 == 0b1111110
 * my_vors[7] == 127 == 0b1111111

The results of the multi-reduction start at 120 and increase to 127. In binary
representation (i.e., bits), :math:`120 = 0b1111000` and :math:`127 = 0b1111111`.
The bins were picked in such a way that all the integers in a bin had the same
remainder modulo 8 so their last 3 binary digits were all the same while their
upper binary digits varied. Because bitwise-or keeps all the set bits, the upper
bits are all set because at least one integer in that bin set them. The last
3 bits were the same in all the integers so the last 3 bits are the same as the
remainder modulo 8 of the bin number.

-----------------------
MultiReduction Policies
-----------------------

For more information about available RAJA multi-reduction policies and guidance
on which to use with RAJA execution policies, please see
:ref:`multi-reducepolicy-label`.
