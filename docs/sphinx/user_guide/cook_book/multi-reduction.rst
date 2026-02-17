.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _cook-book-multi-reductions-label:

============================
Cooking with MultiReductions
============================

Please see the following section for overview discussion about RAJA multi-reductions:

 * :ref:`feat-multi-reductions-label`.


---------------------------------
MultiReductions with RAJA::forall
---------------------------------

Here is the setup for a simple multi-reduction example::

  const int N = 1000;
  const int num_bins = 10;

  int vec[N];
  int bins[N];

  for (int i = 0; i < N; ++i) {

    vec[i] = 1;
    bins[i] = i % num_bins;

  }

Here a simple sum multi-reduction performed in a C-style for-loop::

  int vsum[num_bins] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Run a kernel using the multi-reduction objects
  for (int i = 0; i < N; ++i) {

    vsum[bins[i]] += vec[i];

  }

The results of these operations will yield the following values:

 * ``vsum[0] == 100``
 * ``vsum[1] == 100``
 * ``vsum[2] == 100``
 * ``vsum[3] == 100``
 * ``vsum[4] == 100``
 * ``vsum[5] == 100``
 * ``vsum[6] == 100``
 * ``vsum[7] == 100``
 * ``vsum[8] == 100``
 * ``vsum[9] == 100``

RAJA uses policy types to specify how things are implemented.

The forall *execution policy* specifies how the loop is run by the ``RAJA::forall`` method. The following discussion includes examples of several other RAJA execution policies that could be applied.
For example ``RAJA::seq_exec`` runs a C-style for-loop sequentially on a CPU. The
``RAJA::cuda_exec_with_reduce<256>`` runs the operation as a CUDA GPU kernel with
256 threads per block and other CUDA kernel launch parameters, like the
number of blocks, optimized for performance with multi_reducers.::

  using exec_policy = RAJA::seq_exec;
  // using exec_policy = RAJA::omp_parallel_for_exec;
  // using exec_policy = RAJA::cuda_exec_with_reduce<256>;
  // using exec_policy = RAJA::hip_exec_with_reduce<256>;

The multi-reduction policy specifies how the multi-reduction is done and must be compatible with the
execution policy. For example, ``RAJA::seq_multi_reduce`` does a sequential multi-reduction
and can only be used with sequential execution policies. The
``RAJA::cuda_multi_reduce_atomic`` policy uses atomics and can only be used with
cuda execution policies. Similarly for other RAJA execution back-ends, such as
HIP and OpenMP. Here are example RAJA multi-reduction policies whose names are
indicative of which execution policies they work with::

  using multi_reduce_policy = RAJA::seq_multi_reduce;
  // using multi_reduce_policy = RAJA::omp_multi_reduce;
  // using multi_reduce_policy = RAJA::cuda_multi_reduce_atomic;
  // using multi_reduce_policy = RAJA::hip_multi_reduce_atomic;

Here a simple sum multi-reduction is performed using RAJA::

  RAJA::MultiReduceSum<multi_reduce_policy, int> vsum(num_bins, 0);

  RAJA::forall<exec_policy>( RAJA::RangeSegment(0, N),
    [=](RAJA::Index_type i) {

    vsum[bins[i]] += vec[i];

  });

The results of these operations will yield the following values:

 * ``vsum[0].get() == 100``
 * ``vsum[1].get() == 100``
 * ``vsum[2].get() == 100``
 * ``vsum[3].get() == 100``
 * ``vsum[4].get() == 100``
 * ``vsum[5].get() == 100``
 * ``vsum[6].get() == 100``
 * ``vsum[7].get() == 100``
 * ``vsum[8].get() == 100``
 * ``vsum[9].get() == 100``

Another option for the execution policy when using the CUDA or HIP backends are
the base policies which have a boolean parameter to choose between the general
use ``cuda/hip_exec`` policy and the ``cuda/hip_exec_with_reduce`` policy.::

  // static constexpr bool with_reduce = ...;
  // using exec_policy = RAJA::cuda_exec_base<with_reduce, 256>;
  // using exec_policy = RAJA::hip_exec_base<with_reduce, 256>;


---------------------------
Rarely Used MultiReductions
---------------------------

Multi-reductions consume resources even if they are not used in a
loop kernel. If a multi-reducer is conditionally used to set an error flag, for example, even
if the multi-reduction is not used at runtime in the loop kernel, then the setup
and finalization for the multi-reduction is still done and any resources are
still allocated and deallocated. To minimize these overheads, some backends have
special policies that minimize the amount of work the multi-reducer does in the
case that it is not used at runtime even if it is compiled into a loop kernel.
Here are example RAJA multi-reduction policies that have minimal overhead::

  using rarely_used_multi_reduce_policy = RAJA::seq_multi_reduce;
  // using rarely_used_multi_reduce_policy = RAJA::omp_multi_reduce;
  // using rarely_used_multi_reduce_policy = RAJA::cuda_multi_reduce_atomic_low_performance_low_overhead;
  // using rarely_used_multi_reduce_policy = RAJA::hip_multi_reduce_atomic_low_performance_low_overhead;

Here is a simple rarely used bitwise-or multi-reduction performed using RAJA::

  RAJA::MultiReduceBitOr<rarely_used_multi_reduce_policy, int> vor(num_bins, 0);

  RAJA::forall<exec_policy>( RAJA::RangeSegment(0, N),
    [=](RAJA::Index_type i) {

    if (vec[i] < 0) {
      vor[0] |= 1;
    }

  });

The results of these operations will yield the following value if the condition
is never met:

 * ``vsum[0].get() == 0``

or yield the following value if the condition is ever met:

 * ``vsum[0].get() == 1``
