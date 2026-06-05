.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _cook-book-reductions-label:

=======================
Cooking with Reductions
=======================

Please see the following section for overview discussion about RAJA reductions:

 * :ref:`feat-reductions-label`.


----------------------------
Reductions with RAJA::forall
----------------------------

Here is the setup for a simple reduction example::

  const int N = 1000;

  int vec[N];

  for (int i = 0; i < N; ++i) {

    vec[i] = 1;

  }

Here a simple sum reduction is performed in a for loop::

  int vsum = 0;

  // Run a kernel using the reduction objects
  for (int i = 0; i < N; ++i) {

    vsum += vec[i];

  }

The results of these operations will yield the following values:

 * ``vsum == 1000``

RAJA uses policy types to specify how things are implemented.

The forall *execution policy* specifies how the loop is run by the ``RAJA::forall`` method. The following discussion includes examples of several other RAJA execution policies that could be applied.
For example ``RAJA::seq_exec`` runs a C-style for loop sequentially on a CPU. The
``RAJA::cuda_exec_with_reduce<256>`` runs the loop as a CUDA GPU kernel with
256 threads per block and other CUDA kernel launch parameters, like the
number of blocks, optimized for performance with reducers.::

  using exec_policy = RAJA::seq_exec;
  // using exec_policy = RAJA::omp_parallel_for_exec;
  // using exec_policy = RAJA::omp_target_parallel_for_exec<256>;
  // using exec_policy = RAJA::cuda_exec_with_reduce<256>;
  // using exec_policy = RAJA::hip_exec_with_reduce<256>;
  // using exec_policy = RAJA::sycl_exec<256>;

The reduction policy specifies how the reduction is done and must match the
execution policy. For example ``RAJA::seq_reduce`` does a sequential reduction
and can only be used with sequential execution policies. The
``RAJA::cuda_reduce_atomic`` policy uses atomics, if possible with the given
data type, and can only be used with cuda execution policies. Similarly for other RAJA execution back-ends, such as HIP and OpenMP. Here are example RAJA reduction policies whose names are indicative of which execution policies they work with::

  using reduce_policy = RAJA::seq_reduce;
  // using reduce_policy = RAJA::omp_reduce;
  // using reduce_policy = RAJA::omp_target_reduce;
  // using reduce_policy = RAJA::cuda_reduce_atomic;
  // using reduce_policy = RAJA::hip_reduce_atomic;
  // using reduce_policy = RAJA::sycl_reduce;


Here a simple sum reduction is performed using RAJA::

  RAJA::ReduceSum<reduce_policy, int> vsum(0);

  RAJA::forall<exec_policy>( RAJA::RangeSegment(0, N),
    [=](RAJA::Index_type i) {

    vsum += vec[i];

  });

The results of these operations will yield the following values:

 * ``vsum.get() == 1000``


Another option for the execution policy when using the cuda or hip backends are
the base policies which have a boolean parameter to choose between the general
use ``cuda/hip_exec`` policy and the ``cuda/hip_exec_with_reduce`` policy.::

  // static constexpr bool with_reduce = ...;
  // using exec_policy = RAJA::cuda_exec_base<with_reduce, 256>;
  // using exec_policy = RAJA::hip_exec_base<with_reduce, 256>;

Another option for the reduction policy when using the cuda or hip backends are
the base policies which have a boolean parameter to choose between the atomic
``cuda/hip_reduce_atomic`` policy and the non-atomic ``cuda/hip_reduce`` policy.::

  // static constexpr bool with_atomic = ...;
  // using reduce_policy = RAJA::cuda_reduce_base<with_atomic>;
  // using reduce_policy = RAJA::hip_reduce_base<with_atomic>;
