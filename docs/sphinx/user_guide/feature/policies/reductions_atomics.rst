.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _feat-policies-reductions-atomics-reference-label:

===============================================
Reduction, Multi-Reduction, and Atomic Policies
===============================================

This page contains policy reference material for RAJA reductions,
multi-reductions, and atomics.

.. _reducepolicy-label:

-------------------------
Reduction Policies
-------------------------

Each RAJA reduction object must be defined with a 'reduction policy'
type. Reduction policy types are distinct from loop execution policy types.
It is important to note the following constraint about RAJA reduction usage:

.. important:: To guarantee correctness, a **reduction policy must support
          the loop execution policy** used. For example, a CUDA
          reduction policy must be used when the execution policy is a
          CUDA policy. However an OpenMP reduction policy or a CUDA reduction
          policy may be used when the execution policy is an OpenMP policy,
          and so on.

.. note:: It is undefined behavior to use a reducer object with a loop execution
          policy that does not match the ``RAJA::Policy`` enum argument used to
          most recently setup the reducer object. ``RAJA::Policy::undefined``
          may be used with any of the loop policies supported by the reduction
          policy. For example, if a reducer object with a CUDA reduction policy
          is setup with ``RAJA::Policy::cuda``, but used in a sequential loop,
          then that is undefined behavior. Using either
          ``RAJA::Policy::undefined`` or ``RAJA::Policy::sequential`` is
          correct.

.. important:: RAJA reductions used with SIMD execution policies are not
          guaranteed to generate correct results. So they should not be used
          for kernels containing reductions.

The following table summarizes RAJA reduction policy types:

================================================= ===================== ==========================================
Reduction Policy                                  Loop Policies         Brief description
                                                  Supported
================================================= ===================== ==========================================
seq_reduce                                        seq_exec              Non-parallel (sequential) reduction.
omp_reduce                                        any OpenMP policy,    OpenMP parallel reduction.
                                                  seq_exec
omp_reduce_ordered                                any OpenMP policy,    OpenMP parallel reduction with result
                                                  seq_exec              guaranteed to be reproducible.
omp_target_reduce                                 any OpenMP Target     OpenMP parallel target offload reduction.
                                                  policy,
                                                  seq_exec
cuda/hip_reduce                                   any CUDA/HIP policy,  Parallel reduction in a CUDA/HIP kernel
                                                  any OpenMP policy,    (device synchronization will occur when
                                                  seq_exec              reduction value is finalized).
cuda/hip_reduce_atomic                            any CUDA/HIP policy,  Same as above, but reduction may use
                                                  any OpenMP policy,    atomic operations leading to run to run
                                                  seq_exec              variability in the results.
cuda/hip_reduce_base<with_atomic>                 any CUDA/HIP policy,  Choose between cuda/hip_reduce and
                                                  any OpenMP policy,    cuda/hip_reduce_atomic policies based on
                                                  seq_exec              the with_atomic boolean.
cuda/hip_reduce_device_fence                      any CUDA/HIP policy,  Same as above, and reduction uses normal
                                                  any OpenMP policy,    memory accesses that are not visible
                                                  seq_exec              across the whole device and device scope
                                                                        fences to ensure visibility and ordering.
                                                                        This works on all architectures but
                                                                        incurs higher overheads on some
                                                                        architectures.
cuda/hip_reduce_block_fence                       any CUDA/HIP policy,  Same as above, and reduction uses special
                                                  any OpenMP policy,    memory accesses to a level of cache
                                                  seq_exec              visible to the whole device and block
                                                                        scope fences to ensure ordering. This
                                                                        improves performance on some architectures.
cuda/hip_reduce_atomic_host_init_device_fence     any CUDA/HIP policy,  Same as above with device fence, but
                                                  any OpenMP policy,    initializes the memory used for atomics
                                                  seq_exec              on the host. This works well on recent
                                                                        architectures and incurs lower overheads.
cuda/hip_reduce_atomic_host_init_block_fence      any CUDA/HIP policy,  Same as above with block fence, but
                                                  any OpenMP policy,    initializes the memory used for atomics
                                                  seq_exec              on the host. This works well on recent
                                                                        architectures and incurs lower overheads.
cuda/hip_reduce_atomic_device_init_device_fence   any CUDA/HIP policy,  Same as above with device fence, but
                                                  any OpenMP policy,    initializes the memory used for atomics
                                                  seq_exec              on the device. This works on all
                                                                        architectures
                                                                        but incurs higher overheads.
cuda/hip_reduce_atomic_device_init_block_fence    any CUDA/HIP policy,  Same as above with block fence, but
                                                  any OpenMP policy,    initializes the memory used for atomics
                                                  seq_exec              on the device. This works on all
                                                                        architectures
                                                                        but incurs higher overheads.
sycl_reduce                                       any SYCL policy,      Reduction in a SYCL kernel (device
                                                  seq_exec              synchronization will occur when the
                                                                        reduction value is finalized).
================================================= ===================== ==========================================

.. _multi-reducepolicy-label:

-------------------------
MultiReduction Policies
-------------------------

Each RAJA multi-reduction object must be defined with a 'multi-reduction policy'
type. Multi-reduction policy types are distinct from loop execution policy types.
It is important to note the following constraints about RAJA multi-reduction usage:

.. important:: To guarantee correctness, a **multi-reduction policy must support
               the loop execution policy** used. For example, a CUDA
               multi-reduction policy must be used when the execution policy is a
               CUDA policy. However an OpenMP multi-reduction policy or a CUDA
               multi-reduction policy may be used when the execution policy is an
               OpenMP policy, and so on.

.. note:: It is undefined behavior to use a multi-reducer object with a loop
          execution policy that does not match the ``RAJA::Policy`` enum
          argument used to most recently setup the multi-reducer object.
          ``RAJA::Policy::undefined`` may be used with any of the loop policies
          supported by the multi-reduction policy. For example, if a
          multi-reducer object with a CUDA multi-reduction policy is setup with
          ``RAJA::Policy::cuda``, but used in a sequential loop, then that is
          undefined behavior. Using either ``RAJA::Policy::undefined`` or
          ``RAJA::Policy::sequential`` is correct.

.. important:: RAJA multi-reductions used with SIMD execution policies are not
          guaranteed to generate correct results. So they should not be used
          for kernels containing multi-reductions.

The following table summarizes RAJA multi-reduction policy types:

============================================================= ============= ==========================================
MultiReduction Policy                                         Loop Policies Brief description
                                                              Supported
============================================================= ============= ==========================================
seq_multi_reduce                                              seq_exec      Non-parallel (sequential) multi-reduction.
omp_multi_reduce                                              any OpenMP    OpenMP parallel multi-reduction.
                                                              policy,
                                                              seq_exec
omp_multi_reduce_ordered                                      any OpenMP    OpenMP parallel multi-reduction with result
                                                              policy,       guaranteed to be reproducible.
                                                              seq_exec
cuda/hip_multi_reduce_atomic                                  any CUDA/HIP  Parallel multi-reduction in a CUDA/HIP kernel.
                                                              policy,       Multi-reduction may use atomic operations
                                                              any OpenMP
                                                              policy,
                                                              seq_exec
                                                                            leading to run to run variability in the
                                                                            results.
                                                                            (device synchronization will occur when
                                                                            reduction value is finalized)
cuda/hip_multi_reduce_atomic_low_performance_low_overhead     any CUDA/HIP  Same as above, but multi-reduction uses
                                                              policy,       a low overhead algorithm with a minimal
                                                              any OpenMP
                                                              policy,
                                                              seq_exec
                                                                            set of resources. This minimally affects
                                                                            the performance of loops containing the
                                                                            multi-reducer though it may cause the
                                                                            multi-reducer itself to perform poorly if
                                                                            it is used.
cuda/hip_multi_reduce_atomic_block_then_atomic_grid_host_init any CUDA/HIP  The multi-reduction uses atomics into shared
                                                              policy,       memory and global memory. Atomics into
                                                              any OpenMP
                                                              policy,
                                                              seq_exec
                                                                            shared memory are used each time a value
                                                                            is combined into the multi-reducer and at
                                                                            the end of the life of the block the shared
                                                                            values are combined into global memory with
                                                                            atomics. If there is not enough shared memory
                                                                            available this will fall back to using atomics into
                                                                            global memory only, which may have a
                                                                            performance penalty.
                                                                            The memory for global atomics is
                                                                            initialized on the host.
cuda/hip_multi_reduce_atomic_global_host_init                 any CUDA/HIP  The multi-reduction uses atomics into global
                                                              policy,       global memory only. Atomics into
                                                              any OpenMP
                                                              policy,
                                                              seq_exec
                                                                            global memory are used each time a value
                                                                            is combined into the multi-reducer.
                                                                            The memory for global atomics is
                                                                            initialized on the host.
cuda/hip_multi_reduce_atomic_global_no_replication_host_init  any CUDA/HIP  Same as above, but uses minimal memory
                                                              policy,
                                                              any OpenMP
                                                              policy,
                                                              seq_exec
                                                                            by not replicating global atomics.

============================================================= ============= ==========================================

.. _atomicpolicy-label:

-------------------------
Atomic Policies
-------------------------

Each RAJA atomic operation must be defined with an 'atomic policy'
type. Atomic policy types are distinct from loop execution policy types.

.. important:: An atomic policy type must be consistent with the loop execution
               policy for the kernel in which the atomic operation is used.

The following table summarizes RAJA atomic policies and usage.

============================= ============= ========================================
Atomic Policy                 Loop Policies Brief description
                              to Use With
============================= ============= ========================================
seq_atomic                    seq_exec,     Atomic operation performed in a
                                            non-parallel (sequential) kernel.
omp_atomic                    any OpenMP    Atomic operation in OpenMP
                              policy        multithreading or target kernel;
                                            i.e., apply ``omp atomic`` pragma.
cuda/hip/sycl_atomic          any           Atomic operation performed in a
                              CUDA/HIP/SYCL CUDA/HIP/SYCL kernel.
                              policy

cuda/hip_atomic_explicit      any CUDA/HIP  Atomic operation performed in a CUDA/HIP
                              policy        kernel that may also be used in a host
                                            execution context. The atomic policy
                                            takes a host atomic policy template
                                            argument. See additional explanation
                                            and example below.
builtin_atomic                seq_exec,     Compiler *builtin* atomic operation.
                              any OpenMP
                              policy
auto_atomic                   seq_exec,     Atomic operation *compatible* with
                              any OpenMP    loop execution policy. See example
                              policy,       below. Cannot be used inside CUDA or
                              any           HIP explicit atomic policies.
                              CUDA/HIP/SYCL
                              policy
============================= ============= ========================================

.. note:: The ``cuda_atomic_explicit`` and ``hip_atomic_explicit`` policies
          take a host atomic policy template parameter. They are intended to
          be used with kernels that are host-device decorated to be used in
          either a host or device execution context, possibly decided at run time.

Here is an example illustrating use of the ``cuda_atomic_explicit`` policy with an
OpenMP host policy::

  auto kernel = [=] RAJA_HOST_DEVICE (RAJA::Index_type i) {
    RAJA::atomicAdd< RAJA::cuda_atomic_explicit<omp_atomic> >(&sum, 1);
  };

  RAJA::forall< RAJA::cuda_exec<BLOCK_SIZE> >(RAJA::TypedRangeSegment<int> seg(0, N), kernel);

  RAJA::forall< RAJA::omp_parallel_for_exec >(RAJA::TypedRangeSegment<int> seg(0, N), kernel);

In this case, the atomic operation knows when it is compiled for the device
in a CUDA kernel context and the CUDA atomic operation is applied. Similarly
when it is compiled for the host in an OpenMP kernel the omp_atomic policy is
used and the OpenMP version of the atomic operation is applied.

Here is an example illustrating use of the ``auto_atomic`` policy::

  RAJA::forall< RAJA::cuda_exec<BLOCK_SIZE> >(RAJA::TypedRangeSegment<int> seg(0, N),
    [=] RAJA_DEVICE (RAJA::Index_type i) {

    RAJA::atomicAdd< RAJA::auto_atomic >(&sum, 1);

  });

In this case, the atomic operation knows that it is used in a CUDA kernel
context and the CUDA atomic operation is applied. Similarly, if an OpenMP
execution policy was used, the OpenMP version of the atomic operation would
be used.

.. note:: The ``builtin_atomic`` policy may be preferable to the
          ``omp_atomic`` policy in terms of performance.
