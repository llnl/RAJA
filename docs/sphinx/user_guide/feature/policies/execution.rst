.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _feat-policies-execution-reference-label:

==========================
Execution Policy Reference
==========================

This page contains execution policy reference material for loops, launches,
index sets, GPU back-ends, OpenMP target offload, and device aliases.

-----------------------------------------------------
RAJA Loop/Kernel Execution Policies
-----------------------------------------------------

The following tables summarize RAJA policies for executing kernels. Specifically,
they provide a brief explanation of each policy and state which policies work
with which RAJA abstractions. Please see notes below policy descriptions for
additional usage details and caveats.

.. _feat-policies-cpu-label:

Sequential CPU Policies
~~~~~~~~~~~~~~~~~~~~~~~~

For the sequential CPU back-end, RAJA provides policies that allow developers
to have some control over the optimizations that compilers are allowed to
apply.

 ====================================== ============= ==========================
 Sequential/SIMD Execution Policies     Works with    Brief description
 ====================================== ============= ==========================
 seq_launch_t                           launch        Creates a sequential
                                                      execution space.
 seq_exec                               forall,       Sequential execution,
                                        kernel (For), where the compiler is
                                        scan,         allowed to apply any
                                        sort          optimizations
                                                      that its heuristics deem
                                                      beneficial; i.e., no loop
                                                      decorations (pragmas or
                                                      intrinsics) are used in
                                                      the RAJA implementation.
 simd_exec                              forall,       Try to force generation of
                                        kernel (For), SIMD instructions via
                                        scan          compiler hints in RAJA's
                                                      internal implementation.
 ====================================== ============= ==========================


.. _feat-policies-openmp-cpu-label:

OpenMP CPU Policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the OpenMP CPU multithreading back-end, RAJA has policies that create
an OpenMP parallel region and execute a kernel within it. We refer to these
as **full policies**. They are provided to support common OpenMP use cases.

.. note:: To control the number of threads used by OpenMP, users may set
          the value of the environment variable 'OMP_NUM_THREADS' (which is
          fixed for duration of run), or call the OpenMP routine
          'omp_set_num_threads(nthreads)' in their applications, which allows
          changing the number of threads at run time.

The full policies are described in the following table. Partial policies
are described in other tables below.

 ========================================= ============== ======================
 OpenMP CPU Full Policies                  Works with     Brief description
 ========================================= ============== ======================
 omp_parallel_for_exec                     forall,        Same as applying the
                                           kernel (For),  OpenMP pragma
                                           launch (loop), 'omp parallel for
                                           scan,          schedule(auto)'
                                           sort
 omp_parallel_for_static_exec<ChunkSize>   forall,        Same as applying
                                           kernel (For)   'omp parallel for
                                                          schedule(static,
                                                          ChunkSize)'
 omp_parallel_for_dynamic_exec<ChunkSize>  forall,        Same as applying
                                           kernel (For)   'omp parallel for
                                                          schedule(dynamic,
                                                          ChunkSize)'
 omp_parallel_for_guided_exec<ChunkSize>   forall,        Same as applying
                                           kernel (For)   'omp parallel for
                                                          schedule(guided,
                                                          ChunkSize)'
 omp_parallel_for_runtime_exec             forall,        Same as applying
                                           kernel (For)   'omp parallel for
                                                          schedule(runtime)'
 ========================================= ============== ======================

RAJA also provides other OpenMP policies, which we refer to as
**partial policies**, since they must be used in combination with other
policies. Partial policies work by providing an *outer policy* and an
*inner policy* as a template parameter to the outer policy. These give users
flexibility to create more complex execution patterns.

RAJA provides *outer* OpenMP CPU policies to create a parallel region in
which to execute a kernel. The outer policies require an inner policy that
defines how a kernel will execute in parallel inside the region.

 ====================================== ============= ==========================
 OpenMP CPU Outer Policies              Works with    Brief description
 ====================================== ============= ==========================
 omp_launch_t                           launch        Creates an OpenMP parallel
                                                      region. Same as applying
                                                      'omp parallel' pragma.
 omp_parallel_exec<InnerPolicy>         forall,       Creates OpenMP parallel
                                        kernel (For), region and requires an
                                        scan          **InnerPolicy**. Same as
                                                      applying 'omp parallel'
                                                      pragma.
 ====================================== ============= ==========================

The table below summarizes the *inner* policies that RAJA provides for OpenMP.
These policies are passed to the RAJA ``omp_parallel_exec`` outer policy as
a template argument as described above.

 ====================================== ============= ==========================
 OpenMP CPU Inner Policies              Works with    Brief description
 ====================================== ============= ==========================
 omp_for_exec                           forall,       Parallel execution within
                                        kernel (For), existing parallel
                                        launch (loop) region, specifically
                                        scan          apply the OpenMP pragma
                                                      'omp for schedule (auto)'
                                                      pragma.
 omp_for_static_exec<ChunkSize>         forall,       Same as applying
                                        kernel (For)  'omp for
                                                      schedule(static,
                                                      ChunkSize)'
 omp_for_nowait_static_exec<ChunkSize>  forall,       Same as applying
                                        kernel (For)  'omp for
                                                      schedule(static,
                                                      ChunkSize) nowait'
 omp_for_dynamic_exec<ChunkSize>        forall,       Same as applying
                                        kernel (For)  'omp for
                                                      schedule(dynamic,
                                                      ChunkSize)'
 omp_for_guided_exec<ChunkSize>         forall,       Same as applying
                                        kernel (For)  'omp for
                                                      schedule(guided,
                                                      ChunkSize)'
 omp_for_runtime_exec                   forall,       Same as applying
                                        kernel (For)  'omp for
                                                      schedule(runtime)'
 omp_parallel_collapse_exec             kernel        Use in Collapse statement
                                        (Collapse +   to parallelize multiple
                                        ArgList)      loop levels in loop nest
                                                      indicated using ArgList
 ====================================== ============= ==========================

.. note:: For the OpenMP scheduling policies above that take a ``ChunkSize``
          parameter, the chunk size is optional. If not provided, the
          default chunk size that the OpenMP implementation applies is used.
          For this case, the RAJA policy syntax is
          ``omp_parallel_for_{static|dynamic|guided}_exec< >``, which will
          result in the OpenMP pragma
          ``omp parallel for schedule({static|dynamic|guided})`` being applied.

.. important:: **RAJA only provides a nowait policy option for static
               scheduling** since that is the only case in which nowait can be
               used and be correct in general when executing multiple loops
               in a single parallel region. Paraphrasing the OpenMP standard:
               *programs that depend on which thread executes a particular
               loop iteration under any circumstance other than static schedule
               are non-conforming.*

As noted above, RAJA inner OpenMP policies must be used within an
**existing** parallel region to work properly. Embedding an inner
policy inside the RAJA outer ``omp_parallel_exec`` will allow you to
apply the OpenMP execution prescription specified by the policies to
a single kernel. To support use cases with multiple kernels inside an
OpenMP parallel region, RAJA provides a **region** construct that
takes a template argument to specify the execution back-end. For example::

  RAJA::region<RAJA::omp_parallel_region>([=]() {

    RAJA::forall<RAJA::omp_for_nowait_static_exec< > >(segment,
      [=] (int idx) {
        // do something at iterate 'idx'
      }
    );

    RAJA::forall<RAJA::omp_for_static_exec< > >(segment,
      [=] (int idx) {
        // do something else at iterate 'idx'
      }
    );

  });

Here, the ``RAJA::region<RAJA::omp_parallel_region>`` method call
creates an OpenMP parallel region, which contains two ``RAJA::forall``
kernels. The first uses the ``RAJA::omp_for_nowait_static_exec< >``
policy, meaning that no thread synchronization is needed after the
kernel. Thus, threads can start working on the second kernel while
others are still working on the first kernel. In general, this will
be correct when the iteration segments used in the two kernels are
the same and there are no loop carried dependences in either kernel.
Static scheduling is applied to both kernels. The second kernel uses the
``RAJA::omp_for_static_exec`` policy (without 'no wait' clause), which
means that all threads will complete before the kernel exits. In
this example, this is not really needed since there is no
more code to execute in the parallel region and the
``RAJA::omp_parallel_region`` construct applies a barrier
at the end of it.

.. _parallelregionpolicy-label:

-------------------------
Parallel Region Policies
-------------------------

Earlier, we discussed using the ``RAJA::region`` construct to
execute multiple kernels in an OpenMP parallel region. To support source code
portability, RAJA provides a sequential region concept that can be used to
surround code that uses execution back-ends other than OpenMP. This simplifies
switching user code between sequential and parallel execution and does not
require changing the code structure. For example::

  #if RAJA_ENABLE_OPENMP
    #define REGION_POLICY RAJA::omp_parallel_region
    #define EXEC_POLICY RAJA::omp_for_exec
  #else
    #define REGION_POLICY RAJA::seq_region
    #define EXEC_POLICY RAJA::seq_exec
  #endif

  RAJA::region<REGION_POLICY>([=]() {

     RAJA::forall<EXEC_POLICY>(segment, [=] (int idx) {
         // do something at iterate 'idx'
     } );

     RAJA::forall<EXEC_POLICY>(segment, [=] (int idx) {
         // do something else at iterate 'idx'
     } );

   });

.. note:: The sequential region specialization is essentially a *pass through*
          operation. It is provided so that if you want to turn off OpenMP in
          your code, for example, you can simply replace the region policy
          type and you do not have to change your algorithm source code.

.. _indexsetpolicy-label:

-----------------------------------------------------
RAJA IndexSet Execution Policies
-----------------------------------------------------

When an IndexSet iteration space is used in RAJA by passing a ``RAJA::IndexSet``
to a ``RAJA::forall`` method, an index set execution policy is
required. An index set execution policy is a **two-level policy**: an 'outer'
policy for iterating over segments in the index set, and an 'inner' policy
used to execute the iterations defined by each segment. An index set execution
policy type has the form::

  RAJA::ExecPolicy< segment_iteration_policy, segment_execution_policy >

In general, any policy that can be used with a ``RAJA::forall`` method
can be used as an (inner) segment execution policy. The following policies are
available to use for the outer segment iteration policy:

====================================== =========================================
Execution Policy                       Brief description
====================================== =========================================
**Serial**
seq_segit                              Iterate over index set segments
                                       sequentially.

**OpenMP CPU multithreading**
omp_parallel_segit                     Create OpenMP parallel region and
                                       iterate over segments in parallel inside
                                       it; i.e., apply ``omp parallel for``
                                       pragma on loop over segments.
omp_parallel_for_segit                 Same as above.
====================================== =========================================

.. _feat-policies-gpu-label:

-----------------------------------------------------
GPU Policies for CUDA and HIP
-----------------------------------------------------

RAJA policies for GPU execution using CUDA or HIP are similar. The only
syntactic difference is that CUDA policies have the prefix ``cuda_`` and HIP
policies have the prefix ``hip_``. In the tables below, policy names use
``cuda/hip_`` as shorthand for the corresponding CUDA or HIP policy when their
behavior is the same. Angle brackets indicate template parameters that are
used to specialize execution behavior.

Basic ``forall`` and ``launch`` policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These policies are the typical starting point for CUDA/HIP execution.
Use the ``exec`` policies for single-loop ``RAJA::forall`` kernels, scans, and sorts.
Use ``cuda/hip_launch_t`` when using ``RAJA::launch`` to create a device
execution environment that contains one or more RAJA loop kernels. The occupancy
variants are tuning policies for kernels where the default grid-size choice is
not the desired mapping.

.. list-table::
   :widths: 38 18 44
   :header-rows: 1

   * - Policy
     - Works with
     - Brief description
   * - cuda/hip_exec<BLOCK_SIZE>
     - forall, scan, sort
     - Execute loop iterations directly mapped to global threads in a GPU
       kernel launched with the given thread-block size and unbounded grid
       size. ``BLOCK_SIZE`` is required; there is no default.
   * - cuda/hip_exec_with_reduce<BLOCK_SIZE>
     - forall
     - Recommended for kernels containing reductions. It uses RAJA's internal
       occupancy calculation to choose a grid size that generally performs well
       for reduction kernels.
   * - cuda/hip_exec_base<with_reduce, BLOCK_SIZE>
     - forall
     - Chooses between ``cuda/hip_exec`` and ``cuda/hip_exec_with_reduce``
       based on the boolean ``with_reduce`` template parameter.
   * - cuda/hip_exec_grid<BLOCK_SIZE, GRID_SIZE>
     - forall
     - Execute loop iterations mapped to global threads via grid-stride loops
       in a GPU kernel launched with the given thread-block size and grid size.
       Both template parameters are required.
   * - cuda/hip_exec_occ_max<BLOCK_SIZE>
     - forall
     - Similar to ``cuda/hip_exec_grid``, but the grid size is bounded by the
       maximum occupancy of the kernel.
   * - cuda/hip_exec_occ_calc<BLOCK_SIZE>
     - forall
     - Similar to ``cuda/hip_exec_occ_max``, but may use less than maximum
       occupancy for performance reasons.
   * - cuda/hip_exec_occ_fraction<BLOCK_SIZE, Fraction<size_t, numerator, denominator>>
     - forall
     - Similar to ``cuda/hip_exec_occ_max``, but uses a fraction of the maximum
       occupancy of the kernel.
   * - cuda/hip_exec_occ_custom<BLOCK_SIZE, Concretizer>
     - forall
     - Similar to ``cuda/hip_exec_occ_max``, but the grid size is determined by
       the given concretizer.
   * - cuda/hip_launch_t
     - launch
     - Launches a device kernel. Code inside the lambda expression is executed
       on the device.

Mapping policy rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The thread, block, global, and flattened mapping policies described below share
a few important rules:

.. note:: * ``*_direct_unchecked`` policies do not mask out threads or blocks
            that are out of range. Use them only when the size of the range
            matches the selected block or grid shape.
          * ``*_direct`` policies mask out-of-range threads or blocks. Use them
            when the size of the range is **less than or equal to** the selected
            block or grid shape.
          * ``*_loop`` policies perform a block- or grid-stride loop. Use them
            when the loop iteration space may exceed the selected block or grid
            shape, or when the loop size is not known to fit a direct mapping.
          * Repeating direct or direct-unchecked policies with the same
            dimension in perfectly nested loops is not recommended. The code may
            compile and run, but likely will not do what you expect.
          * If multiple direct or direct-unchecked policies are used in a kernel
            with different dimensions, the product of the corresponding
            iteration-space sizes cannot be greater than the maximum allowed
            threads per block or blocks per grid. Attempting to exceed that
            limit will cause the CUDA/HIP runtime to report illegal launch
            parameters.

Thread mapping policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These policies map one loop level to GPU threads within a thread block. They
are most useful inside ``RAJA::kernel`` or ``RAJA::launch`` when expressing a
nested loop mapping, especially for tiled loop bodies where a tile is processed
by the threads in a block. Prefer ``*_loop`` variants when the loop extent may
be larger than the available threads in the selected dimension. Thread-direct
policies are recommended only for loop patterns, such as block tiling, that
produce small fixed-size iteration spaces within each block.

.. note:: ``cuda/hip_thread_loop`` policies are not safe to use with
          ``Cuda/HipSyncThreads``. Use
          ``cuda/hip_thread_syncable_loop<dims...>`` instead. For example,
          use ``cuda_thread_syncable_loop<named_dim::x>`` instead of
          ``cuda_thread_x_loop``.

.. list-table::
   :widths: 38 18 44
   :header-rows: 1

   * - Policy family
     - Works with
     - Brief description
   * - cuda/hip_thread_{x,y,z}_direct_unchecked
     - kernel ``For``, launch ``loop``
     - Map loop iterates directly to GPU threads in the selected dimension,
       without checking loop bounds. Each thread handles one iterate.
   * - cuda/hip_thread_{x,y,z}_direct
     - kernel ``For``, launch ``loop``
     - Map loop iterates directly to GPU threads in the selected dimension,
       with bounds masking.
   * - cuda/hip_thread_{x,y,z}_loop
     - kernel ``For``, launch ``loop``
     - Map loop iterates to GPU threads in the selected dimension using a
       block-stride loop.
   * - cuda/hip_thread_syncable_loop<dims...>
     - kernel ``For``, launch ``loop``
     - Similar to ``cuda/hip_thread_{x,y,z}_loop``, but safe to use with
       ``Cuda/HipSyncThreads``.
   * - cuda/hip_thread_size_{x,y,z}_direct_unchecked<n_threads>
     - kernel ``For``, launch ``loop``
     - Compile-time-size version of
       ``cuda/hip_thread_{x,y,z}_direct_unchecked``.
   * - cuda/hip_thread_size_{x,y,z}_direct<n_threads>
     - kernel ``For``, launch ``loop``
     - Compile-time-size version of ``cuda/hip_thread_{x,y,z}_direct``.

Flattened thread mapping policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These policies are used for ``RAJA::launch`` loops that need to treat a
multi-dimensional thread team as a one-dimensional set of workers. They are
useful when an algorithm naturally has a linear indexing operation but the
launch configuration is specified with multiple thread dimensions.

.. list-table::
   :widths: 38 18 44
   :header-rows: 1

   * - Policy family
     - Works with
     - Brief description
   * - cuda/hip_flatten_threads_{xyz}_direct_unchecked
     - launch ``loop``
     - Reshape threads in a multi-dimensional thread team into one dimension
       without checking loop bounds. Accepts any permutation of one, two, or
       three dimensions.
   * - cuda/hip_flatten_threads_{xyz}_direct
     - launch ``loop``
     - Same as above, but with direct mapping and bounds masking.
   * - cuda/hip_flatten_threads_{xyz}_loop
     - launch ``loop``
     - Same as above, but with strided-loop mapping.

Block mapping policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These policies map one loop level to GPU thread blocks. They are useful for
outer loop levels, tiles, or other coarse-grained work where each block handles
one or more iterations. Prefer block ``*_direct`` or ``*_direct_unchecked``
policies for tiled patterns when each block owns a known tile, and use
``*_loop`` variants when the loop extent may exceed the grid shape.

.. note:: CUDA/HIP block-direct-unchecked or block-direct policies may be
          preferable to block-loop policies when block load balancing may be an
          issue. In those cases, block-direct-unchecked or block-direct policies
          may yield better performance.

.. list-table::
   :widths: 38 18 44
   :header-rows: 1

   * - Policy family
     - Works with
     - Brief description
   * - cuda/hip_block_{x,y,z}_direct_unchecked
     - kernel ``For``, launch ``loop``
     - Map loop iterates directly to GPU thread blocks in the selected
       dimension, without checking loop bounds. Each block handles one iterate.
   * - cuda/hip_block_{x,y,z}_direct
     - kernel ``For``, launch ``loop``
     - Map loop iterates directly to GPU thread blocks in the selected
       dimension, with bounds masking.
   * - cuda/hip_block_{x,y,z}_loop
     - kernel ``For``, launch ``loop``
     - Map loop iterates to GPU thread blocks in the selected dimension using a
       grid-stride loop.
   * - cuda/hip_block_size_{x,y,z}_direct_unchecked<n_blocks>
     - kernel ``For``, launch ``loop``
     - Compile-time-size version of
       ``cuda/hip_block_{x,y,z}_direct_unchecked``.
   * - cuda/hip_block_size_{x,y,z}_direct<n_blocks>
     - kernel ``For``, launch ``loop``
     - Compile-time-size version of ``cuda/hip_block_{x,y,z}_direct``.
   * - cuda/hip_block_size_{x,y,z}_loop<n_blocks>
     - kernel ``For``, launch ``loop``
     - Compile-time-size version of ``cuda/hip_block_{x,y,z}_loop``.

Global thread mapping policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These policies map loop iterations to the unique global thread id formed from
the block and thread indices. They are often the most direct fit for simple
data-parallel loops in ``RAJA::kernel`` or ``RAJA::launch`` because each
iteration can be assigned to one global GPU thread. They may be inappropriate
for kernels that require block-local synchronization, where thread or block
mapping gives more explicit control.

.. note:: Global-direct-sized policies are recommended for most loop patterns,
          but may be inappropriate for kernels using block-level
          synchronization.

.. list-table::
   :widths: 38 18 44
   :header-rows: 1

   * - Policy family
     - Works with
     - Brief description
   * - cuda/hip_global_{x,y,z}_direct_unchecked
     - kernel ``For``, launch ``loop``
     - Map loop iterates directly to global GPU thread ids in the selected
       dimension, without checking loop bounds. In the x dimension, this is
       equivalent to ``threadIdx.x + blockDim.x * blockIdx.x``.
   * - cuda/hip_global_{x,y,z}_direct
     - kernel ``For``, launch ``loop``
     - Map loop iterates directly to global GPU thread ids in the selected
       dimension, with bounds masking.
   * - cuda/hip_global_{x,y,z}_loop
     - kernel ``For``, launch ``loop``
     - Map loop iterates to global GPU thread ids in the selected dimension
       using a grid-stride loop.
   * - cuda/hip_global_size_{x,y,z}_direct_unchecked<n_threads>
     - kernel ``For``, launch ``loop``
     - Compile-time-size version of
       ``cuda/hip_global_{x,y,z}_direct_unchecked``.
   * - cuda/hip_global_size_{x,y,z}_direct<n_threads>
     - kernel ``For``, launch ``loop``
     - Compile-time-size version of ``cuda/hip_global_{x,y,z}_direct``.
   * - cuda/hip_global_size_{x,y,z}_loop<n_threads>
     - kernel ``For``, launch ``loop``
     - Compile-time-size version of ``cuda/hip_global_{x,y,z}_loop``.

Warp and block reduction mapping policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These policies are specialized building blocks for warp-level and block-level
execution patterns inside ``RAJA::kernel``. Use them when the algorithm needs
explicit warp mapping or a reduction across a single warp or block. For ordinary
``RAJA::forall`` reductions, start with a RAJA reduction object and a compatible
CUDA/HIP reduction policy as described in earlier examples instead.

.. list-table::
   :widths: 38 18 44
   :header-rows: 1

   * - Policy
     - Works with
     - Brief description
   * - cuda/hip_warp_direct_unchecked
     - kernel ``For``
     - Map work directly to threads in a warp without checking loop bounds.
       Cannot be used with ``cuda/hip_thread_x_*`` policies. Multiple warps can
       be created by using ``cuda/hip_thread_y/z_*`` policies.
   * - cuda/hip_warp_direct
     - kernel ``For``
     - Similar to ``cuda/hip_warp_direct_unchecked``, but with direct mapping
       semantics.
   * - cuda/hip_warp_loop
     - kernel ``For``
     - Similar to ``cuda/hip_warp_direct``, but maps work to threads in a warp
       using a warp-stride loop.
   * - cuda/hip_warp_masked_direct<BitMask<..>>
     - kernel ``For``
     - Map work directly to threads in a warp using a bit mask. Cannot be used
       with ``cuda/hip_thread_x_*`` policies. Multiple warps can be created by
       using ``cuda/hip_thread_y/z_*`` policies.
   * - cuda/hip_warp_masked_loop<BitMask<..>>
     - kernel ``For``
     - Map work to threads in a warp using a bit mask and a warp-stride loop.
       Cannot be used with ``cuda/hip_thread_x_*`` policies. Multiple warps can
       be created by using ``cuda/hip_thread_y/z_*`` policies.
   * - cuda/hip_block_reduce
     - kernel ``Reduce``
     - Perform a reduction across a single GPU thread block.
   * - cuda/hip_warp_reduce
     - kernel ``Reduce``
     - Perform a reduction across a single GPU thread warp.

Occupancy concretizers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a CUDA or HIP policy leaves parameters like the block size and/or grid size
unspecified, such as ``cuda/hip_exec_occ_custom`` in the table above, a
concretizer object is used to decide those parameters. RAJA provides the
following concretizers to use with the ``cuda/hip_exec_occ_custom`` policies:

+----------------------------------------------------+-----------------------------------------+
| Execution Policy                                   | Brief description                       |
+====================================================+=========================================+
| Cuda/HipDefaultConcretizer                         | The default concretizer, expected to    |
|                                                    | provide good performance in general.    |
|                                                    | Note that it may not use max occupancy. |
+----------------------------------------------------+-----------------------------------------+
| Cuda/HipRecForReduceConcretizer                    | Expected to provide good performance    |
|                                                    | in loops with reducers.                 |
|                                                    | Note that it may not use max occupancy. |
+----------------------------------------------------+-----------------------------------------+
| Cuda/HipMaxOccupancyConcretizer                    | Uses max occupancy.                     |
+----------------------------------------------------+-----------------------------------------+
| Cuda/HipAvoidDeviceMaxThreadOccupancyConcretizer   | Avoids using the max occupancy of the   |
|                                                    | device in terms of threads.             |
|                                                    | Note that it may use the max occupancy  |
|                                                    | of the kernel if that is below the max  |
|                                                    | occupancy of the device.                |
+----------------------------------------------------+-----------------------------------------+
| Cuda/HipFractionOffsetOccupancyConcretizer<        | Uses a fraction and offset to choose an |
| Fraction<size_t, numerator, denominator>,          | occupancy based on the max occupancy    |
| BLOCKS_PER_SM_OFFSET>                              | using the following formula:            |
|                                                    | (Fraction * kernel_max_blocks_per_sm +  |
|                                                    | BLOCKS_PER_SM_OFFSET) * sm_per_device   |
+----------------------------------------------------+-----------------------------------------+

Several notes regarding the CUDA/HIP policy implementation allow you to
write more explicit policies.

.. note:: * Policies are a class template like cuda/hip_exec_explicit or
            cuda/hip_indexer. The various template parameters specify the
            behavior of the policy.
          * Policies have a mapping from loop iterations to iterates in the
            index set via a iteration_mapping enum template parameter. The
            possible values are DirectUnchecked, Direct, and StridedLoop.
          * Policies can be safely used with some synchronization constructs
            via a kernel_sync_requirement enum template parameter. **The
            possible values are none and sync.**
          * Policies get their indices via an iteration getter class template
            like cuda/hip::IndexGlobal.
          * Iteration getters can be used with different dimensions via the
            named_dim enum. The possible values are x, y and z.
          * Iteration getters know the number of threads per block (block_size)
            and number of blocks per grid (grid_size) via integer template
            parameters. These can be positive integers, in which case they must
            match the number used in the kernel launch. These can also be values
            of the named_usage enum. The possible values are unspecified and
            ignored. For example, in cuda_thread_x_direct block_size is
            unspecified so a runtime number of threads is used, but grid_size is
            ignored so blocks are ignored when getting indices.

.. _feat-policies-sycl-label:

-----------------------------------------------------
GPU Policies for SYCL
-----------------------------------------------------

.. note:: SYCL uses C++-style ordering for its work group and global thread
          dimension/indexing types. This is due, in part, to SYCL's closer
          alignment with C++ multi-dimensional indexing, which is "row-major".
          This is the reverse of the thread indexing used in CUDA or HIP,
          which is "column-major". For example, suppose we have a thread-block
          or work-group where we specify the shape as (nx, ny, nz). Consider
          an element in the thread-block or work-group with id (x, y, z).
          In CUDA or HIP, the element index is x + y * nx + z * nx * ny. In
          SYCL, the element index is z + y * nz + x * nz * ny.

          In terms of the CUDA or HIP built-in variables to support threads,
          we have::

            Thread ID: threadIdx.x/y/z
            Block ID: blockIdx.x/y/z
            Block dimension: blockDim.x/y/z
            Grid dimension: gridDim.x/y/z

          The analogues in SYCL are::

            Thread ID: sycl::nd_item.get_local_id(2/1/0)
            Work-group ID: sycl::nd_item.get_group(2/1/0)
            Work-group dimensions: sycl::nd_item.get_local_range().get(2/1/0)
            ND-range dimensions: sycl::nd_item.get_group_range(2/1/0)

	  When using ``RAJA::launch``, thread and block configuration
	  follows CUDA and HIP programming models and is always
	  configured in three-dimensions. This means that SYCL dimension
	  2 always exists and should be used as one would use the
	  x dimension for CUDA and HIP.

          Similarly, ``RAJA::kernel`` uses a three-dimensional work-group
          configuration. SYCL dimension 2 always exists and should be used as
          one would use the x dimension in CUDA and HIP.

Policies with ``{0,1,2}`` indicate that the same policy family is available for
each SYCL dimension. See the note above for how SYCL dimensions relate to
CUDA/HIP x/y/z-style indexing.

.. list-table::
   :widths: 38 18 44
   :header-rows: 1

   * - SYCL execution policy family
     - Works with
     - Brief description
   * - sycl_exec<WORK_GROUP_SIZE>
     - forall
     - Execute loop iterations in a GPU kernel launched with the given work
       group size.
   * - sycl_launch_t
     - launch
     - Launches a SYCL kernel. Code inside the lambda expression is executed on
       the device.
   * - sycl_global_{0,1,2}<WORK_GROUP_SIZE>
     - kernel ``For``
     - Map loop iterates directly to SYCL global ids in the selected dimension,
       one iterate per work item. Group execution into work groups of the given
       size.
   * - sycl_global_item_{0,1,2}
     - launch ``loop``
     - Create a unique global work-item id in the selected dimension. For
       dimension 0, this is equivalent to
       ``itm.get_group(0) * itm.get_local_range(0) + itm.get_local_id(0)``.
       Similarly, for dimensions 1 and 2.
   * - sycl_local_{0,1,2}_direct
     - kernel ``For``, launch ``loop``
     - Map loop iterates directly to SYCL local work-items in the selected
       dimension, one iterate per work item.
   * - sycl_local_{0,1,2}_loop
     - kernel ``For``, launch ``loop``
     - Map loop iterates to SYCL local work-items in the selected dimension
       using a work-group-stride loop.
   * - sycl_group_{0,1,2}_direct
     - kernel ``For``, launch ``loop``
     - Map loop iterates directly to SYCL group ids in the selected dimension,
       one iterate per group.
   * - sycl_group_{0,1,2}_loop
     - kernel ``For``, launch ``loop``
     - Map loop iterates to SYCL group ids in the selected dimension using a
       group-stride loop.

.. _feat-policies-omp-target-label:

-----------------------------------------------------
OpenMP Target Offload Policies
-----------------------------------------------------

RAJA provides policies to use OpenMP to offload kernel execution to a GPU
device, for example. They are summarized in the following table.

 ====================================== ============= ==========================
 OpenMP Target Execution Policies       Works with    Brief description
 ====================================== ============= ==========================
 omp_target_parallel_for_exec<#>        forall,       Create parallel target
                                        kernel(For)   region and execute with
                                                      given number of threads
                                                      per team inside it. Number
                                                      of teams is calculated
                                                      internally; i.e.,
                                                      apply ``omp teams
                                                      distribute parallel for
                                                      num_teams(iteration space
                                                      size/#)
                                                      thread_limit(#)`` pragma
 omp_target_parallel_collapse_exec      kernel        Similar to above, but
                                        (Collapse)    collapse
                                                      *perfectly-nested*
                                                      loops, indicated in
                                                      arguments to RAJA
                                                      Collapse statement. Note:
                                                      compiler determines number
                                                      of thread teams and
                                                      threads per team
 ====================================== ============= ==========================

.. _feat-policies-gpu-aliases-label:

-----------------------------------------------------
Device Policy Aliases
-----------------------------------------------------

To simplify transitions between GPU back-ends (CUDA/HIP/SYCL) and reduce
downstream preprocessor conditionals, RAJA provides a set of
``device_*`` policy aliases that resolve to the *active* GPU back-end.
Use these aliases when code should follow the GPU back-end selected in the
RAJA build configuration. Use explicit ``cuda_*``, ``hip_*``, or ``sycl_*``
policies when code depends on behavior that is specific to one back-end.

.. note:: These aliases do not cover all RAJA policies for CUDA/HIP or SYCL.

For example, ``device_exec`` and ``device_reduce`` can be used together so
the loop execution and reduction policies both follow the active GPU back-end:

.. code-block:: C++

  using EXEC_POL = RAJA::device_exec<256>;
  using REDUCE_POL = RAJA::device_reduce;

  RAJA::ReduceSum<REDUCE_POL, double> sum(0.0);

  RAJA::forall<EXEC_POL>(segment,
    [=] RAJA_DEVICE (RAJA::Index_type i) {
      sum += values[i];
    });

.. note:: Use ``RAJA_HOST_DEVICE`` instead of ``RAJA_DEVICE`` when the same
          lambda may be used with either GPU or CPU execution policies. This
          makes the code more portable because one can swap execution policies
          to run on the device or host without changing the lambda annotation.

The table below summarizes coverage of the device alias policies available in RAJA.
In the table, *partial* means that the alias family exists for that
back-end, but not every variant in the row is available for that back-end.
The table lists aliases available when building with a GPU device back-end
(i.e., when ``ENABLE_CUDA``, ``ENABLE_HIP``, or ``RAJA_ENABLE_SYCL`` is turned
on in a RAJA build configuration).

.. list-table::
   :widths: 34 14 14 14 24
   :header-rows: 1

   * - Device policy alias family
     - CUDA
     - HIP
     - SYCL
     - Notes
   * - device_exec*
     - yes
     - yes
     - partial
     - Execution policy aliases for simple GPU loops, for example
       ``device_exec<256>``. SYCL supports ``device_exec`` and
       ``device_exec_async``; CUDA/HIP occupancy and reduction exec variants do
       not have SYCL aliases.
   * - device_atomic and device_atomic_explicit<host_policy>
     - yes
     - yes
     - yes
     - Atomic policy aliases for all three GPU back-ends, for example
       ``device_atomic_explicit<RAJA::seq_atomic>``.
   * - device_reduce
     - yes
     - yes
     - yes
     - Default reduction policy alias for the active GPU back-end.
   * - device_reduce_atomic and device_reduce_base<with_atomic>
     - yes
     - yes
     - no
     - Reduction tuning aliases for CUDA/HIP, for example
       ``device_reduce_base<true>``. SYCL currently only exposes
       ``device_reduce``.
   * - device_multi_reduce_atomic and
       device_multi_reduce_atomic_low_performance_low_overhead
     - yes
     - yes
     - no
     - Multi-reduction aliases for CUDA/HIP. SYCL does not currently provide a
       back-end-equivalent ``device_*`` mapping.
   * - device_launch_t
     - yes
     - yes
     - yes
     - Launch policy alias for the active GPU back-end, for example
       ``device_launch_t<false>``.
   * - device_global_size_{x,y,z}_{direct,direct_unchecked,loop}<N>
     - yes
     - yes
     - partial
     - Size-templated global mapping aliases, for example
       ``device_global_size_x_direct<64>``. SYCL currently exposes only the
       direct x/y/z aliases; unchecked and loop variants are CUDA/HIP only.
   * - device_thread_{x,y,z}_{direct,loop}
     - yes
     - yes
     - yes
     - Thread mapping aliases for one GPU dimension.
   * - device_thread_size_{x,y,z}_{direct,direct_unchecked,loop}<N>
     - yes
     - yes
     - no
     - Size-templated thread mapping aliases, for example
       ``device_thread_size_x_direct<128>``. These aliases are declared as
       compile-time errors under SYCL.
   * - device_block_{x,y,z}_{direct,loop}
     - yes
     - yes
     - yes
     - Block mapping aliases for one GPU dimension.
   * - device_block_size_{x,y,z}_{direct,direct_unchecked,loop}<N>
     - yes
     - yes
     - no
     - Size-templated block mapping aliases, for example
       ``device_block_size_x_direct<128>``. These aliases are declared as
       compile-time errors under SYCL.
   * - device_flatten_thread_size_*, device_flatten_block_size_*,
       device_flatten_global_size_*
     - yes
     - yes
     - no
     - Size-templated flattened mapping aliases, for example
       ``device_flatten_thread_size_x_direct<128>``,
       ``device_flatten_block_size_x_direct<128>``, and
       ``device_flatten_global_size_x_direct<128>``. These flattened size
       aliases exist for CUDA/HIP but are declared as compile-time errors
       under SYCL.

.. important::
   For SYCL, these aliases use CUDA-like ``(x,y,z)`` naming with the standard
   RAJA mapping described above: ``x`` corresponds to SYCL dimension 2,
   ``y`` to SYCL dimension 1, and ``z`` to SYCL dimension 0. These build options enable
   the corresponding internal ``RAJA_*_ACTIVE`` compile-time macros used by
   the implementation. Device aliases that have no SYCL equivalent are
   intentionally not defined under SYCL as usable policies. Attempting to use
   them will cause compile-time failure so unsupported code paths are caught
   immediately.

See also the example code ``examples/device-policy-aliases.cpp``.
