.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _feat-policies-label:

==================
Policies
==================

RAJA kernel execution methods take an execution policy type template parameter
to specialize execution behavior. Typically, the policy indicates which
programming model back-end to use and other information about the execution
pattern, such as number of CUDA threads per thread block, whether execution is
synchronous or asynchronous, etc. This section describes RAJA policies for
loop kernel execution, scans, sorts, reductions, atomics, etc. Please
see detailed examples in :ref:`tutorial-label` for a variety of use cases.

As RAJA functionality evolves, new policies are added and some may
be redefined and to work in new ways.

.. note:: * All RAJA policies are in the namespace ``RAJA``.
          * All RAJA policies have a prefix indicating the back-end
            implementation that they use; e.g., ``omp_`` for OpenMP, ``cuda_``
            for CUDA, etc.

-----------------------------------------------------
Policy Basics
-----------------------------------------------------

An execution policy is a C++ type that tells RAJA how to run a loop, kernel,
or related operation. In most cases, choosing a policy answers two questions:

  * Which back-end should run the work, such as sequential CPU, OpenMP, CUDA,
    HIP, SYCL, or OpenMP target?
  * Which execution pattern should RAJA use, such as a simple loop, a nested
    loop mapping, a reduction, or an atomic update?

Different RAJA features use different policy categories. The policy categories
are related, but they are not interchangeable.

====================================== ========================================
Policy category                        Used for
====================================== ========================================
Loop execution policies                ``RAJA::forall`` loops, scans, sorts,
                                       and individual loop levels in
                                       ``RAJA::kernel`` and ``RAJA::launch``.
Launch policies                        Creating an execution environment for
                                       ``RAJA::launch``.
Index set execution policies           Running a ``RAJA::IndexSet`` iteration
                                       pattern composed of multiple index
                                       segments by choosing one policy for
                                       iterating over segments and another
                                       policy for executing the 
                                       loop iterations within each segment.
Reduction and multi-reduction policies ``RAJA::Reduce*`` and
                                       ``RAJA::MultiReduce*`` objects.
Atomic policies                        ``RAJA::atomic*`` operations.
Local array memory policies            Choosing where ``RAJA::LocalArray``
                                       storage lives.
====================================== ========================================

For a first implementation, users usually start with one of the policies below
and move to more specialized policies only when they need a specific scheduling,
mapping, synchronization, or performance behavior.

============================================== =================================
Common need                                    Typical starting policy
============================================== =================================
Sequential CPU loop                            ``seq_exec``
CPU loop with SIMD compiler hints              ``simd_exec``
OpenMP parallel CPU loop                       ``omp_parallel_for_exec``
OpenMP launch region                           ``omp_launch_t`` with
                                               ``omp_for_exec`` loops
CUDA ``forall`` loop                           ``cuda_exec<BLOCK_SIZE>``
HIP ``forall`` loop                            ``hip_exec<BLOCK_SIZE>``
Portable GPU ``forall`` loop                   ``device_exec<BLOCK_SIZE>``
SYCL ``forall`` loop                           ``sycl_exec<WORK_GROUP_SIZE>``
Sequential reduction                           ``seq_reduce``
OpenMP reduction                               ``omp_reduce``
CUDA/HIP/SYCL reduction                        ``cuda_reduce``, ``hip_reduce``,
                                               or ``sycl_reduce``
Atomic update                                  Match the atomic policy to the
                                               loop back-end, such as
                                               ``omp_atomic`` or
                                               ``cuda_atomic``
============================================== =================================

The tutorial sections provide worked examples that are easier to follow than
the full reference tables on this page. See :ref:`tut-kernelexecpols-label`
for ``RAJA::kernel`` examples and :ref:`tut-launchexecpols-label` for
``RAJA::launch`` examples.

-----------------------------------------------------
How to Read Policy Names
-----------------------------------------------------

RAJA policy names are intentionally descriptive. Most names combine a back-end
prefix with words that describe where work runs and how iterations are mapped.
Understanding these pieces makes the reference tables easier to scan.

Back-end prefixes identify the implementation used by the policy:

================= =============================================================
Prefix            Meaning
================= =============================================================
``seq_``          Sequential CPU execution.
``simd_``         Sequential CPU execution with compiler vectorization hints.
``omp_``          OpenMP CPU threading.
``omp_target_``   OpenMP target offload.
``cuda_``         CUDA device execution.
``hip_``          HIP device execution.
``sycl_``         SYCL device execution.
``device_``       Alias for the active GPU device back-end when RAJA is built
                  with CUDA, HIP, or SYCL support.
================= =============================================================

Common words in policy names describe the execution structure:

================= =============================================================
Name part         Meaning
================= =============================================================
``exec``          A loop execution policy, often usable with ``RAJA::forall``.
``launch``        A policy that creates an execution environment for
                  ``RAJA::launch``.
``parallel_for``  An OpenMP policy that creates a parallel loop.
``for``           An OpenMP loop inside an existing parallel region.
``thread``        Map loop iterations to GPU threads or work-items.
``block``         Map loop iterations to GPU thread blocks or work-groups.
``global``        Map loop iterations to a unique global GPU thread or
                  work-item index.
``warp``          Map work to CUDA/HIP warp-level execution.
``reduce``        A policy for RAJA reduction objects or reduction statements.
``atomic``        A policy for RAJA atomic operations.
================= =============================================================

Several GPU mapping suffixes appear often:

======================= =======================================================
Suffix                  Meaning
======================= =======================================================
``_loop``               Use a strided loop. This is usually the most forgiving
                        choice when the iteration space may be larger than the
                        available threads, blocks, or work-items.
``_direct``             Map iterations directly and mask out-of-range
                        iterations. This is useful when the iteration space is
                        known to fit within the chosen execution shape.
``_direct_unchecked``   Map iterations directly without bounds checks. Use this
                        only when the iteration space exactly matches the
                        execution shape.
``_size_*``             Use a compile-time size supplied as a template
                        parameter, such as
                        ``cuda_thread_size_x_direct<128>``.
======================= =======================================================

Template parameters customize a policy at compile time. For example,
``cuda_exec<256>`` launches CUDA kernels with 256 threads per thread block, and
``omp_parallel_for_static_exec<4>`` requests OpenMP static scheduling with a
chunk size of 4. Some template parameters are optional; the policy descriptions
below call out those cases explicitly.

-----------------------------------------------------
Choosing a Policy
-----------------------------------------------------

The tables below are reference material. When choosing a policy for new code,
start from the RAJA interface you are using, then choose the execution back-end,
then add specialized policies only when a kernel needs them.

Choose the RAJA interface first:

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - If your code uses
     - Start with
   * - ``RAJA::forall``
     - A single loop execution policy such as ``seq_exec``,
       ``omp_parallel_for_exec``, ``cuda_exec<BLOCK_SIZE>``,
       ``hip_exec<BLOCK_SIZE>``, or ``sycl_exec<WORK_GROUP_SIZE>``.
   * - ``RAJA::kernel``
     - A ``RAJA::KernelPolicy`` constructed as a sequence of statements. Each
       ``RAJA::statement::For`` chooses a loop execution policy for one loop
       level.
   * - ``RAJA::launch``
     - A launch policy such as ``seq_launch_t``, ``omp_launch_t``, or
       ``cuda_launch_t``, to create an execution regions, plus loop policies
       inside the launch body.
   * - ``RAJA::IndexSet`` with ``RAJA::forall``
     - A ``RAJA::ExecPolicy`` with one policy for iterating over index-set
       segments and one policy for executing iterations within each segment.
   * - Reductions, multi-reductions, or atomics
     - A helper policy that is compatible with the loop execution policy used
       by the loop kernel.

Next, choose the back-end:

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Back-end
     - Typical starting point
   * - Sequential CPU
     - Use ``seq_exec`` for loops and ``seq_reduce`` for reductions.
   * - CPU with SIMD hints
     - Use ``simd_exec`` only for loops that that are data parallel; e.g.,
       they do not use RAJA reductions or multi-reductions.
   * - OpenMP CPU threading
     - Use ``omp_parallel_for_exec`` for a single ``RAJA::forall`` loop. Use
       ``omp_launch_t`` with ``omp_for_exec`` for ``RAJA::launch``.
   * - CUDA or HIP
     - Use ``cuda_exec<BLOCK_SIZE>`` or ``hip_exec<BLOCK_SIZE>`` for simple
       ``RAJA::forall`` loops. Use mapping policies such as
       ``cuda_thread_x_loop`` or ``hip_block_x_direct`` when expressing a
       nested loop or launch mapping.
   * - Active GPU device back-end
     - Use ``device_*`` aliases when code should follow the enabled CUDA, HIP,
       or SYCL back-end without downstream preprocessor conditionals.
   * - SYCL
     - Use ``sycl_exec<WORK_GROUP_SIZE>`` for simple ``RAJA::forall`` loops.
       Pay attention to the SYCL dimension-ordering note below when using
       lower-level mapping policies.
   * - OpenMP target
     - Use ``omp_target_parallel_for_exec<#>`` for OpenMP target offload.

Then choose specialized behavior when needed:

.. list-table::
   :widths: 32 68
   :header-rows: 1

   * - Need
     - Policy choice
   * - OpenMP schedule control
     - Use static, dynamic, guided, or runtime OpenMP policy variants.
   * - Multiple loops in one OpenMP parallel region
     - Use ``RAJA::region`` with OpenMP inner policies such as
       ``omp_for_exec`` or ``omp_for_static_exec``.
   * - Nested loop reordering or collapse
     - Use ``RAJA::kernel`` statements such as ``RAJA::statement::For`` and
       ``RAJA::statement::Collapse``.
   * - GPU tiled loop mapping
     - Use GPU thread, block, or global mapping policies. Prefer ``*_loop``
       until the direct mapping constraints are understood. The ``*_loop``,
       while potentially less optimal, are most forgiving and do not need
       the mapping constraints of other policies.
   * - GPU reduction inside ``RAJA::forall``
     - Use the reduction-aware CUDA/HIP execution policy variants where
       appropriate, and match the reducer policy to the loop back-end.
   * - Atomic update
     - Match the atomic policy to the loop back-end, or use ``auto_atomic``
       where supported.

-----------------------------------------------------
Basic Policy Examples
-----------------------------------------------------

The examples below show how to usethe basic ``RAJA::forall`` construct to execute
a simple loop kernel with various execution policies. In each case, the lambda
expression body describes the work done for each kernel iterate and the
policy selects where and how the loop runs. For brevity, these snippets omit allocation,
data movement, and backend-specific build guards.

Sequential CPU execution:

.. code-block:: C++

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, N),
    [=] (RAJA::Index_type i) {
      y[i] = a * x[i] + y[i];
    });

OpenMP parallel CPU execution:

.. code-block:: C++

  RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0, N),
    [=] (RAJA::Index_type i) {
      y[i] = a * x[i] + y[i];
    });

CUDA and HIP execution policies take a block-size template parameter. Device lambdas must
be marked with the appropriate device annotation. Here, the ``RAJA_DEVICE``
macro constant is shown:

.. code-block:: C++

  RAJA::forall<RAJA::cuda_exec<256>>(RAJA::RangeSegment(0, N),
    [=] RAJA_DEVICE (RAJA::Index_type i) {
      y[i] = a * x[i] + y[i];
    });

  RAJA::forall<RAJA::hip_exec<256>>(RAJA::RangeSegment(0, N),
    [=] RAJA_DEVICE (RAJA::Index_type i) {
      y[i] = a * x[i] + y[i];
    });

.. note:: Use ``RAJA_HOST_DEVICE`` instead of ``RAJA_DEVICE`` when the same
          lambda may be used with either GPU or CPU execution policies. This
          makes the code more portable because one can swap execution policies
          to run on the device or host without changing the lambda annotation.

When code should use whichever GPU back-end is active in the RAJA build, use
the ``device_*`` aliases:

.. code-block:: C++

  RAJA::forall<RAJA::device_exec<256>>(RAJA::RangeSegment(0, N),
    [=] RAJA_DEVICE (RAJA::Index_type i) {
      y[i] = a * x[i] + y[i];
    });

Reduction policies are separate from loop execution policies. The reducer
policy must support the loop policy used by the kernel:

.. code-block:: C++

  RAJA::ReduceSum<RAJA::omp_reduce, double> sum(0.0);

  RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0, N),
    [=] (RAJA::Index_type i) {
      sum += x[i];
    });

For complete examples with setup, memory management, and backend-specific build
guards, see :ref:`tut-kernelexecpols-label`, :ref:`tut-launchexecpols-label`,
and :ref:`feat-reductions-label`.

-----------------------------------------------------
RAJA Loop/Kernel Execution Policies
-----------------------------------------------------

The following tables summarize RAJA policies for executing kernels.
Please see notes below policy descriptions for additional usage details and
caveats.


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
the same and their are no loop carried dependences in either kernel.
Static scheduling is applied to both kernels. The second kernel uses the 
``RAJA::omp_for_static_exec`` policy (without 'no wait' clause), which
means that all threads will complete before the kernel exits. In
this example, this is not really needed since there is no
more code to execute in the parallel region and the 
``RAJA::omp_parallel_region`` construct applies a barrier
at the end of it.

-------------------------
Parallel Region Policies
-------------------------

Earlier, we discussed using the ``RAJA::region`` construct to
execute multiple kernels in an OpenMP parallel region. To support source code
portability, RAJA provides a sequential region concept that can be used to
surround code that uses execution back-ends other than OpenMP. This simplifies
switching user code between sequential and parallel execution and does not
require changing the code structure. For example::

  RAJA::region<RAJA::seq_region>([=]() {

     RAJA::forall<RAJA::seq_exec>(segment, [=] (int idx) {
         // do something at iterate 'idx'
     } );

     RAJA::forall<RAJA::seq_exec>(segment, [=] (int idx) {
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
concretizer object is used to decide those parameters. RAJA provides the following concretizers
to use with the ``cuda/hip_exec_occ_custom`` policies:

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
| Fraction<size_t, numerator, denomenator>,          | occupancy based on the max occupancy    |
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

.. list-table:: SYCL Execution Policies
   :widths: 38 18 44
   :header-rows: 1

   * - Policy family
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
       Similary, for dimensions 1 and 2.
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

-----------------------------------------------------
Device policy aliases
-----------------------------------------------------

To simplify transitions between GPU back-ends (CUDA/HIP/SYCL) and reduce
downstream preprocessor conditionals, RAJA provides a set of
``device_*`` policy aliases that resolve to the *active* GPU back-end.

.. note:: These aliases do not cover all RAJA policies for CUDA/HIP.

In particular, the following aliases are available when building with a GPU
device back-end (i.e., when ``ENABLE_CUDA``, ``ENABLE_HIP``, or
``RAJA_ENABLE_SYCL`` is turned on in a RAJA build configuration):

.. list-table:: ``device_*`` alias coverage
   :widths: 34 14 14 14 24
   :header-rows: 1

   * - Alias family
     - CUDA
     - HIP
     - SYCL
     - Notes
   * - device_exec*
     - yes
     - yes
     - partial
     - Block-size templated exec aliases, for example
       ``device_exec<256>``. ``device_exec`` and ``device_exec_async`` exist
       for SYCL; the CUDA/HIP occupancy and reduction variants do not.
   * - device_atomic and device_atomic_explicit<host_policy>
     - yes
     - yes
     - yes
     - Back-end atomic policy aliases are available on all three back-ends, for
       example ``device_atomic_explicit<RAJA::seq_atomic>``.
   * - device_reduce
     - yes
     - yes
     - yes
     - Back-end default reduce policy alias.
   * - device_reduce_atomic and device_reduce_base<with_atomic>
     - yes
     - yes
     - no
     - CUDA/HIP expose tuning and base reduce aliases, for example
       ``device_reduce_base<true>``; SYCL currently only exposes
       ``device_reduce``.
   * - device_multi_reduce_atomic and
       device_multi_reduce_atomic_low_performance_low_overhead
     - yes
     - yes
     - no
     - CUDA/HIP expose multi_reduce aliases; SYCL does not currently
       provide a back-end-equivalent device_* mapping.
   * - device_launch_t
     - yes
     - yes
     - yes
     - Back-end launch policy alias, for example ``device_launch_t<false>``.
   * - device_global_size_{x,y,z}_{direct,direct_unchecked,loop}<N>
     - yes
     - yes
     - partial
     - Size-templated aliases, for example ``device_global_size_x_direct<64>``.
       SYCL currently exposes only the direct x/y/z aliases; the unchecked and
       loop variants are CUDA/HIP only.
   * - device_thread_{x,y,z}_{direct,loop}
     - yes
     - yes
     - yes
     - Single-dimension thread mapping.
   * - device_thread_size_{x,y,z}_{direct,direct_unchecked,loop}<N>
     - yes
     - yes
     - no
     - Size-templated aliases, for example ``device_thread_size_x_direct<128>``.
       These aliases are declared as compile-time errors under SYCL.
   * - device_block_{x,y,z}_{direct,loop}
     - yes
     - yes
     - yes
     - Single-dimension block mapping.
   * - device_block_size_{x,y,z}_{direct,direct_unchecked,loop}<N>
     - yes
     - yes
     - no
     - Size-templated aliases, for example ``device_block_size_x_direct<128>``.
       These aliases are declared as compile-time errors under SYCL.
   * - device_flatten_thread_size_*, device_flatten_block_size_*,
       device_flatten_global_size_*
     - yes
     - yes
     - no
     - Size-templated aliases, for example
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
   the implementation. Device aliases that have no SYCL equivalent are intentionally not defined
   under SYCL as usable policies. Attempting to use them will cause compile time failure
   so unsupported code paths are caught immediately.

See also the example code ``examples/device-policy-aliases.cpp``.

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
                                                                            set of resources. This minimally effects
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

.. _localarraypolicy-label:

----------------------------
Local Array Memory Policies
----------------------------

``RAJA::LocalArray`` types must use a memory policy indicating
where the memory for the local array will live. These policies are described
in :ref:`feat-local_array-label`.

The following memory policies are available to specify memory allocation
for ``RAJA::LocalArray`` objects:

  *  ``RAJA::cpu_tile_mem`` - Allocate CPU memory on the stack
  *  ``RAJA::cuda/hip_shared_mem`` - Allocate CUDA or HIP shared memory
  *  ``RAJA::cuda/hip_thread_mem`` - Allocate CUDA or HIP thread private memory


.. _loop_elements-kernelpol-label:

--------------------------------
RAJA Kernel Execution Policies
--------------------------------

RAJA kernel execution policy constructs form a simple domain specific language
for composing and transforming complex loops that relies
**solely on standard C++20 template support**.
RAJA kernel policies are constructed using a combination of *Statements* and
*Statement Lists*. A RAJA Statement is an action, such as execute a loop,
invoke a lambda, set a thread barrier, etc. A StatementList is an ordered list
of Statements that are composed in the order that they appear in the kernel
policy to construct a kernel. A Statement may contain an enclosed StatementList. Thus, a ``RAJA::KernelPolicy`` type is really just a StatementList.

The main Statement types provided by RAJA are ``RAJA::statement::For`` and
``RAJA::statement::Lambda``, that we discussed in
:ref:`loop_elements-kernel-label`.
A ``RAJA::statement::For<ArgID, ExecPolicy, Enclosed Statements>`` type
indicates a for-loop structure. The ``ArgID`` parameter is an integral constant
that identifies the position of the iteration space in the iteration space
tuple passed to the ``RAJA::kernel`` method to be used for the loop. The
``ExecPolicy`` is the RAJA execution policy to use on the loop, which is
similar to ``RAJA::forall`` usage. The ``EnclosedStatements`` type is a
nested template parameter that contains whatever is needed to execute the
kernel and which forms a valid StatementList. The
``RAJA::statement::Lambda<LambdaID>``
type invokes the lambda expression corresponding to its position 'LambdaID'
in the sequence of lambda expressions in the ``RAJA::kernel`` argument list.
For example, a simple sequential for-loop::

  for (int i = 0; i < N; ++i) {
    // loop body
  }

can be represented using the RAJA kernel interface as::

  using KERNEL_POLICY =
    RAJA::KernelPolicy<
      RAJA::statement::For<0, RAJA::seq_exec,
        RAJA::statement::Lambda<0>
      >
    >;

  RAJA::kernel<KERNEL_POLICY>(
    RAJA::make_tuple(range),
    [=](int i) {
      // loop body
    }
  );

.. note:: All ``RAJA::forall`` functionality can be done using the
          ``RAJA::kernel`` interface. We maintain the ``RAJA::forall``
          interface since it is less verbose and thus more convenient
          for users.

RAJA::kernel Statement Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The list below summarizes the current collection of statement types that
can be used with ``RAJA::kernel`` and ``RAJA::kernel_param``. More detailed
explanation along with examples of how they are used can be found in
the ``RAJA::kernel`` examples in :ref:`tutorial-label`.

.. note:: All of the statement types described below are in the namespace
          ``RAJA::statement``. For brevity, we omit the namespaces in
          the discussion in this section.

.. note::  ``RAJA::kernel_param`` functions similarly to ``RAJA::kernel``
           except that the second argument is a *tuple of parameters* used
           in a kernel for local arrays, thread local variables, tiling
           information, etc.

Several RAJA statements can be specialized with auxiliary types, which are
described in :ref:`auxilliarypolicy_label`.

The following list contains the most commonly used statement types.

* ``For< ArgId, ExecPolicy, EnclosedStatements >`` abstracts a for-loop associated with kernel iteration space at tuple index ``ArgId``, to be run with ``ExecPolicy`` execution policy, and containing the ``EnclosedStatements`` which are executed for each loop iteration.

* ``Lambda< LambdaId >`` invokes the lambda expression that appears at position 'LambdaId' in the sequence of lambda arguments. With this statement, the lambda expression must accept all arguments associated with the tuple of iteration space segments and tuple of parameters (if kernel_param is used).

* ``Lambda< LambdaId, Args...>`` extends the Lambda statement. The second template parameter indicates which arguments (e.g., which segment iteration variables) are passed to the lambda expression.

* ``Collapse< ExecPolicy, ArgList<...>, EnclosedStatements >`` collapses multiple perfectly nested loops specified by tuple iteration space indices in ``ArgList``, using the ``ExecPolicy`` execution policy, and places ``EnclosedStatements`` inside the collapsed loops which are executed for each iteration. **Note that this only works for CPU execution policies (e.g., sequential, OpenMP).** It may be available for CUDA in the future if such use cases arise.

There is one statement specific to OpenMP kernels.

* ``OmpSyncThreads`` applies the OpenMP ``#pragma omp barrier`` directive.

Statement types that launch CUDA or HIP GPU kernels are listed next. They work
similarly for each back-end and their names are distinguished by the prefix
``Cuda`` or ``Hip``. For example, ``CudaKernel`` or ``HipKernel``.

* ``Cuda/HipKernel< EnclosedStatements>`` launches ``EnclosedStatements`` as a GPU kernel; e.g., a loop nest where the iteration spaces of each loop level are associated with threads and/or thread blocks as described by the execution policies applied to them. This kernel launch is synchronous.

* ``Cuda/HipKernelAsync< EnclosedStatements>`` asynchronous version of Cuda/HipKernel.

* ``Cuda/HipKernelFixed<num_threads, EnclosedStatements>`` similar to Cuda/HipKernel but enables a fixed number of threads (specified by num_threads). This kernel launch is synchronous.

* ``Cuda/HipKernelFixedAsync<num_threads, EnclosedStatements>`` asynchronous version of Cuda/HipKernelFixed.

* ``CudaKernelFixedSM<num_threads, min_blocks_per_sm, EnclosedStatements>`` similar to CudaKernelFixed but enables a minimum number of blocks per sm (specified by min_blocks_per_sm), this can help increase occupancy. This kernel launch is synchronous.  **Note: there is no HIP variant of this statement.**

* ``CudaKernelFixedSMAsync<num_threads, min_blocks_per_sm, EnclosedStatements>`` asynchronous version of CudaKernelFixedSM. **Note: there is no HIP variant of this statement.**

* ``Cuda/HipKernelOcc<EnclosedStatements>`` similar to CudaKernel but uses the CUDA occupancy calculator to determine the optimal number of threads/blocks. Statement is intended for use with RAJA::cuda/hip_block_{xyz}_loop policies. This kernel launch is synchronous.

* ``Cuda/HipKernelOccAsync<EnclosedStatements>`` asynchronous version of Cuda/HipKernelOcc.

* ``Cuda/HipKernelExp<num_blocks, num_threads, EnclosedStatements>`` similar to CudaKernelOcc but with the flexibility to fix the number of threads and/or blocks and let the CUDA occupancy calculator determine the unspecified values. This kernel launch is synchronous.

* ``Cuda/HipKernelExpAsync<num_blocks, num_threads, EnclosedStatements>`` asynchronous version of Cuda/HipKernelExp.

* ``Cuda/HipSyncThreads`` invokes CUDA or HIP ``__syncthreads()`` barrier.

* ``Cuda/HipSyncWarp`` invokes CUDA ``__syncwarp()`` barrier. Warp sync is not supported in HIP, so the HIP variant is a no-op.

Statement types that launch SYCL kernels are listed next.

* ``SyclKernel<EnclosedStatements>`` launches ``EnclosedStatements`` as a SYCL kernel.  This kernel launch is synchronous.

* ``SyclKernelAsync<EnclosedStatements>`` asynchronous version of SyclKernel.

RAJA provides statements to define loop tiling which can improve performance;
e.g., by allowing CPU cache blocking or use of GPU shared memory.

* ``Tile< ArgId, TilePolicy, ExecPolicy, EnclosedStatements >`` abstracts an outer tiling loop containing an inner for-loop over each tile. The ``ArgId`` indicates which entry in the iteration space tuple to which the tiling loop applies and the ``TilePolicy`` specifies the tiling pattern to use, including its dimension. The ``ExecPolicy`` and ``EnclosedStatements`` are similar to what they represent in a ``statement::For`` type.

* ``TileTCount< ArgId, ParamId, TilePolicy, ExecPolicy, EnclosedStatements >`` abstracts an outer tiling loop containing an inner for-loop over each tile, **where it is necessary to obtain the tile number in each tile**. The ``ArgId`` indicates which entry in the iteration space tuple to which the loop applies and the ``ParamId`` indicates the position of the tile number in the parameter tuple. The ``TilePolicy`` specifies the tiling pattern to use, including its dimension. The ``ExecPolicy`` and ``EnclosedStatements`` are similar to what they represent in a ``statement::For`` type.

* ``ForICount< ArgId, ParamId, ExecPolicy, EnclosedStatements >`` abstracts an inner for-loop within an outer tiling loop **where it is necessary to obtain the local iteration index in each tile**. The ``ArgId`` indicates which entry in the iteration space tuple to which the loop applies and the ``ParamId`` indicates the position of the tile index parameter in the parameter tuple. The ``ExecPolicy`` and ``EnclosedStatements`` are similar to what they represent in a ``statement::For`` type.

It is often advantageous to use local arrays for data accessed in tiled loops.
RAJA provides a statement for allocating data in a :ref:`feat-local_array-label`
object according to a memory policy. See :ref:`localarraypolicy-label` for more information about such policies.

* ``InitLocalMem< MemPolicy, ParamList<...>, EnclosedStatements >`` allocates memory for a ``RAJA::LocalArray`` object used in kernel. The ``ParamList`` entries indicate which local array objects in a tuple will be initialized. The ``EnclosedStatements`` contain the code in which the local array will be accessed; e.g., initialization operations.

RAJA provides some statement types that apply in specific kernel scenarios.

* ``Reduce< ReducePolicy, Operator, ParamId, EnclosedStatements >`` reduces a value across threads in a multithreaded code region to a single thread. The ``ReducePolicy`` is similar to what it represents for RAJA reduction types. ``ParamId`` specifies the position of the reduction value in the parameter tuple passed to the ``RAJA::kernel_param`` method. ``Operator`` is the binary operator used in the reduction; typically, this will be one of the operators that can be used with RAJA scans (see :ref:`feat-scanops-label`). After the reduction is complete, the ``EnclosedStatements`` execute on the thread that received the final reduced value.

* ``If< Conditional >`` chooses which portions of a policy to run based on run-time evaluation of conditional statement; e.g., true or false, equal to some value, etc.

* ``Hyperplane< ArgId, HpExecPolicy, ArgList<...>, ExecPolicy, EnclosedStatements >`` provides a hyperplane (or wavefront) iteration pattern over multiple indices. A hyperplane is a set of multi-dimensional index values: i0, i1, ... such that h = i0 + i1 + ... for a given h. Here, ``ArgId`` is the position of the loop argument we will iterate on (defines the order of hyperplanes), ``HpExecPolicy`` is the execution policy used to iterate over the iteration space specified by ArgId (often sequential), ``ArgList`` is a list of other indices that along with ArgId define a hyperplane, and ``ExecPolicy`` is the execution policy that applies to the loops in ``ArgList``. Then, for each iteration, everything in the ``EnclosedStatements`` is executed.


.. _auxilliarypolicy_label:

--------------------------------
Auxilliary Types
--------------------------------

The following list summarizes auxiliary types used in the above statements. These
types live in the ``RAJA`` namespace.

  * ``tile_fixed<TileSize>`` tile policy argument to a ``Tile`` or ``TileTCount`` statement; partitions loop iterations into tiles of a fixed size specified by ``TileSize``. This statement type can be used as the ``TilePolicy`` template parameter in the ``Tile`` statements above.

  * ``tile_dynamic<ParamIdx>`` TilePolicy argument to a Tile or TileTCount statement; partitions loop iterations into tiles of a size specified by a ``TileSize{}`` positional parameter argument. This statement type can be used as the ``TilePolicy`` template parameter in the ``Tile`` statements above.

  * ``Segs<...>`` argument to a Lambda statement; used to specify which segments in a tuple will be used as lambda arguments.

  * ``Offsets<...>`` argument to a Lambda statement; used to specify which segment offsets in a tuple will be used as lambda arguments.

  * ``Params<...>`` argument to a Lambda statement; used to specify which params in a tuple will be used as lambda arguments.

  * ``ValuesT<T, ...>`` argument to a Lambda statement; used to specify compile time constants, of type T, that will be used as lambda arguments.

Examples that show how to use a variety of these statement types can be found
in :ref:`loop_elements-kernel-label`.
