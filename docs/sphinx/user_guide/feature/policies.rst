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
be redefined to work in new ways.

.. note:: * All RAJA policies are in the namespace ``RAJA``.
          * All RAJA policies have a prefix indicating the back-end
            implementation that they use; e.g., ``omp_`` for OpenMP, ``cuda_``
            for CUDA, etc.

This page includes mostly introductory and mildly pedagogical material about RAJA
execution policies. The sections include: policy basics/categories, policy naming conventions,
advice on policies to try first based on RAJA interface method and execution back-end,
basic examples, and a glossary of terms. 

Those who are familar with RAJA policies may want to skip the introductory material
and jump right to the details policy reference sections:

.. toctree::
   :maxdepth: 1

   policies/execution
   policies/reductions_atomics
   policies/kernel

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

For a first implementation, we recommend starting with one of the policies below
and then move to more specialized policies when specific scheduling, mapping,
synchronization, or performance behavior is needed.

Common Goals Quick Reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this table to find a reasonable starting policy for common use cases.
The reference links point to detailed policy tables in the focused reference
pages linked below.

.. list-table::
   :widths: 34 32 34
   :header-rows: 1

   * - Goal
     - Starting policy
     - Reference
   * - Sequential CPU loop
     - ``seq_exec``
     - :ref:`Sequential CPU Policies <feat-policies-cpu-label>`
   * - CPU loop with SIMD compiler hints
     - ``simd_exec``
     - :ref:`Sequential CPU Policies <feat-policies-cpu-label>`
   * - OpenMP parallel CPU loop
     - ``omp_parallel_for_exec``
     - :ref:`OpenMP CPU Policies <feat-policies-openmp-cpu-label>`
   * - Multiple loops in an OpenMP parallel region
     - ``RAJA::region`` with ``omp_for_exec`` or another OpenMP inner policy
     - :ref:`Parallel Region Policies <parallelregionpolicy-label>`
   * - CUDA ``forall`` loop
     - ``cuda_exec<BLOCK_SIZE>``
     - :ref:`GPU Policies for CUDA and HIP <feat-policies-gpu-label>`
   * - HIP ``forall`` loop
     - ``hip_exec<BLOCK_SIZE>``
     - :ref:`GPU Policies for CUDA and HIP <feat-policies-gpu-label>`
   * - SYCL ``forall`` loop
     - ``sycl_exec<WORK_GROUP_SIZE>``
     - :ref:`GPU Policies for SYCL <feat-policies-sycl-label>`
   * - Portable GPU ``forall`` loop (works for CUDA, HIP, SYCL)
     - ``device_exec<BLOCK_SIZE>``
     - :ref:`Device policy aliases <feat-policies-gpu-aliases-label>`
   * - OpenMP target offload loop
     - ``omp_target_parallel_for_exec<#>``
     - :ref:`OpenMP Target Offload Policies <feat-policies-omp-target-label>`
   * - Nested loop kernel
     - ``RAJA::KernelPolicy`` with ``For`` and ``Lambda`` statements
     - :ref:`RAJA::kernel Execution Policies <loop_elements-kernelpol-label>`
   * - Nested GPU loop mapping
     - CUDA/HIP/SYCL thread, block, group, or global mapping policies
     - :ref:`GPU Policies for CUDA and HIP <feat-policies-gpu-label>` and
       :ref:`GPU Policies for SYCL <feat-policies-sycl-label>`
   * - Reduction
     - Match the reduction policy to the loop back-end, such as
       ``seq_reduce``, ``omp_reduce``, ``cuda_reduce``, ``hip_reduce``, or
       ``sycl_reduce``
     - :ref:`Reduction Policies <reducepolicy-label>`
   * - Multi-reduction
     - Match runtime determined number of reduction operations in a kernel to kernel
       back-end, such as ``seq_multi_reduce``, ``omp_multi_reduce``, ``cuda_multi_reduce``,
       ``hip_multi_reduce``, or ``sycl_multireduce``
     - :ref:`Multi Reduction Examples <feat-multi-reductions-label>`
   * - Atomic update
     - Match the atomic policy to the loop back-end, such as ``omp_atomic`` or
       ``cuda_atomic``
     - :ref:`Atomic Policies <atomicpolicy-label>`

The tutorial sections provide worked examples that are easier to follow than
the full reference tables on this page. Please see :ref:`tutorial-examples-label`.

-----------------------------------------------------
How to Read Policy Names
-----------------------------------------------------

RAJA policy names are intentionally descriptive. Most names combine a back-end
prefix with words that describe where work runs and how iterations are mapped.
Understanding these pieces makes the reference tables easier to scan. See
:ref:`policy-glossary-label` for definitions of terms that appear throughout
the policy reference tables.

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
chunk size of 4. Some template parameters are optional; the policy reference
pages call out those cases explicitly.

-----------------------------------------------------
Choosing a Policy
-----------------------------------------------------

The detailed reference pages are linked below. When choosing a policy for new
code, start from the RAJA interface you are using, then choose the execution
back-end, then add specialized policies only when a kernel needs them.

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
       level. Note that there are back-end specific policies such as ``RAJA::CudaKernel``
       and ``RAJA::HipKernel`` for controlling how loop-level iterations are mapped
       to processor resources. See :ref:`loop_elements-kernelpol-label`.
   * - ``RAJA::launch``
     - A launch policy such as ``seq_launch_t``, ``omp_launch_t``, or
       ``cuda_launch_t`` to create an execution region, plus loop policies
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
     - Use ``simd_exec`` only for loops that are data parallel; e.g.,
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
     - Use ``device_*`` aliases when code can follow the enabled CUDA, HIP,
       or SYCL back-end without preprocessor conditionals.
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
     - Use static, dynamic, guided, or runtime OpenMP policy variants. Nowait
       variants are also available for static scheduling.
   * - Multiple loops in one OpenMP parallel region
     - Use ``RAJA::region`` with OpenMP inner policies such as
       ``omp_for_exec`` or ``omp_for_static_exec``.
   * - Nested loop reordering or collapse
     - Use ``RAJA::kernel`` statements such as ``RAJA::statement::For`` and
       ``RAJA::statement::Collapse``.
   * - GPU tiled loop mapping
     - Use GPU thread, block, or global mapping policies. Typically, users
       will want to use a global direct policy for simple implicit tiling
       or block direct unchecked and thread direct policies for explicit tiling.
       The ``*_loop`` policies, while potentially less optimal, are most applicable
       when the iteration space is not rectangular or when more work is mapped
       to a tile than fits in a single pass.
   * - GPU reduction inside ``RAJA::forall``
     - Use the reduction-aware CUDA/HIP execution policy variants where
       appropriate, and match the reducer policy to the loop back-end.
   * - Atomic update
     - Match the atomic policy to the loop back-end, or use ``auto_atomic``
       where supported.

-----------------------------------------------------
Basic Policy Examples
-----------------------------------------------------

The examples below show how to use the basic ``RAJA::forall`` construct to execute
a simple loop kernel with various execution policies. In each case, the lambda
expression body describes the work done for each kernel iterate and the
policy selects where and how the loop runs. For brevity, these snippets omit
allocation, data movement, and backend-specific build guards.

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

CUDA and HIP execution policies take a block-size template parameter. Device
lambdas must be marked with the appropriate device annotation. Here, the
``RAJA_DEVICE`` macro is shown:

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

.. note:: Reduction objects generally support sequential reduction in addition to the 
          reduction for the parallel reduction back-end. Please see the reduction
          policy table in :ref:`reducepolicy-label`.

-----------------------------------------------------
Detailed Policy Reference
-----------------------------------------------------

The detailed reference material is split into focused pages:

.. toctree::
   :maxdepth: 1

   policies/execution
   policies/reductions_atomics
   policies/kernel

.. _policy-glossary-label:

--------------------------------
Glossary of Policy Terms
--------------------------------

This glossary defines terms that appear throughout the policy tables. It is not
intended to be a complete parallel programming glossary; the definitions focus
on how the terms are used in RAJA policy names and descriptions.

.. glossary::

   execution policy
     A C++ type that tells RAJA how to execute a loop or related operation.
     Loop execution policies are used with ``RAJA::forall`` and with
     individual loop levels in ``RAJA::kernel`` and ``RAJA::launch``.

   launch policy
     A policy that creates an execution environment for ``RAJA::launch``. Loop
     policies used inside the launch body then describe how individual loops
     map onto that environment.

   back-end
     The programming model or execution target used to run work, such as
     sequential CPU execution, OpenMP, CUDA, HIP, SYCL, or OpenMP target.

   forall policy
     A loop execution policy used directly with ``RAJA::forall`` to execute a
     single loop kernel.

   kernel policy
     A ``RAJA::KernelPolicy`` type used with ``RAJA::kernel``. A kernel policy
     is built from statement types that describe loop nesting, loop execution
     policies, lambda invocation, and other kernel behavior.

   mapping policy
     A policy that maps a loop level to execution resources such as CPU
     threads, GPU threads, GPU blocks, SYCL work-items, or SYCL work-groups.

   direct mapping
     A mapping in which each execution resource maps directly to one loop
     iterate, with bounds masking for out-of-range resources.

   direct unchecked mapping
     A direct mapping without bounds checks. Use unchecked policies only when
     the execution shape exactly matches the iteration space required by the
     policy.

   loop mapping
     A mapping in which execution resources cover the iteration space using a
     strided loop. Loop mappings are usually the most forgiving GPU mapping
     choice when the iteration space may be larger than the selected execution
     shape.

   strided loop
     A loop pattern where one execution resource handles multiple loop iterates
     separated by a fixed stride, often the number of threads,
     blocks, work-items, or work-groups.

   thread
     In CUDA/HIP policy names, a GPU thread in a thread block. In more general
     descriptions, this may also refer to a CPU thread when the context is
     OpenMP.

   block
     In CUDA/HIP policy names, a GPU thread block. A block contains one or more
     GPU threads and can support block-local synchronization.

   warp
     A CUDA/HIP group of GPU threads that execute together at warp level. RAJA
     provides specialized warp mapping and warp reduction policies for kernels
     that need this level of control.

   global thread
     A unique GPU thread index formed from block and thread indices. Global
     thread mapping policies are often a good fit for simple data-parallel
     loops.

   work-item
     In SYCL policy names, an individual unit of work within a work-group. It
     is roughly analogous to a CUDA/HIP thread.

   work-group
     In SYCL policy names, a group of work-items that execute together. It is
     roughly analogous to a CUDA/HIP thread block.

   occupancy
     A measure of how much GPU execution capacity a kernel can use. RAJA
     provides CUDA/HIP occupancy policies that choose launch parameters using
     occupancy information.

   concretizer
     A helper type used by some CUDA/HIP occupancy policies to choose launch
     parameters that were not specified directly in the policy.

   reduction policy
     A policy used by RAJA reduction objects or reduction statements to combine
     values across loop iterates or threads.

   atomic policy
     A policy used by RAJA atomic operations to select the atomic implementation
     appropriate for the active loop back-end.

   device alias
     A ``device_*`` policy alias that resolves to the active GPU back-end when
     RAJA is built with CUDA, HIP, or SYCL support.
