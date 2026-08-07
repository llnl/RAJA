.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _feat-policies-kernel-reference-label:

========================================
RAJA::kernel and Local Array Policies
========================================

This page contains policy reference material for ``RAJA::LocalArray`` memory
policies and ``RAJA::kernel`` statement policies.

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
RAJA::kernel Execution Policies
--------------------------------

What is a KernelPolicy?
~~~~~~~~~~~~~~~~~~~~~~~~

The ``RAJA::kernel`` interface is designed to support portability for complex
nested loop kernels. It provides compile-time porting of kernels to execute
with different RAJA back-ends and transformations such as loop nest reordering.

The constructs used in ``RAJA::kernel`` execution policies form a simple
domain-specific language that composes and transforms complex loops and relies
**solely on standard C++20 template support**.
RAJA kernel policies are constructed using a combination of *Statements* and
*Statement Lists*. A RAJA Statement is an action, such as executing a loop,
invoking a lambda, or setting a thread barrier. A StatementList is an ordered list
of Statements that are composed in the order that they appear in the kernel
policy to construct the execution behavior of a kernel. A Statement may contain
a StatementList. Thus, a ``RAJA::KernelPolicy`` type is really just a StatementList.

The main Statement types provided by RAJA are ``RAJA::statement::For`` and
``RAJA::statement::Lambda``, as discussed in
:ref:`loop_elements-kernel-label`. A ``For`` statement describes a loop over
one entry in the iteration-space tuple passed to ``RAJA::kernel``. A
``Lambda`` statement invokes a lambda expressions passed to ``RAJA::kernel``.

``RAJA::statement``: Basic For and Lambda Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have seen how a simple sequential for-loop::

  for (int i = 0; i < N; ++i) {
    // loop body
  }

can be written using ``RAJA::forall`` as::

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, N),
    [=] (RAJA::Index_type i) {
      // loop body
    });

The same single-loop kernel can be represented using the ``RAJA::kernel``
interface as::

  using KERNEL_POLICY =
    RAJA::KernelPolicy<
      RAJA::statement::For<0, RAJA::seq_exec,
        RAJA::statement::Lambda<0>
      >
    >;

  RAJA::kernel<KERNEL_POLICY>(
    RAJA::make_tuple(RAJA::RangeSegment(0, N)),
    [=](int i) {
      // loop body
    }
  );

Clearly, the ``RAJA::kernel`` implementation is more verbose. However, keep in mind
that ``RAJA::kernel`` is designed to be used for more complex, nested loop kernels.
A key difference between ``RAJA::kernel`` and ``RAJA::forall`` is that ``RAJA::kernel``
takes a *tuple* of segments, each one representing the iteration space of a loop in
a loop nest whereas ``RAJA::forall`` takes a single segment because it can only execute
a single loop kernel. In addition, ``RAJA::kernel`` accepts one or more lambda expression
arguments depending on how one choose to break apart and represent the work done in the
kernel.

.. note:: All ``RAJA::forall`` functionality can be done using the
          ``RAJA::kernel`` interface. We maintain the ``RAJA::forall``
          interface since it is less verbose and thus more convenient
          for users working with simple single-loop kernels.

For more realistic and advanced examples of ``RAJA::kernel``, please see :ref:`tut-kernelexecpols-label`

How to Read Statement Type Signatures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many statement types use the same template argument names. Understanding these
names makes the statement reference below easier to scan.

.. list-table::
   :widths: 24 76
   :header-rows: 1

   * - Template argument
     - Meaning
   * - ``ArgId``
     - Position of an iteration segment in the tuple passed to
       ``RAJA::kernel`` or ``RAJA::kernel_param``.
   * - ``LambdaId``
     - Position of a lambda expression in the sequence of lambda arguments
       passed to ``RAJA::kernel`` or ``RAJA::kernel_param``.
   * - ``ExecPolicy``
     - Execution policy used by a loop, collapse, tiling, or reduction
       statement.
   * - ``EnclosedStatements``
     - Nested statement list executed inside the current statement.
   * - ``ArgList``
     - Compile-time list of iteration segment indices used by statements such
       as ``Collapse`` and ``Hyperplane``.
   * - ``ParamId``
     - Position of an entry in the parameter tuple passed to
       ``RAJA::kernel_param``.
   * - ``TilePolicy``
     - Tiling policy, such as ``tile_fixed`` or ``tile_dynamic``, that controls
       how a loop iteration space is partitioned into tiles.
   * - ``Operator``
     - Binary operation used by a reduction statement.

.. note:: All of the statement types described below are in the namespace
          ``RAJA::statement``. For brevity, we omit the ``RAJA`` namespace in
          the discussion in this section.

.. important:: ``RAJA::kernel_param`` functions similarly to ``RAJA::kernel``
                except that the second argument is a *tuple of parameters* used
                in a kernel for local arrays, thread local variables, tiling
                information, etc. that are used in one or more lambda expressions.

RAJA::kernel Statement Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The list below summarizes the current collection of statement types that
can be used with ``RAJA::kernel`` and ``RAJA::kernel_param``. More detailed
explanation along with examples of how they are used can be found in
the ``RAJA::kernel`` examples in :ref:`tutorial-label`.

Several RAJA statements can be specialized with auxiliary types, which are
described in :ref:`auxiliarypolicy_label`.

Core Loop and Lambda Statements
"""""""""""""""""""""""""""""""

These statements describe the loop structure of a kernel policy and the lambda
expressions that perform the work inside that structure.

.. list-table::
   :widths: 45 55
   :header-rows: 1

   * - Statement type
     - Description
   * - For< ArgId, ExecPolicy, EnclosedStatements >
     - Abstracts a for-loop associated with kernel iteration space at tuple
       index ``ArgId``, to be run with ``ExecPolicy`` execution policy, and
       containing the ``EnclosedStatements`` which are executed for each loop
       iteration.
   * - Lambda< LambdaId >
     - Invokes the lambda expression that appears at position ``LambdaId`` in
       the sequence of lambda arguments. With this statement, the lambda
       expression must accept all arguments associated with the tuple of
       iteration space segments and tuple of parameters, if ``kernel_param`` is
       used.
   * - Lambda< LambdaId, Args...>
     - Extends the ``Lambda`` statement. The second template parameter
       indicates which arguments, such as which segment iteration variables,
       are passed to the lambda expression.
   * - Collapse< ExecPolicy, ArgList<...>, EnclosedStatements >
     - Collapses multiple perfectly nested loops specified by tuple iteration
       space indices in ``ArgList``, using the ``ExecPolicy`` execution policy,
       and places ``EnclosedStatements`` inside the collapsed loops which are
       executed for each iteration. **Note that this only works for CPU
       execution policies (e.g., sequential, OpenMP).** It may be available for
       CUDA in the future if such use cases arise.

Backend Kernel Launch and Synchronization Statements
""""""""""""""""""""""""""""""""""""""""""""""""""""

OpenMP, CUDA/HIP, and SYCL kernel policies use backend-specific statements to
create parallel regions, launch device kernels, and synchronize work. CUDA and
HIP statements work similarly for each back-end and their names are
distinguished by the prefix ``Cuda`` or ``Hip``. For example, ``CudaKernel`` or
``HipKernel``.

.. list-table::
   :widths: 45 55
   :header-rows: 1

   * - Statement type
     - Description
   * - OmpSyncThreads
     - Applies the OpenMP ``#pragma omp barrier`` directive.
   * - Cuda/HipKernel< EnclosedStatements >
     - Launches ``EnclosedStatements`` as a GPU kernel; e.g., a loop nest where
       the iteration spaces of each loop level are associated with threads
       and/or thread blocks as described by the execution policies applied to
       them. This kernel launch is synchronous.
   * - Cuda/HipKernelAsync< EnclosedStatements >
     - Asynchronous version of ``Cuda/HipKernel``.
   * - Cuda/HipKernelFixed< num_threads, EnclosedStatements >
     - Similar to ``Cuda/HipKernel`` but enables a fixed number of threads
       specified by ``num_threads``. This kernel launch is synchronous.
   * - Cuda/HipKernelFixedAsync< num_threads, EnclosedStatements >
     - Asynchronous version of ``Cuda/HipKernelFixed``.
   * - CudaKernelFixedSM< num_threads, min_blocks_per_sm, EnclosedStatements >
     - Similar to ``CudaKernelFixed`` but enables a minimum number of blocks per
       SM, specified by ``min_blocks_per_sm``, which can help increase
       occupancy. This kernel launch is synchronous. **Note: there is no HIP
       variant of this statement.**
   * - CudaKernelFixedSMAsync< num_threads, min_blocks_per_sm, EnclosedStatements >
     - Asynchronous version of ``CudaKernelFixedSM``. **Note: there is no HIP
       variant of this statement.**
   * - Cuda/HipKernelOcc< EnclosedStatements >
     - Similar to ``Cuda/HipKernel`` but uses the CUDA or HIP occupancy
       calculator to determine the optimal number of threads/blocks. This
       statement is intended for use with ``RAJA::cuda/hip_block_{xyz}_loop``
       policies. This kernel launch is synchronous.
   * - Cuda/HipKernelOccAsync< EnclosedStatements >
     - Asynchronous version of ``Cuda/HipKernelOcc``.
   * - Cuda/HipKernelExp< num_blocks, num_threads, EnclosedStatements >
     - Similar to ``Cuda/HipKernelOcc`` but with the flexibility to fix the
       number of threads and/or blocks and let the CUDA or HIP occupancy
       calculator determine the unspecified values. This kernel launch is
       synchronous.
   * - Cuda/HipKernelExpAsync< num_blocks, num_threads, EnclosedStatements >
     - Asynchronous version of ``Cuda/HipKernelExp``.
   * - Cuda/HipSyncThreads
     - Invokes CUDA or HIP ``__syncthreads()`` barrier.
   * - Cuda/HipSyncWarp
     - Invokes CUDA ``__syncwarp()`` barrier. Warp sync is not supported in
       HIP, so the HIP variant is a no-op.
   * - SyclKernel< EnclosedStatements >
     - Launches ``EnclosedStatements`` as a SYCL kernel. This kernel launch is
       synchronous.
   * - SyclKernelAsync< EnclosedStatements >
     - Asynchronous version of ``SyclKernel``.

Tiling and Local Memory Statements
""""""""""""""""""""""""""""""""""

RAJA provides statements to define loop tiling, which can improve performance;
e.g., by allowing CPU cache blocking or use of GPU shared memory. It also
provides a statement for allocating data in a :ref:`feat-local_array-label`
object according to a memory policy. See :ref:`localarraypolicy-label` for more
information about such policies.

.. list-table::
   :widths: 45 55
   :header-rows: 1

   * - Statement type
     - Description
   * - Tile< ArgId, TilePolicy, ExecPolicy, EnclosedStatements >
     - Abstracts an outer tiling loop containing an inner for-loop over each
       tile. The ``ArgId`` indicates which entry in the iteration space tuple
       to which the tiling loop applies and the ``TilePolicy`` specifies the
       tiling pattern to use, including its dimension. The ``ExecPolicy`` and
       ``EnclosedStatements`` are similar to what they represent in a
       ``statement::For`` type.
   * - TileTCount< ArgId, ParamId, TilePolicy, ExecPolicy, EnclosedStatements >
     - Abstracts an outer tiling loop containing an inner for-loop over each
       tile, **where it is necessary to obtain the tile number in each tile**.
       The ``ArgId`` indicates which entry in the iteration space tuple to which
       the loop applies and the ``ParamId`` indicates the position of the tile
       number in the parameter tuple. The ``TilePolicy`` specifies the tiling
       pattern to use, including its dimension. The ``ExecPolicy`` and
       ``EnclosedStatements`` are similar to what they represent in a
       ``statement::For`` type.
   * - ForICount< ArgId, ParamId, ExecPolicy, EnclosedStatements >
     - Abstracts an inner for-loop within an outer tiling loop, **where it is
       necessary to obtain the local iteration index in each tile**. The
       ``ArgId`` indicates which entry in the iteration space tuple to which the
       loop applies and the ``ParamId`` indicates the position of the tile index
       parameter in the parameter tuple. The ``ExecPolicy`` and
       ``EnclosedStatements`` are similar to what they represent in a
       ``statement::For`` type.
   * - InitLocalMem< MemPolicy, ParamList<...>, EnclosedStatements >
     - Allocates memory for a ``RAJA::LocalArray`` object used in kernel. The
       ``ParamList`` entries indicate which local array objects in a tuple will
       be initialized. The ``EnclosedStatements`` contain the code in which the
       local array will be accessed; e.g., initialization operations.

Specialized Statements
""""""""""""""""""""""

RAJA provides some statement types that apply in specific kernel scenarios, such
as reductions, conditional execution, and hyperplane iteration.

.. list-table::
   :widths: 45 55
   :header-rows: 1

   * - Statement type
     - Description
   * - Reduce< ReducePolicy, Operator, ParamId, EnclosedStatements >
     - Reduces a value across threads in a multithreaded code region to a
       single thread. The ``ReducePolicy`` is similar to what it represents for
       RAJA reduction types. ``ParamId`` specifies the position of the
       reduction value in the parameter tuple passed to the
       ``RAJA::kernel_param`` method. ``Operator`` is the binary operator used
       in the reduction; typically, this will be one of the operators that can
       be used with RAJA scans (see :ref:`feat-scanops-label`). After the
       reduction is complete, the ``EnclosedStatements`` execute on the thread
       that received the final reduced value.
   * - If< Conditional >
     - Chooses which portions of a policy to run based on run-time evaluation
       of a conditional statement; e.g., true or false, equal to some value,
       etc.
   * - Hyperplane< ArgId, HpExecPolicy, ArgList<...>, ExecPolicy, EnclosedStatements >
     - Provides a hyperplane (or wavefront) iteration pattern over multiple
       indices. A hyperplane is a set of multi-dimensional index values:
       ``i0``, ``i1``, ... such that ``h = i0 + i1 + ...`` for a given ``h``.
       Here, ``ArgId`` is the position of the loop argument we will iterate on
       (defines the order of hyperplanes), ``HpExecPolicy`` is the execution
       policy used to iterate over the iteration space specified by ``ArgId``
       (often sequential), ``ArgList`` is a list of other indices that along
       with ``ArgId`` define a hyperplane, and ``ExecPolicy`` is the execution
       policy that applies to the loops in ``ArgList``. Then, for each
       iteration, everything in the ``EnclosedStatements`` is executed.


.. _auxiliarypolicy_label:

--------------------------------
Auxiliary Types
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
