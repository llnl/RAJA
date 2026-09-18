.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _cook-book-fornest-label:

===========================
Cooking with RAJA::fornest
===========================

``RAJA::fornest`` runs a logical multi-dimensional loop nest from one loop body while
letting a policy choose how the logical dimensions are traversed and mapped
between loop levels. The body is written in terms of logical indices such as
``(i,j)`` or ``(i,j,k)``; the policy can keep those dimensions as nested loops,
collapse them to a 1-D iteration space (and reconstruct logical indices), map
them onto device hierarchy, or tile them.
The loop body is treated as a single kernel entity: one callable is invoked for
each logical point, regardless of whether the policy results in nested loops, a
collapsed 1-D traversal, or a device launch.

.. note::
   ``RAJA::fornest`` is intended as a general loop-nest abstraction; the current
   implementation supports only 2- or 3-level loop nests.

The interface supports several policy families:

 * ``RAJA::fornest_collapsed_policy<ExecPolicy, LayoutTag>`` maps the product
   of the logical dimensions to a 1-D iteration space. The ``ExecPolicy`` is a
   regular forall-style policy such as ``RAJA::seq_exec`` or
   ``RAJA::device_exec<256>``. RAJA performs the linear-to-logical index
   reconstruction internally. On host OpenMP builds, ``RAJA::fornest_omp_collapse_policy<...>``
   is a related collapsed-style option that uses OpenMP collapse across the
   logical dimensions (rather than flattening to a 1-D index and reconstructing
   indices).
 * ``RAJA::fornest_mapping_policy<ExecPolicy, LoopPolicies...>`` maps the logical
   dimensions directly using one loop mapping tag per dimension. For mapping
   policies, the ``ExecPolicy`` selects the launch backend (typically
   ``RAJA::seq_launch_t`` on host and ``RAJA::device_launch_t<false>`` on GPU),
   and RAJA derives ``LaunchParams(Teams, Threads)`` from the mapping tags and
   segment extents. Device builds can use ``RAJA::device_*`` mapping tags; on
   CUDA/HIP these map to CUDA/HIP global, block, and thread indices, and on SYCL
   they map to the corresponding work-group and work-item indices where
   supported.
 * ``RAJA::fornest_tiling_policy<...>`` adds per-dimension tiling on top of a
   loop nest. Tile sizes may be fixed, runtime-provided, or chosen by RAJA.
 * ``RAJA::dynamic_fornest<camp::list<...>>(pol, ...)`` selects a policy from a
   list at runtime.

This makes it possible to put multiple implementations behind one abstraction
and choose the policy from problem size, backend, or measurements while
preserving one logical loop body.

----------------
Policy Tradeoffs
----------------

Many application kernels are logically multi-dimensional but have historically run on
a 1-D iteration space, with division and modulo operations used to recover the
logical indices. That approach can expose more parallelism when one logical
dimension is small.

For example, a ``cells x components`` kernel with only a few components may not
fit a fixed multi-dimensional thread-block shape well. An explicit mapping can
leave many threads inactive in each block when one logical dimension is small. The
collapsed mapping instead runs over ``cells * components`` as a 1-D space, which
can produce fuller blocks. The tradeoff is the extra index reconstruction
arithmetic. The explicit mapping can still win when the dimensions fit the block
shape, when the body is very small and index reconstruction dominates, or when
direct multi-dimensional mapping improves memory access or scheduling.

---------------
Mapping Policies
---------------

The minimal example source defines backend-specific policy aliases. CUDA/HIP can
use unsized global device mapping tags for explicit mapping. CUDA/HIP/SYCL can
use supported ``RAJA::device_*`` aliases for block/work-group and
thread/work-item mappings:

.. literalinclude:: ../../../../examples/fornest-basic.cpp
   :start-after: // _fornest_policy_aliases_start
   :end-before: // _fornest_policy_aliases_end
   :language: C++

Direct (Non-collapsed) Mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both “standard nested loops” and “explicit mapping” are *direct* mappings: the
logical loop dimensions are preserved (not flattened) and the body is invoked at
each logical point ``(i,j)`` / ``(i,j,k)``.

There are two common ways to express a direct mapping:

- **Nested loops**: an alias such as ``RAJA::fornest_basic_seq_2d`` traverses the
  logical dimensions as a nest of loops. On GPU builds this uses backend-aware
  mapping under the hood.
- **Explicit mapping**: a ``RAJA::fornest_mapping_policy`` executes via
  ``RAJA::launch`` and uses one loop mapping tag per dimension (e.g.,
  ``device_global_*``, ``device_block_*``, ``device_thread_*``). This makes the
  device hierarchy mapping explicit and gives you direct control via mapping
  tags.

Collapsed
~~~~~~~~~

The collapsed policy uses a forall-style execution policy such as
``RAJA::device_exec<block_size_1d>`` (when GPU is enabled) and runs a flattened
1D iteration space, reconstructing the logical indices internally.

On host OpenMP builds, ``RAJA::fornest_omp_collapse_policy<...>`` is a related
option that requests OpenMP collapse across the logical dimensions while keeping
the indices explicit (no linear index reconstruction). This can be preferable
when you want OpenMP's multi-loop collapse behavior without changing the loop
body to use a linear index.

For CUDA/HIP, *unsized* mapping tags (for example, ``device_global_x_direct``)
do not specify a compile-time thread/block shape. In that case RAJA chooses a
thread shape from a thread budget (either fixed by the launch policy or a
backend default) and the logical extents, then computes teams as
``ceil(extent / threads)`` per mapped dimension. The ``map-*-sized`` variants in
``fornest-basic`` demonstrate using *sized* mapping tags
(``device_*_size_*``) to explicitly request ``Threads`` and/or ``Teams`` at
compile time.

Tiling
~~~~~~

Tiling wraps a base mapping and adds tile loops per dimension. In the
``fornest-basic`` example, the tiling policies use a forall-style policy
(``forall_exec_pol``) to provide a fixed GPU thread budget, then execute a
launch where ``Threads`` are the tile sizes and ``Teams`` are computed as
``ceil(extent / tile)`` per dimension.

On GPU backends, this is largely a *mapping choice*: selecting a 2-D
``Threads(tx,ty)`` shape and launching enough teams/blocks to cover the logical
extents. In that sense, fixed tiling is equivalent to an explicit mapping that
fully specifies the thread shape (for CUDA/HIP, sized global mapping tags such
as ``device_global_size_x_direct<tx>`` and ``device_global_size_y_direct<ty>``),
and auto-tiling is similar in spirit to unsized global mapping where RAJA
chooses a reasonable 2-D thread shape.

Conceptually, a 2-D tiled CUDA/HIP mapping is similar to::

  // tile sizes tx, ty; extents rows, cols
  dim3 threads(tx, ty, 1);
  dim3 blocks(ceil(rows/tx), ceil(cols/ty), 1);

  __global__ void kernel(...)
  {
    // one tile per block; one point in the tile per thread
    int tile_i = blockIdx.x;
    int tile_j = blockIdx.y;
    int li = threadIdx.x;
    int lj = threadIdx.y;
    int i = tile_i * tx + li;  // == blockIdx.x * blockDim.x + threadIdx.x
    int j = tile_j * ty + lj;  // == blockIdx.y * blockDim.y + threadIdx.y
    if (i < rows && j < cols) body(i, j);
  }

  kernel<<<blocks, threads>>>(...);

The example demonstrates:

* ``tile-fixed``: compile-time tile sizes (e.g., ``fornest_tile_fixed<2>``)
* ``tile-runtime``: tile sizes provided at runtime via ``RAJA::TileSize``
* ``tile-auto``: RAJA chooses tile sizes from the iteration extents and thread
  budget

Unlike fixed/auto tiling, ``tile-runtime`` does not have a direct “mapping tag”
equivalent because mapping tags are compile-time types; if you need to choose a
CUDA/HIP block shape at runtime, use ``RAJA::launch`` directly.

Both mappings call the same logical body through one ``RAJA::fornest`` call:

.. literalinclude:: ../../../../examples/fornest-basic.cpp
   :start-after: // _fornest_call_start
   :end-before: // _fornest_call_end
   :language: C++

----------------------
Runtime Policy Choice
----------------------

This example selects a mapping at run time by switching between policy objects
that were *already instantiated at compile time* and compiled into the binary.
So the “runtime choice” here is choosing among pre-built implementations.

For true runtime policy selection from a single call site (choose a policy from
a type list by integer index), see the ``dynamic-fornest`` example
(``RAJA::dynamic_fornest``).

.. literalinclude:: ../../../../examples/fornest-basic.cpp
   :start-after: // _fornest_runtime_select_start
   :end-before: // _fornest_runtime_select_end
   :language: C++

Run the example and compare timing with the profiling tool normally used for
the target backend::

  ./fornest-basic collapse
  ./fornest-basic map
  ./fornest-basic map-global-sized
  ./fornest-basic map-block-thread-sized
  ./fornest-basic nested-loops
  ./fornest-basic tile-fixed
  ./fornest-basic tile-runtime
  ./fornest-basic tile-auto

The ``fornest-basic`` example also accepts numeric indices for these mappings
(printed in its help/usage output).

The ``map-*-sized`` variants demonstrate using compile-time sized mapping tags
(e.g., ``device_global_size_x_direct<N>`` and ``device_thread_size_x_loop<N>``)
to explicitly request ``Threads`` and/or ``Teams`` dimensions for the launch.

On OpenMP builds::

  ./fornest-basic omp-outer
  ./fornest-basic omp-collapse

Dynamic Policy Selection
------------------------

The ``dynamic-fornest`` example shows selecting among multiple fornest policies
at runtime via ``RAJA::dynamic_fornest<camp::list<...>>(pol, ...)``. In addition
to basic host/device variants, it includes CUDA/HIP launch-mapped variants that
use unsized and sized mapping tags to control how ``Threads`` and ``Teams`` are
chosen.

Restrictions
------------

The current ``RAJA::fornest`` implementation supports 2- and 3-level loop nests
over standard RAJA segments (e.g., ``RAJA::RangeSegment``). The mapping policy requires one loop
policy per segment. The collapsed mapping uses ``RAJA::layout_right`` by
default, or ``RAJA::layout_left`` when that layout tag is supplied, to control
which logical index is unit stride in the collapsed space.

Device mapping support follows the active ``RAJA::device_*`` aliases. On SYCL,
``device_block_{x,y,z}_{direct,loop}`` map to work-groups and
``device_thread_{x,y,z}_{direct,loop}`` map to work-items, with ``x/y/z``
corresponding to SYCL dimensions ``2/1/0``. Not every CUDA/HIP mapping alias has
a SYCL equivalent; for example, unsized ``device_global_{x,y,z}_direct`` and
sized thread/block aliases are CUDA/HIP-only.

Use direct ``RAJA::launch`` when the kernel needs explicit shared memory, team
synchronization, multiple cooperating loops in a single launch body, or other
hierarchical launch features that do not fit the single logical body accepted by
``RAJA::fornest``.

---------------------
Profiling with Caliper
---------------------

``RAJA::Name`` labels each ``RAJA::fornest`` call so Caliper can attribute time
to individual kernels. Build RAJA with Caliper + runtime plugins enabled and run
with Caliper environment variables::

  RAJA_CALIPER=1 CALI_CONFIG=runtime-report ./fornest-basic collapse

To produce a profile for offline analysis::

  RAJA_CALIPER=1 CALI_CONFIG=runtime-profile(output=fornest-basic.cali,output.format=cali) \
    ./fornest-basic map
