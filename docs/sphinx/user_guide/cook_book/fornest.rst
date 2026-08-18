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

.. note::
   ``RAJA::fornest`` is intended as a general loop-nest abstraction; the current
   implementation supports only 2- or 3-level loop nests.

The interface supports several policy families:

 * ``RAJA::fornest_collapsed_policy<ExecPolicy, LayoutTag>`` maps the product
   of the logical dimensions to a 1-D iteration space. The ``ExecPolicy`` is a
   regular forall-style policy such as ``RAJA::seq_exec`` or
   ``RAJA::device_exec<256>``. RAJA performs the linear-to-logical index
   reconstruction internally.
 * ``RAJA::fornest_mapping_policy<ExecPolicy, LoopPolicies...>`` maps the logical
   dimensions directly using one loop policy per dimension. Device builds can
   use ``RAJA::device_*`` loop-policy aliases; on CUDA/HIP these aliases map to
   CUDA/HIP global, block, and thread policies, and on SYCL they map to the
   corresponding work-group and work-item policies where supported.
 * ``RAJA::fornest_tiling_policy<...>`` adds per-dimension tiling on top of a
   loop nest. Tile sizes may be fixed, runtime-provided, or chosen by RAJA.
 * ``RAJA::fornest_omp_collapse_policy<...>`` requests OpenMP host collapse-style
   traversal (host OpenMP builds).
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

----------------
Mapping Policies
----------------

The minimal example source defines backend-specific policy aliases. CUDA/HIP can
use unsized global device mapping tags for explicit mapping. CUDA/HIP/SYCL can
use supported ``RAJA::device_*`` aliases for block/work-group and
thread/work-item mappings:

.. literalinclude:: ../../../../examples/fornest-basic.cpp
   :start-after: // _fornest_policy_aliases_start
   :end-before: // _fornest_policy_aliases_end
   :language: C++

The collapsed policy uses a forall-style execution policy such as
``RAJA::device_exec<block_size_1d>`` (when GPU is enabled). The mapping policy
uses one loop policy per dimension.

Both mappings call the same logical body through one ``RAJA::fornest`` call:

.. literalinclude:: ../../../../examples/fornest-basic.cpp
   :start-after: // _fornest_call_start
   :end-before: // _fornest_call_end
   :language: C++

----------------------
Runtime Policy Choice
----------------------

The mapping can be selected at run time by choosing which policy object is
passed to the common implementation:

.. literalinclude:: ../../../../examples/fornest-basic.cpp
   :start-after: // _fornest_runtime_select_start
   :end-before: // _fornest_runtime_select_end
   :language: C++

Run the example and compare timing with the profiling tool normally used for
the target backend::

  ./fornest-basic collapse
  ./fornest-basic map
  ./fornest-basic basic
  ./fornest-basic tile-fixed
  ./fornest-basic tile-runtime
  ./fornest-basic tile-auto
  ./fornest-basic dynamic

On OpenMP builds::

  ./fornest-basic omp-outer
  ./fornest-basic omp-collapse

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
