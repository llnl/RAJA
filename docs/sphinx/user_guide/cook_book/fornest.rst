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

``RAJA::fornest`` runs a logical 2-D or 3-D loop body through selectable
mappings. It is intended for kernels where the source code should stay written
in logical multi-dimensional indices, but the best GPU mapping is not known from
the source alone.

The interface supports two mapping policy families:

 * ``RAJA::fornest_flattened_policy<ExecPolicy, LayoutTag>`` maps the product
   of the logical dimensions to a 1-D iteration space. The ``ExecPolicy`` is a
   regular forall-style policy such as ``RAJA::device_exec<256>`` (CUDA/HIP) or
   ``RAJA::seq_exec`` (host). RAJA performs the linear-to-logical index
   reconstruction internally.
 * ``RAJA::fornest_mapping_policy<ExecPolicy, LoopPolicies...>`` maps the logical
   dimensions directly using one loop policy per dimension. For CUDA/HIP this
   allows explicit mapping to global, block, or thread spaces.

This makes it possible to put both mappings behind one abstraction and choose
the mapping policy from problem size, backend, or measurements while preserving
one logical loop body.

----------------------------------
Why Compare Flat and Explicit Mappings
-------------------------------------

Many application kernels are logically 2-D or 3-D but have historically run on
a 1-D iteration space, with division and modulo operations used to recover the
logical indices. That approach can expose more parallelism when one logical
dimension is small.

For example, a ``cells x components`` kernel with only a few components may not
fit a fixed 2-D thread-block shape well. An explicit 2-D mapping can leave many
threads inactive in each block when one logical dimension is small. The
flattened mapping instead runs over ``cells * components`` as a 1-D space, which
can produce fuller blocks. The tradeoff is the extra index reconstruction
arithmetic. The explicit mapping can still win when the dimensions fit the block
shape, when the body is very small and index reconstruction dominates, or when
direct multi-dimensional mapping improves memory access or scheduling.

----------------
Mapping Policies
----------------

The minimal example source defines backend-specific policy aliases. CUDA/HIP can
use direct device mapping tags for explicit mapping:

.. literalinclude:: ../../../../examples/fornest-basic.cpp
   :start-after: // _fornest_policy_aliases_start
   :end-before: // _fornest_policy_aliases_end
   :language: C++

The flattened policy uses a forall-style execution policy such as
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

Run the example both ways and compare timing with the profiling tool normally
used for the target backend::

  ./fornest-basic flat
  ./fornest-basic map

--------------------
Current Capabilities
--------------------

``RAJA::fornest`` supports rank-2 and rank-3 loop nests over standard RAJA
segments (e.g., ``RAJA::RangeSegment``). The mapping policy requires one loop
policy per segment. The flattened mapping uses ``RAJA::layout_right`` by
default, or ``RAJA::layout_left`` when that layout tag is supplied, to control
which logical index is unit stride in the flattened space.

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

  RAJA_CALIPER=1 CALI_CONFIG=runtime-report ./fornest-basic flat

To produce a profile for offline analysis::

  RAJA_CALIPER=1 CALI_CONFIG=runtime-profile(output=fornest-basic.cali,output.format=cali) \
    ./fornest-basic map
