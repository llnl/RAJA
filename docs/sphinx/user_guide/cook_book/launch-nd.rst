.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _cook-book-launch-nd-label:

============================
Cooking with RAJA::launch_nd
============================

``RAJA::launch_nd`` runs a logical 2-D or 3-D loop body through selectable
launch-backed mappings. It is intended for kernels where the source code should
stay written in logical multi-dimensional indices, but the best GPU mapping is
not known from the source alone.

The interface supports two mapping policy families:

 * ``RAJA::launch_nd_flattened_policy<ExecPolicy, LayoutTag>`` maps the product
   of the logical dimensions to a 1-D launch. The ``ExecPolicy`` is a regular
   forall-style policy such as ``RAJA::cuda_exec<256>`` or
   ``RAJA::hip_exec<256>``. RAJA derives the launch parameters and performs the
   linear-to-logical index reconstruction internally.
 * ``RAJA::launch_nd_grid_policy<LaunchPolicy, LoopPolicies...>`` maps the
   logical dimensions directly to a true 2-D or 3-D launch. The user supplies
   the launch policy, one loop policy per segment, and the ``RAJA::LaunchParams``.

This makes it possible to put both mappings behind one abstraction and choose
the mapping policy from problem size, backend, or measurements while preserving
one logical loop body.

----------------------------------
Why Compare Flat and Grid Mappings
----------------------------------

Many application kernels are logically 2-D or 3-D but have historically run on
a 1-D iteration space, with division and modulo operations used to recover the
logical indices. That approach can expose more parallelism when one logical
dimension is small.

For example, a ``cells x components`` kernel with only a few components may not
fit a fixed ``16 x 16`` block well. A true 2-D grid can leave many component
threads inactive in each block. The flattened mapping instead launches over
``cells * components`` as a 1-D space, which can produce fuller blocks. The
tradeoff is the extra index reconstruction arithmetic. The true grid mapping
can still win when the dimensions fit the block shape, when the body is very
small and index reconstruction dominates, or when direct multi-dimensional
mapping improves memory access or scheduling.

----------------
Mapping Policies
----------------

The example source defines backend-specific policy aliases. CUDA uses direct
global loop policies for the true grid mapping:

.. literalinclude:: ../../../../examples/launch_nd.cpp
   :start-after: // _launch_nd_policy_aliases_start
   :end-before: // _launch_nd_policy_aliases_end
   :language: C++

The flattened policy uses a forall-style execution policy such as
``RAJA::cuda_exec<block_size_1d>`` or ``RAJA::hip_exec<block_size_1d>``. The
grid policy uses ``RAJA::LaunchPolicy`` and direct global loop policies such as
``RAJA::cuda_global_y_direct`` and ``RAJA::cuda_global_x_direct``.

Both mappings call the same logical body through one ``RAJA::launch_nd`` call:

.. literalinclude:: ../../../../examples/launch_nd.cpp
   :start-after: // _launch_nd_call_start
   :end-before: // _launch_nd_call_end
   :language: C++

----------------------
Runtime Policy Choice
----------------------

The mapping can be selected at run time by choosing which policy object is
passed to the common implementation:

.. literalinclude:: ../../../../examples/launch_nd.cpp
   :start-after: // _launch_nd_runtime_select_start
   :end-before: // _launch_nd_runtime_select_end
   :language: C++

Run the example both ways and compare timing with the profiling tool normally
used for the target backend::

  ./launch_nd flat
  ./launch_nd grid

The example uses ``num_cells = 257`` and ``num_comp = 5`` to show a case where
the ``16 x 16`` grid mapping has a small logical component dimension. Change
``num_cells``, ``num_comp``, ``block_size_1d``, ``block_x``, and ``block_y`` in
``RAJA/examples/launch_nd.cpp`` to explore when the flattened or true-grid
mapping is better for a kernel shape.

--------------------
Current Capabilities
--------------------

``RAJA::launch_nd`` currently supports ``RAJA::TypedRangeSegment`` packs created
with ``RAJA::nd_segments``. The grid mapping supports 2-D and 3-D loops and
requires one loop policy per segment. The flattened mapping uses
``RAJA::layout_right`` by default, or ``RAJA::layout_left`` when that layout tag
is supplied, to control which logical index is unit stride in the flattened
space.

Use direct ``RAJA::launch`` when the kernel needs explicit shared memory, team
synchronization, multiple cooperating loops in a single launch body, or other
hierarchical launch features that do not fit the single logical body accepted by
``RAJA::launch_nd``.
