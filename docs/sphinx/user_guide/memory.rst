.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _memory-label:

******
Memory
******

.. _memory-release-internal-pool-label:

Releasing RAJA Internal Pool Memory
===================================

Some RAJA backend operations, including reductions, scans, and sorts, may use
temporary storage. RAJA keeps some of this storage for reuse in internal memory
pools. Applications can call ``RAJA::release_unused_internal_memory()`` after
outstanding RAJA and device work has completed to return unused internal memory
to the backend allocator.

The function returns the number of bytes released and does not synchronize
CUDA, HIP, or SYCL work. RAJA does not do this automatically during static
destruction because CUDA and HIP runtime teardown can make late ``cudaFree`` or
``hipFree`` calls fail.  As such, this function must only be called before the end of main to avoid racing with backend runtime cleanup.
