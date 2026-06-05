.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _tut-atomichist-label:

--------------------------------------------------
Atomic Operations: Computing a Histogram
--------------------------------------------------

This section contains an exercise file ``RAJA/exercises/atomic-histogram.cpp``
for you to work through if you wish to get some practice with RAJA. The
file ``RAJA/exercises/atomic-histogram_solution.cpp`` contains complete
working code for the examples discussed in this section. You can use the
solution file to check your work and for guidance if you get stuck. To build
the exercises execute ``make atomic-histogram`` and ``make atomic-histogram_solution``
from the build directory.

Key RAJA features shown in this exercise are:

  * ``RAJA::forall`` kernel execution template and execution policies
  * ``RAJA::TypedRangeSegment`` iteration space construct
  * RAJA atomic add operation and RAJA atomic operation policies

The example uses an integer array of length 'N' randomly initialized with 
values in the interval [0, M). 

.. literalinclude:: ../../../../exercises/atomic-histogram_solution.cpp
   :start-after: _array_atomic_histogram_start
   :end-before: _array_atomic_histogram_end
   :language: C++

Each kernel iterates over the array and accumulates the number of occurrences 
of each value in [0, M) in another array named 'hist'. The kernels use atomic 
operations for the accumulation, which allow one to update a memory location 
referenced by a specific address in parallel without data races. The example 
shows how to use RAJA portable atomic operations and that they are used 
similarly for different programming model back-ends. 

.. note:: Each RAJA atomic operation requires an atomic policy type
          parameter that must be compatible with the execution policy for 
          the kernel in which it is used. This is similar to the reduction
          policies we described in :ref:`tut-dotproduct-label`.

For a complete description of supported RAJA atomic operations and 
atomic policies, please see :ref:`feat-atomics-label`.

All code snippets described below use the stride-1 iteration space range:

.. literalinclude:: ../../../../exercises/atomic-histogram_solution.cpp
   :start-after: _range_atomic_histogram_start
   :end-before: _range_atomic_histogram_end
   :language: C++

Here is the OpenMP version:

.. literalinclude:: ../../../../exercises/atomic-histogram_solution.cpp
   :start-after: _rajaomp_atomic_histogram_start
   :end-before: _rajaomp_atomic_histogram_end
   :language: C++

One is added to a slot in the 'bins' array when a value associated 
with that slot is encountered. Note that the ``RAJA::atomicAdd`` 
operation uses an OpenMP atomic policy, which is compatible with the OpenMP 
kernel execution policy.

The CUDA and HIP versions are similar:

.. literalinclude:: ../../../../exercises/atomic-histogram_solution.cpp
   :start-after: _rajacuda_atomic_histogram_start
   :end-before: _rajacuda_atomic_histogram_end
   :language: C++

and:

.. literalinclude:: ../../../../exercises/atomic-histogram_solution.cpp
   :start-after: _rajahip_atomic_histogram_start
   :end-before: _rajahip_atomic_histogram_end
   :language: C++

Here, the atomic add operations uses CUDA and HIP atomic policies, which are 
compatible with the CUDA and HIP kernel execution policies.

Note that RAJA provides an ``auto_atomic`` policy for easier usage and 
improved portability. This policy will choose the proper atomic operation 
for the execution policy used to run the kernel. Specifically, when OpenMP 
is enabled, the OpenMP atomic policy will be used, which is correct in a 
sequential or OpenMP execution context. Otherwise, the sequential atomic 
policy will be applied. Similarly, if it is encountered in a CUDA or HIP 
execution context, the corresponding GPU back-end atomic policy 
will be applied. 

For example, here is the CUDA version that uses the 'auto' atomic policy:

.. literalinclude:: ../../../../exercises/atomic-histogram_solution.cpp
   :start-after: _rajacuda_atomicauto_histogram_start
   :end-before: _rajacuda_atomicauto_histogram_end
   :language: C++

and the HIP version:

.. literalinclude:: ../../../../exercises/atomic-histogram_solution.cpp
   :start-after: _rajahip_atomicauto_histogram_start
   :end-before: _rajahip_atomicauto_histogram_end
   :language: C++

The same CUDA and HIP kernel execution policies as in the previous examples 
are used.

