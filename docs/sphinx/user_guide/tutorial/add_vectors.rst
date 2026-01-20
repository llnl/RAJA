.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _tut-addvectors-label:

--------------------------------------
Basic Loop Execution: Vector Addition
--------------------------------------

This section contains an exercise file ``RAJA/exercises/vector-addition.cpp`` 
for you to work through if you wish to get some practice with RAJA. The 
file ``RAJA/exercises/vector-addition_solution.cpp`` contains complete 
working code for the examples discussed in this section. You can use the 
solution file to check your work and for guidance if you get stuck. To build
the exercises execute ``make vector-addition`` and ``make vector-addition_solution``
from the build directory.

Key RAJA features shown in this example are:

  * ``RAJA::forall`` loop execution template and execution policies
  * ``RAJA::TypedRangeSegment`` iteration space construct

In the example, we add two vectors 'a' and 'b' of length N and
store the result in vector 'c'. A simple C-style loop that does this is:

.. literalinclude:: ../../../../exercises/vector-addition_solution.cpp
   :start-after: _cstyle_vector_add_start
   :end-before: _cstyle_vector_add_end
   :language: C++ 

^^^^^^^^^^^^^^^^^^^^^
RAJA Variants
^^^^^^^^^^^^^^^^^^^^^

For the RAJA variants of the vector addition kernel, we replace the C-style 
for-loop with a call to the ``RAJA::forall`` loop execution template method.
The method takes an iteration space and the vector addition loop body as
a C++ lambda expression. We pass the object::

  RAJA::TypedRangeSegment<int>(0, N)

for the iteration space, which is contiguous sequence of integral 
values [0, N) (for more information about RAJA loop indexing concepts, 
see :ref:`feat-index-label`). The loop execution template method requires an 
execution policy template type that specifies how the loop is to run
(for more information about RAJA execution policies,
see :ref:`feat-policies-label`).

For a RAJA sequential variant, we use the ``RAJA::seq_exec`` execution
policy type:

.. literalinclude:: ../../../../exercises/vector-addition_solution.cpp
   :start-after: _rajaseq_vector_add_start
   :end-before: _rajaseq_vector_add_end
   :language: C++ 

When using the RAJA sequential execution policy, the resulting loop
implementation is essentially the same as writing a C-style for-loop
with no directives applied to the loop. The compiler is allowed to 
perform any optimizations that its heuristics deem are safe and potentially
beneficial for performance. To attempt to force the compiler to generate SIMD 
vector instructions, we would use the RAJA SIMD execution policy:: 

  RAJA::simd_exec

To run the kernel with OpenMP multithreaded parallelism on a CPU, we use the
``RAJA::omp_parallel_for_exec`` execution policy:

.. literalinclude:: ../../../../exercises/vector-addition_solution.cpp
   :start-after: _rajaomp_vector_add_start
   :end-before: _rajaomp_vector_add_end
   :language: C++ 

This will distribute the loop iterations across CPU threads and run the 
loop over threads in parallel. In particular, this is what you would get if
you wrote the kernel using a C-style loop with an OpenMP pragma directly::

  #pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    c[i] = a[i] + b[i];
  }

To run the kernel on a CUDA GPU device, we use the ``RAJA::cuda_exec``
policy:

.. literalinclude:: ../../../../exercises/vector-addition_solution.cpp
   :start-after: _rajacuda_vector_add_start
   :end-before: _rajacuda_vector_add_end
   :language: C++ 

Since the lambda defining the loop body will be passed to a device kernel, 
it must be decorated with the ``__device__`` attribute.
This can be done directly or by using the ``RAJA_DEVICE`` macro.

Note that the CUDA execution policy type requires a template argument 
``CUDA_BLOCK_SIZE``, which specifies the number of threads to run in each 
CUDA thread block launched to run the kernel.

For additional performance tuning options, the ``RAJA::cuda_exec_explicit`` 
policy is also provided, which allows a user to specify the minimum number 
of thread blocks to launch at a time on each streaming multiprocessor (SM):

.. literalinclude:: ../../../../exercises/vector-addition_solution.cpp
   :start-after: _rajacuda_explicit_vector_add_start
   :end-before: _rajacuda_explicit_vector_add_end
   :language: C++ 

Note that the third boolean template argument is used to express whether the
kernel launch is synchronous or asynchronous. This is optional and is 
'false' by default. A similar defaulted optional argument is supported for 
other RAJA GPU (e.g., CUDA or HIP) policies.

Lastly, to run the kernel on a GPU using the RAJA HIP back-end, 
we use the ``RAJA::hip_exec`` policy:

.. literalinclude:: ../../../../exercises/vector-addition_solution.cpp
   :start-after: _rajahip_vector_add_start
   :end-before: _rajahip_vector_add_end
   :language: C++

