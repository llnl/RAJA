.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _tut-matrixtranspose-label:

----------------------
Matrix Transpose
----------------------

In :ref:`tut-kernelexecpols-label` and :ref:`tut-launchexecpols-label`,
we presented a simple array initialization kernel using ``RAJA::kernel`` and 
``RAJA::launch`` interfaces, respectively, and compared the two. This 
section describes the implementation of a matrix transpose kernel using both 
``RAJA::kernel`` and ``RAJA::launch`` interfaces. The intent is to 
compare and contrast the two, as well as introduce additional features of the 
interfaces.

There are exercise files 
``RAJA/exercises/kernel-matrix-transpose.cpp`` and
``RAJA/exercises/launch-matrix-transpose.cpp`` for you to work through if you 
wish to get some practice with RAJA. The files 
``RAJA/exercises/kernel-matrix-transpose_solution.cpp`` and
``RAJA/exercises/launch-matrix-transpose_solution.cpp`` contain
complete working code for the examples. You can use the solution files to 
check your work and for guidance if you get stuck. To build
the exercises execute ``make (kernel/launch)-matrix-transpose`` and ``make (kernel/launch)-matrix-transpose_solution``
from the build directory.

Key RAJA features shown in this example are:

  * ``RAJA::kernel`` method and kernel execution policies
  * ``RAJA::launch`` method and kernel execution interface

In the example, we compute the transpose of an input matrix
:math:`A` of size :math:`N_r \times N_c` and store the result in a second
matrix :math:`At` of size :math:`N_c \times N_r`.

First we define our matrix dimensions

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose_solution.cpp
   :start-after: // _mattranspose_dims_start
   :end-before: // _mattranspose_dims_end
   :language: C++

and wrap the data pointers for the matrices in ``RAJA::View`` objects to
simplify the multi-dimensional indexing:

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose_solution.cpp
   :start-after: // _mattranspose_views_start
   :end-before: // _mattranspose_views_end
   :language: C++

Then, a C-style for-loop implementation looks like this:

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose_solution.cpp
   :start-after: // _cstyle_mattranspose_start
   :end-before: // _cstyle_mattranspose_end
   :language: C++

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``RAJA::kernel`` Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ``RAJA::kernel`` variants, we use ``RAJA::statement::For`` and 
``RAJA::statement::Lambda`` statement types in the execution policies.
The complete sequential ``RAJA::kernel`` variant is:

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose_solution.cpp
   :start-after: // _raja_mattranspose_start
   :end-before: // _raja_mattranspose_end
   :language: C++

A CUDA ``RAJA::kernel`` variant for the GPU is similar with different policies
in the ``RAJA::statement::For`` statements:

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose_solution.cpp
   :start-after: // _raja_mattranspose_cuda_start
   :end-before: // _raja_mattranspose_cuda_end
   :language: C++

A notable difference between the CPU and GPU execution policy is the insertion 
of the ``RAJA::statement::CudaKernel`` type in the GPU version, which indicates
that the execution will launch a CUDA device kernel.

In the CUDA ``RAJA::kernel`` variant above, the thread-block size and
and number of blocks to launch is determined by the implementation of the 
``RAJA::kernel`` execution policy constructs using the sizes of the 
``RAJA::TypedRangeSegment`` objects in the iteration space tuple.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``RAJA::launch`` Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ``RAJA::launch`` variants, we use ``RAJA::loop`` methods 
to write a loop hierarchy within the kernel execution space. For a sequential 
implementation, we pass the ``RAJA::seq_launch_t`` template parameter
to the launch method and pass the ``RAJA::seq_exec`` parameter to the loop 
methods. The complete sequential ``RAJA::launch`` variant is:

.. literalinclude:: ../../../../exercises/launch-matrix-transpose_solution.cpp
   :start-after: // _raja_mattranspose_start
   :end-before: // _raja_mattranspose_end
   :language: C++

A CUDA ``RAJA::launch`` variant for the GPU is similar with CUDA
policies in the ``RAJA::loop`` methods. The complete 
``RAJA::launch`` variant is:

.. literalinclude:: ../../../../exercises/launch-matrix-transpose_solution.cpp
   :start-after: // _raja_mattranspose_cuda_start
   :end-before: // _raja_mattranspose_cuda_end
   :language: C++

A notable difference between the CPU and GPU ``RAJA::launch`` 
implementations is the definition of the compute grid. For the CPU
version, the argument list is empty for the ``RAJA::LaunchParams`` constructor.
For the CUDA GPU implementation, we define a 'Team' of one two-dimensional 
thread-block with 16 x 16 = 256 threads.
