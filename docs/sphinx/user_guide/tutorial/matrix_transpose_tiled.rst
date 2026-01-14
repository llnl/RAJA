.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _tut-tiledmatrixtranspose-label:

----------------------
Tiled Matrix Transpose
----------------------

This section describes the implementation of a tiled matrix transpose kernel 
using both ``RAJA::kernel`` and ``RAJA::launch`` interfaces. The intent
is to compare and contrast the two. The discussion builds on 
:ref:`tut-matrixtranspose-label` by adding tiling to the matrix transpose 
implementation.

There are exercise files
``RAJA/exercises/kernel-matrix-transpose-tiled.cpp`` and
``RAJA/exercises/launch-matrix-transpose-tiled.cpp`` for you to work through 
if you wish to get some practice with RAJA. The files
``RAJA/exercises/kernel-matrix-transpose-tiled_solution.cpp`` and
``RAJA/exercises/launch-matrix-transpose-tiled_solution.cpp`` contain
complete working code for the examples. You can use the solution files to
check your work and for guidance if you get stuck. To build
the exercises execute ``make (kernel/launch)-matrix-transpose-tiled`` and 
``make (kernel/launch)-matrix-transpose-tiled_solution``
from the build directory.

Key RAJA features shown in this example are:

  * ``RAJA::kernel`` method and execution policies, and the ``RAJA::statement::Tile`` type
  * ``RAJA::launch`` method and execution policies, and the ``RAJA::tile`` type

As in :ref:`tut-matrixtranspose-label`, we compute the transpose of an input 
matrix :math:`A` of size :math:`N_r \times N_c` and storing the result in a 
second matrix :math:`At` of size :math:`N_c \times N_r`.

We will compute the matrix transpose using a tiling algorithm, which iterates 
over tiles and transposes the matrix entries in each tile.
The algorithm involves outer and inner loops to iterate over the tiles and
matrix entries within each tile, respectively.

As in :ref:`tut-matrixtranspose-label`, we start by defining the matrix 
dimensions. Additionally, we define a tile size smaller than the matrix 
dimensions and determine the number of tiles in each dimension. Note that we 
do not assume that tiles divide evenly the number of rows and and columns of 
the matrix. However, we do assume square tiles.

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose-tiled_solution.cpp
   :start-after: // _tiled_mattranspose_dims_start
   :end-before: // _tiled_mattranspose_dims_end
   :language: C++

Then, we wrap the matrix data pointers in ``RAJA::View`` objects to  
simplify the multi-dimensional indexing:

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose-tiled_solution.cpp
   :start-after: // _tiled_mattranspose_views_start
   :end-before: // _tiled_mattranspose_views_end
   :language: C++

The C-style for-loop implementation looks like this:

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose-tiled_solution.cpp
   :start-after: // _cstyle_tiled_mattranspose_start
   :end-before: // _cstyle_tiled_mattranspose_end
   :language: C++

.. note:: To prevent indexing out of bounds, when the tile dimensions do not
          divide evenly the matrix dimensions, the algorithm requires a 
          bounds check in the inner loops.

^^^^^^^^^^^^^^^^^^^^^^^^^^^
``RAJA::kernel`` Variants
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ``RAJA::kernel`` variants, we use ``RAJA::statement::Tile`` types
for the outer loop tiling and ``RAJA::tile_fixed`` types to 
indicate the tile dimensions. The complete sequential RAJA variant is:

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose-tiled_solution.cpp
   :start-after: // _raja_tiled_mattranspose_start
   :end-before: // _raja_tiled_mattranspose_end
   :language: C++

The ``RAJA::statement::Tile`` types compute the number of tiles needed to 
iterate over all matrix entries in each dimension and generate iteration 
index bounds for each tile, which are used to generate loops for the inner  
``RAJA::statement::For`` types. Thus, the explicit bounds checking logic in the 
C-style variant is not needed. Note that the integer template parameters
in the ``RAJA::statement::For`` types refer to the entries in the iteration 
space tuple passed to the ``RAJA::kernel`` method.

The ``RAJA::kernel`` CUDA variant is similar with sequential policies replaced 
with CUDA execution policies:

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose-tiled_solution.cpp
   :start-after: // _raja_mattranspose_cuda_start
   :end-before: // _raja_mattranspose_cuda_end
   :language: C++

A notable difference between the CPU and GPU execution policy is the insertion
of the ``RAJA::statement::CudaKernel`` type in the GPU version, which indicates
that the execution will launch a CUDA device kernel.

The CUDA thread-block dimensions are set based on the tile dimensions and the
iterates withing each tile are mapped directly to GPU threads in each block
due to the ``RAJA::cuda_thread_{x, y}_direct`` policies.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``RAJA::launch`` Variants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ``RAJA::launch`` variants, we use ``RAJA::tile`` methods
for the outer loop tiling and ``RAJA::loop`` methods
to iterate within the tiles. The complete sequential tiled 
``RAJA::launch`` variant is:

.. literalinclude:: ../../../../exercises/launch-matrix-transpose-tiled_solution.cpp
   :start-after: // _raja_tiled_mattranspose_start
   :end-before: // _raja_tiled_mattranspose_end
   :language: C++

Similar to the ``RAJA::statement::Tile`` type in the ``RAJA::kernel`` variant
above, the ``RAJA::tile`` method computes the number of tiles needed to 
iterate over all matrix entries in each dimension and generates a corresponding
iteration space for each tile, which is used to generate loops for the inner  
``RAJA::loop`` methods. Thus, the explicit bounds checking logic in the 
C-style variant is not needed.

A CUDA ``RAJA::launch`` tiled variant for the GPU is similar with 
CUDA policies in the ``RAJA::loop`` methods. The complete
``RAJA::launch`` variant is:

.. literalinclude:: ../../../../exercises/launch-matrix-transpose-tiled_solution.cpp
   :start-after: // _raja_mattranspose_cuda_start
   :end-before: // _raja_mattranspose_cuda_end
   :language: C++

A notable difference between the CPU and GPU ``RAJA::launch``
implementations is the definition of the compute grid. For the CPU
version, the argument list is empty for the ``RAJA::LaunchParams`` constructor.
For the CUDA GPU implementation, we define a 'Team' of one two-dimensional
thread-block with 16 x 16 = 256 threads.




