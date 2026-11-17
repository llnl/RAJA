.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _tut-launchexecpols-label:

-----------------------------------------------------------
``RAJA::Launch`` Execution Policies
-----------------------------------------------------------

This section contains an exercise file 
``RAJA/exercises/launchintro-execpols.cpp`` for you to work through if you 
wish to get some practice with RAJA. The file 
``RAJA/exercises/launchintro-execpols_solution.cpp`` contains complete working 
code for the examples discussed in this section. You can use the solution file 
to check your work and for guidance if you get stuck. To build the exercises 
execute ``make launchintro-execpols`` and ``make launchintro-execpols_solution``
from the build directory.

Key RAJA features shown in this section are:

  * ``RAJA::launch`` kernel execution environment template
  * ``RAJA::loop`` loop execution template and execution policies

The examples in this section illustrate how to construct nested loop kernels
inside an ``RAJA::launch`` execution environment. In particular,
the goal is for you to gain an understanding of how to use execution policies
with nested ``RAJA::loop`` method calls to perform various nested
loop execution patterns. All examples use the same simple kernel, which
is a three-level loop nest to initialize the entries in an array. The kernels 
perform the same operations as the examples in :ref:`tut-kernelexecpols-label`.
By comparing the two sets of examples, you will gain an understanding of the
differences between the ``RAJA::kernel`` and the ``RAJA::launch`` 
interfaces.

We begin by defining some constants used throughout the examples and allocating
arrays to represent the array data:

.. literalinclude:: ../../../../exercises/launchintro-execpols_solution.cpp
   :start-after: _init_define_start
   :end-before: _init_define_end
   :language: C++

Note that we use the 'memory manager' routines contained in the exercise
directory to simplify the allocation process. In particular, CUDA unified 
memory is used when CUDA is enabled to simplify accessing the data on the
host or device.

Next, we execute a C-style nested for-loop version of the kernel to initialize
the entries in the 'reference' array that we will use to compare the results
of other variants for correctness:

.. literalinclude:: ../../../../exercises/launchintro-execpols_solution.cpp
   :start-after: _cstyle_tensorinit_seq_start
   :end-before: _cstyle_tensorinit_seq_end
   :language: C++

Note that we manually compute the pointer offsets for the (i,j,k) indices. 
To simplify the remaining kernel variants we introduce a ``RAJA::View``
object, which wraps the array data pointer and simplifies the multi-dimensional
indexing:

.. literalinclude:: ../../../../exercises/launchintro-execpols_solution.cpp
   :start-after: _3D_raja_view_start
   :end-before: _3D_raja_view_end
   :language: C++

Here 'aView' is a three-dimensional View with extent 'N' in each
coordinate based on a three-dimensional ``RAJA::Layout`` object where the
array entries will be accessed using indices of type 'int'.
indices of type ``int``. Please see :ref:`feat-view-label` for more information 
about the View and Layout types that RAJA provides for various indexing
patterns and data layouts.

Using the View, the C-style kernel looks like:

.. literalinclude:: ../../../../exercises/launchintro-execpols_solution.cpp
   :start-after: _cstyle_tensorinit_view_seq_start
   :end-before: _cstyle_tensorinit_view_seq_end
   :language: C++

Notice how accessing each (i,j,k) entry in the array is more natural,
and less error prone, using the View.

The corresponding RAJA sequential version using ``RAJA::launch`` is:

.. literalinclude:: ../../../../exercises/launchintro-execpols_solution.cpp
   :start-after: _raja_tensorinit_seq_start
   :end-before: _raja_tensorinit_seq_end
   :language: C++

This should be familiar to the reader who has read through the preceding
:ref:`tut-launchintro-label` section of this tutorial. As the 
``RAJA::launch`` method is templated on a host execution policy, the 
``RAJA::LaunchParams`` object can be defined without arguments as loop methods 
will get dispatched as standard C-Style for-loops.
     
Suppose we wanted to parallelize the outer 'k' loop using OpenMP multithreading.
A C-style version of this is:

.. literalinclude:: ../../../../exercises/launchintro-execpols_solution.cpp
   :start-after: _cstyle_tensorinit_omp_outer_start
   :end-before: _cstyle_tensorinit_omp_outer_end
   :language: C++

where we have placed the OpenMP directive ``#pragma omp parallel for`` before
the outer loop of the kernel.

The corresponding RAJA versions of the C-style OpenMP variant is:

.. literalinclude:: ../../../../exercises/launchintro-execpols_solution.cpp
   :start-after: _raja_tensorinit_omp_outer_start
   :end-before: _raja_tensorinit_omp_outer_end
   :language: C++

With the OpenMP version above, ``RAJA::launch`` method is templated on
a ``RAJA::omp_launch_t`` execution policy. The policy is used
to create an OpenMP parallel region, loop iterations may then be distributed
using ``RAJA::loop`` methods templated on ``RAJA::omp_for_exec`` 
execution policies. As before, the ``RAJA::LaunchParams`` object may be 
initialized without grid dimensions as the CPU does not require specifying a 
compute grid.

.. note::
  ``RAJA::omp_launch_t`` behaves like an OpenMP ``parallel`` region: the launch
  body is executed by every OpenMP thread. To distribute work, use
  ``RAJA::loop<RAJA::omp_for_exec>(...)`` (or another OpenMP *inner* policy)
  inside the launch body. If you use a sequential loop policy inside an OpenMP
  launch, each OpenMP thread will execute the **entire** loop independently.

The first RAJA-based kernel for parallel GPU execution using the RAJA CUDA
back-end we introduce is:

.. literalinclude:: ../../../../exercises/launchintro-execpols_solution.cpp
   :start-after: _raja_tensorinit_cuda_start
   :end-before: _raja_tensorinit_cuda_end
   :language: C++

where we have defined the CUDA thread-block dimensions as:

.. literalinclude:: ../../../../exercises/launchintro-execpols_solution.cpp
   :start-after: _cuda_blockdim_start
   :end-before: _cuda_blockdim_end
   :language: C++      

Here, we use the ``RAJA::cuda_launch_t`` policy type to
indicate that we want a CUDA kernel to be launched. The 'k', 'j', 'i'
iteration variables are mapped to CUDA threads and blocks using the CUDA 
execution policy types ``RAJA::cuda_block_z_direct``, 
``RAJA::cuda_global_y_direct``, and ``RAJA::cuda_global_x_direct``,
respectively. Thus, we use a two-dimensional CUDA thread-block and 
three-dimensional compute grid to map the loop iterations to CUDA threads. In 
comparison to the RAJA CUDA example in :ref:`tut-kernelexecpols-label` , 
``RAJA::loop`` methods support execution policies, which enable mapping 
directly to the global thread ID of a compute grid.

----------------------------------------
Global thread loop policies
----------------------------------------

RAJA provides both *team-local* thread mappings (e.g., ``cuda_thread_x_direct``)
and *global* thread mappings (e.g., ``cuda_global_x_direct``). The global thread
policies map loop indices to the global thread id of the grid:

  * ``cuda_global_x_direct`` corresponds to ``threadIdx.x + blockIdx.x * blockDim.x``
  * ``cuda_global_y_direct`` corresponds to ``threadIdx.y + blockIdx.y * blockDim.y``
  * ``cuda_global_z_direct`` corresponds to ``threadIdx.z + blockIdx.z * blockDim.z``

This is useful when you want a loop to be executed over the full grid rather
than within a single team, for example to flatten a 1D iteration space over all
threads in the launch.

When using global thread policies, ensure that the grid dimensions provided via
``RAJA::LaunchParams(RAJA::Teams(...), RAJA::Threads(...))`` provide enough
threads to cover the loop segment extents. RAJA will mask out-of-bounds indices
when the grid is larger than the iteration space.

.. note::
  HIP provides analogous policies (e.g., ``hip_global_x_direct``) with the same
  semantics when the HIP back-end is enabled.


Using a combination of ``RAJA::tile`` and ``RAJA::loop`` methods, 
we can create a loop tiling platform portable implementation. Here, is a 
CUDA variant: 

.. literalinclude:: ../../../../exercises/launchintro-execpols_solution.cpp
   :start-after: _raja_tensorinit_cuda_tiled_direct_start
   :end-before: _raja_tensorinit_cuda_tiled_direct_end
   :language: C++

We consider the kernel to be portable, because all of the execution policy types
and execution parameters can be replaced by other types and values without
changing the kernel code directly. 

The ``RAJA::tile`` methods are used to partition an iteration space into
tiles to be used within a ``RAJA::loop`` method. The '{i,j,k}_block_sz'
arguments passed to the ``RAJA::tile`` function specify the tile size
for each loop. In the case of GPU programming models, we define the tile size 
to correspond to the number of threads in a given dimension. Execution tile 
and loop execution policies are chosen to have CUDA blocks and threads map 
directly to tiles and entries in a tile.
	      
For context and comparison, here is the same kernel implementation using
CUDA directly:

.. literalinclude:: ../../../../exercises/launchintro-execpols_solution.cpp
   :start-after: _cuda_tensorinit_tiled_direct_start
   :end-before: _cuda_tensorinit_tiled_direct_end
   :language: C++

The ``nested_init`` device kernel used here is:

.. literalinclude:: ../../../../exercises/launchintro-execpols_solution.cpp
   :start-after: _cuda_tensorinit_kernel_start
   :end-before: _cuda_tensorinit_kernel_end
   :language: C++

A few differences between the CUDA and RAJA-CUDA versions are worth noting. 
First, the CUDA version uses the CUDA ``dim3`` construct to express the 
threads-per-block and number of thread-blocks to use: i.e., the 
``nthreads_per_block`` and ``nblocks`` variable definitions. The
``RAJA::launch`` interface takes compute dimensions through a
``RAJA::LaunchParams`` object. RAJA provides a macro ``RAJA_DIVIDE_CEILING_INT``
to perform the proper integer arithmetic to calculate the number of blocks 
based on the size of the array and the block size in each dimension. Second, the
mapping of thread identifiers to the (i,j,k) indices is explicit in the device 
kernel. Third, an explicit check of the (i,j,k) values is required in the CUDA 
implementation to avoid addressing memory out-of-bounds; i.e., 
``if ( i < N && j < N && k < N )...``. The RAJA variants set similar
definitions internally and **mask out indices that would be out-of-bounds.**
Note that we also inserted additional error checking with ``static_assert``
and ``cudaErrchk``, which is a RAJA macro, for printing CUDA device error
codes, to catch device errors if there are any.

Lastly, we show the RAJA HIP variants of the kernel, which are semantically
identical to the RAJA CUDA variants. First, the RAJA-HIP global-thread 
variant:

.. literalinclude:: ../../../../exercises/launchintro-execpols_solution.cpp
   :start-after: _raja_tensorinit_hip_start
   :end-before: _raja_tensorinit_hip_end
   :language: C++

and then the RAJA Launch HIP fixed thread-block size, tiled, direct thread 
mapping version:

.. literalinclude:: ../../../../exercises/launchintro-execpols_solution.cpp
   :start-after: _raja_tensorinit_hip_tiled_direct_start
   :end-before: _raja_tensorinit_hip_tiled_direct_end
   :language: C++

The only differences are that type names are changed to replace 'CUDA' types 
with 'HIP' types to use the RAJA HIP back-end.
