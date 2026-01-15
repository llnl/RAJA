.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _tut-matrixtransposelocalarray-label:

-----------------------------------------
Tiled Matrix Transpose with Local Array
-----------------------------------------

This section extends the discussion in :ref:`tut-tiledmatrixtranspose-label`
by adding *local array* objects which are used to store data for each tile in
CPU stack-allocated arrays or GPU thread local and shared memory to be used
within kernels.

There are exercise files
``RAJA/exercises/kernel-matrix-transpose-local-array.cpp`` and
``RAJA/exercises/launch-matrix-transpose-local-array.cpp`` for you to work
through if you wish to get some practice with RAJA. The files
``RAJA/exercises/kernel-matrix-transpose-local-array._solutioncpp`` and
``RAJA/exercises/launch-matrix-transpose-local-array_solution.cpp`` contain
complete working code for the examples. You can use the solution files to
check your work and for guidance if you get stuck. To build
the exercises execute ``make (kernel/launch)-matrix-transpose-local-array`` and ``make (kernel/launch)-matrix-transpose-local-array_solution``
from the build directory.

Key RAJA features shown in this example are:

  * ``RAJA::kernel_param`` method and execution policy usage with multiple lambda expressions
  * ``RAJA::statement::Tile`` type for loop tiling
  * ``RAJA::statement::ForICount`` type for generating local tile indices
  * ``RAJA::LocalArray`` type for thread-local tile memory arrays
  * ``RAJA::launch`` kernel execution interface
  * ``RAJA::tile`` type for loop tiling
  * ``RAJA::loop_icount`` method to generate local tile indices for Launch
  * ``RAJA_TEAM_SHARED`` macro for thread-local tile memory arrays

As in :ref:`tut-tiledmatrixtranspose-label`, this example computes the
transpose of an input matrix :math:`A` of size :math:`N_r \times N_c` and
stores the result in a second matrix :math:`At` of size :math:`N_c \times N_r`.
The operation uses a local memory tiling algorithm, which tiles the outer
loops and iterates over tiles in inner loops. The algorithm first loads
input matrix entries into a local two-dimensional array for a tile, and then
reads from the tile swapping the row and column indices to generate the output
matrix.

We choose tile dimensions smaller than the dimensions of the matrix and note
that it is not necessary for the tile dimensions to divide evenly the number
of rows and columns in the matrix. As in the
:ref:`tut-tiledmatrixtranspose-label` example, we start by defining the number
of rows and columns in the matrices, the tile dimensions, and the number of
tiles.

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose-local-array_solution.cpp
   :start-after: // _mattranspose_localarray_dims_start
   :end-before: // _mattranspose_localarray_dims_end
   :language: C++

We also use RAJA View objects to simplify the multi-dimensional indexing
as in the :ref:`tut-tiledmatrixtranspose-label` example.

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose-local-array_solution.cpp
   :start-after: // _mattranspose_localarray_views_start
   :end-before: // _mattranspose_localarray_views_end
   :language: C++

The complete sequential C-style implementation of the tiled transpose operation
using a stack-allocated local array for the tiles is:

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose-local-array_solution.cpp
   :start-after: // _mattranspose_localarray_cstyle_start
   :end-before: // _mattranspose_localarray_cstyle_end
   :language: C++

.. note:: * To prevent indexing out of bounds, when the tile dimensions do not
            divide evenly the matrix dimensions, we use a bounds check in the
            inner loops.
          * For efficiency, we order the inner loops so that reading from
            the input matrix and writing to the output matrix both use
            stride-1 data access.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``RAJA::kernel`` Variants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``RAJA::kernel`` interface provides mechanisms to tile loops and use
*local arrays* in kernels so that algorithm patterns like the C-style kernel
above can be implemented with RAJA. When using ``RAJA::kernel``, a
``RAJA::LocalArray`` type specifies an object whose memory is created inside
a kernel using a statement type in a RAJA kernel execution policy. The local
array data is only usable within the kernel. See :ref:`feat-local_array-label`
for more information.

``RAJA::kernel`` methods also support loop tiling statements which determine
the number of tiles needed to perform an operation based on tile size and
extent of the corresponding iteration space. Moreover, lambda expressions for
the kernel will not be invoked for iterations outside the bounds of an
iteration space when tile dimensions do not divide evenly the size of the
iteration space; thus, no conditional checks on loop bounds are needed
inside inner loops.

For the RAJA version of the matrix transpose kernel above, we define the
type of the ``RAJA::LocalArray`` used for matrix entries in a tile and
create an object to represent it:

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose-local-array_solution.cpp
   :start-after: // _mattranspose_localarray_start
   :end-before: // _mattranspose_localarray_end
   :language: C++

The template parameters that define the type are: the array data type, the
data stride permutation for the array indices (here the identity permutation
is given, so the default RAJA conventions apply; i.e., the rightmost array
index will be stride-1), and the array dimensions. Next, we compare two
``RAJA::kernel`` implementations of the matrix transpose operation.

The complete RAJA sequential CPU variant with kernel execution policy and
kernel is:

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose-local-array_solution.cpp
   :start-after: // _mattranspose_localarray_raja_start
   :end-before: // _mattranspose_localarray_raja_end
   :language: C++

In the execution policy, the ``RAJA::statement::Tile`` types define
tiling of the outer 'row' (iteration space tuple index '1') and 'col'
(iteration space tuple index '0') loops, as well as tile sizes
(``RAJA::tile_fixed`` types) and loop execution policies. Next,
the ``RAJA::statement::InitLocalMem`` type allocates the local tile array
based on the memory policy type (here, we use ``RAJA::cpu_tile_mem`` for
a CPU stack-allocated array). The ``RAJA::ParamList<2>`` parameter indicates
that the local array object is associated with position '2' in the parameter
tuple argument passed to the ``RAJA::kernel_param`` method. The first two
entries in the parameter tuple indicate storage for the local tile indices
that are used in the two lambda expressions that comprise the kernel body.
Finally, we have two sets of nested inner loops for reading the input matrix
entries into the local tile array and writing them out to the output matrix
transpose. The inner bodies of each of these loop nests are identified by
lambda expression invocation statements ``RAJA::statement::Lambda<0>`` for
the first lambda passed as an argument to the ``RAJA::kernel_param`` method
and ``RAJA::statement::Lambda<1>`` for the second lambda argument.

Note that the loops within tiles use ``RAJA::statement::ForICount`` types
rather than ``RAJA::statement::For`` types that we saw in the
tiled matrix transpose example in :ref:`tut-tiledmatrixtranspose-label`.
The ``RAJA::statement::ForICount`` type generates local tile indices that
are passed to lambda loop body expressions to index into the local tile
memory array. As the reader will observe, there is no local tile index
computation needed in the lambdas for the RAJA version of the kernel as a
result. The first integer template parameter for each
``RAJA::statement::ForICount`` type indicates the item in the iteration space
tuple passed to the ``RAJA::kernel_param`` method to which it applies.
The second template parameter for each
``RAJA::statement::ForICount`` type indicates the position in the parameter
tuple passed to the ``RAJA::kernel_param`` method that will hold the
associated local tile index. For more detailed discussion of RAJA loop tiling
statement types, please see :ref:`feat-tiling-label`.

Now that we have described the execution policy in some detail, let's pull
everything together by briefly walking though the call to the
``RAJA::kernel_param`` method, which is similar to ``RAJA::kernel`` but takes
additional arguments needed to execute the operations involving local
tile indices and the local memory array. The first argument is a tuple of
iteration spaces that define the iteration ranges for the levels in the loop
nest. Again, the first integer parameters given to the ``RAJA::statement::Tile``
and ``RAJA::statement::ForICount`` types identify the tuple entry to which
they apply. The second argument::

  RAJA::make_tuple((int)0, (int)0, Tile_Array)

is a tuple of data parameters that will hold the local tile indices and
``RAJA::LocalArray`` tile memory. The tuple entries are
associated with various statements in the execution policy as we described
earlier. Next, two lambda expression arguments are passed to the
``RAJA::kernel_param`` method for reading and writing the input and output
matrix entries, respectively.

.. note:: ``RAJA::kernel_param`` accepts a parameter tuple argument after
          the iteration space tuple, which enables the parameters to be
          used in multiple lambda expressions in a kernel.

In the kernel, both lambda expressions take the same five arguments. The first
two are the matrix global column and row indices associated with the iteration
space tuple. The next three arguments correspond to the parameter tuple entries.
The first two of these are the local tile indices used to access entries in the
``RAJA::LocalArray`` object memory. The last argument is a reference to the
``RAJA::LocalArray`` object itself.

The next ``RAJA::kernel_param`` variant we present works the same as the one
above. It is different from the previous version since we include
additional template parameters in the ``RAJA::statement::Lambda`` types to
indicate which arguments each lambda expression takes and in which order.
Here is the complete version including execution policy and kernel:

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose-local-array_solution.cpp
   :start-after: // _raja_mattranspose_lambdaargs_start
   :end-before: // _raja_mattranspose_lambdaargs_start
   :language: C++

Here, the two ``RAJA::statement::Lambda`` types in the execution policy show
two different ways to specify the segments (``RAJA::Segs``)
associated with the matrix column and row indices. That is, we can use a
``Segs`` statement for each argument, or include multiple segment ids in one
statement.

Note that we are using ``RAJA::statement::For`` types for the inner tile
loops instead of `RAJA::statement::ForICount`` types used in the first variant.
As a consequence of specifying lambda arguments, there are two main differences.
The local tile indices are properly computed and passed to the lambda
expressions as a result of the ``RAJA::Offsets`` types that appear
in the lambda statement types. The ``RAJA::statement::Lambda`` type for each
lambda shows the two ways to specify the local tile index arguments; we can
use an ``Offsets`` statement for each argument, or include multiple segment
ids in one statement. Lastly, there is only one entry in the parameter
tuple in this case, the local tile array. The placeholders in the
previous example are not needed.

.. note:: In this example, we need all five arguments in each lambda
          expression so the lambda expression argument lists are
          the same. Another use case for the template parameter argument
          specification described here is to be able to pass only the
          arguments used in a lambda expression. In particular when we use
          multiple lambda expressions to represent a kernel, each lambda
          can have a different argument lists from the others.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``RAJA::launch`` Variants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``RAJA::launch`` interface provides mechanisms to tile loops and use
*local arrays* in kernels to support algorithm patterns like the C-style kernel
above. When, using ``RAJA::launch``, the ``RAJA_TEAM_SHARED`` macro is
used to create a GPU a static sized shared memory array when using the CUDA and HIP backends,
static shared memory using SYCL is currently not supported. On the CPU, allocating
``RAJA_TEAM_SHARED`` corresponds to allocating memory on the stack.
Alternatively, one can allocated dynamic shared memory by specifying the amount of shared memory in
the ``RAJA::LaunchParams`` struct. Dynamic shared memory is supported with all
backends and will be demonstrated as our second example. On the CPU dynamic
shared memory is mapped to heap memory and allocated via malloc at kernel launch.

As a first example, we illustrate the usage of static shared memory
and the use of ``RAJA::launch`` tiling methods. RAJA tiling methods
take a iteration space in ``RAJA::tile`` and output tiles which
the ``RAJA::loop_icount`` method can iterate over and generate
global and local tile index offsets.  Moreover, lambda expressions for these
methods will not be invoked for iterations outside the bounds of an iteration
space when tile dimensions do not evenly divide the size of the iteration space;
thus, no conditional checks on loop bounds are needed inside inner loops.

A complete RAJA sequential CPU variant with kernel execution policy and
kernel is:

.. literalinclude:: ../../../../exercises/launch-matrix-transpose-local-array_solution.cpp
   :start-after: // _mattranspose_localarray_raja_start
   :end-before: // _mattranspose_localarray_raja_end
   :language: C++

Here, the ``RAJA::tile`` method is used to create tilings of the outer
'row' and 'col' iteration spaces. The ``RAJA::tile`` method
takes an additional argument specifying the tile size for the corresponding
loop. To traverse the tile, we use the ``RAJA::loop_icount`` method,
which is similar to the ``RAJA::ForICount`` statement used in a
``RAJA::kernel`` execution policy as shown above. A
``RAJA::loop_icount`` method call
will generate local tile index associated with the outer global index.
The local tile index is necessary as we use it to read and write entries
from/to global memory to ``RAJA_TEAM_SHARED`` memory array.

As an alternative to static shared memory, the matrix transpose kernel may be
express using dynamic shared memory. Prior to invoking the amount of shared memory
must be specified

.. literalinclude:: ../../../../examples/dynamic_mat_transpose.cpp
   :start-after: // _dynamic_mattranspose_shared_mem_start
   :end-before: // _dynamic_mattranspose_shared_mem_end
   :language: C++

The amount of shared memory is then specifed in the ``RAJA::LaunchParams`` struct
and then accessed within the kernel using the LaunchContext's ``getSharedMemory`` method.
The ``getSharedMemory`` method may be invoked multiple times each time returning an
offset to the shared memory buffer. Since the offset moves per ``getSharedMemory``
call (also known as bump style allocation), it becomes necessary to reset the allocator offset
count at the end of the shared memory scope to avoid going beyond the buffer size.
The full example of matrix transpose with dynamic shared memory is provided below

.. literalinclude:: ../../../../examples/dynamic_mat_transpose.cpp
   :start-after: // _dynamic_mattranspose_kernel_start
   :end-before: // _dynamic_mattranspose_kernel_end
   :language: C++
