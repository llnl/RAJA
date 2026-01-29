.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _vectorization-label:

==========================
Vectorization (SIMD/SIMT)
==========================

.. warning:: **This section describes an initial draft of an incomplete,
             experimental RAJA capability. It is not considered ready
             for production, but it is ready for interested users to try.** 

             * We provide a basic description here so that interested users 
               can take a look, try it out, and provide input if they wish to 
               do so. The RAJA team values early feedback from users on new 
               capabilities.

             * There are no usage examples available in RAJA yet, except for
               tests. Examples will be made available as they are developed.

The aim of the RAJA API for SIMD/SIMT programming described in this section
is to make an implementation perform as well as if one used
SIMD/SIMT intrinsics directly in her code, but without the 
software complexity and maintenance burden associated with doing that. 
In particular, we want to *guarantee* that specified vectorization
occurs without requiring users to manually insert intrinsics in their code or 
rely on compiler auto-vectorization implementations.

.. note:: All RAJA vectorization types described here are in the namespace 
          ``RAJA::expt``.

Currently, the main abstractions in RAJA for SIMD/SIMT programming are:

  * ``Register`` which wraps underlying SIMD/SIMT hardware registers and 
    provides consistent uniform access to them, using intrinsics behind the
    API when possible. The register abstraction currently supports the 
    following hardware-specific ISAs (instruction set architectures): 
    AVX, AVX2, AVX512, CUDA, and HIP.
  * ``Vector`` which builds on ``Register`` to provide arbitrary length
    vectors and operations on them.
  * ``Matrix`` which builds on ``Register`` to provide arbitrary-sized
    matrices and operations on them, including support for column-major and 
    row-major data layouts.

Using these abstractions, RAJA provides an expression-template system that 
allows users to write linear algebra expressions on arbitrarily sized scalars, 
vectors, and matrices and have the appropriate SIMD/SIMT instructions
performed during expression evaluation. These capabilities integrate with 
RAJA :ref:`feat-view-label` capabilities, which insulate load/store and other 
operations from user code.


------------------------
Why Are We Doing This?
------------------------

Quoting Tim Foley in `Matt Pharr's blog <https://pharr.org/matt/blog/2018/04/18/ispc-origins>`_ -- "Auto-vectorization is not a programming model". This is
true, of course, unless you consider "hope for the best" that the compiler
optimizes the way you want to be a sound code development strategy.

Compiler auto-vectorization is problematic for multiple reasons. First, when 
vectorization is not explicit in source code, compilers must divine correctness 
when attempting to apply vectorization optimizations. Most compilers are very 
conservative in this regard, due to the possibility of data aliasing in C and
C++ and prioritizing correctness over performance. Thus, many vectorization 
opportunities are usually missed when one relies solely on compiler 
auto-vectorization.  Second, every compiler will treat your code differently 
since compiler implementations use different optimization heuristics, even in
different versions of the same compiler. So performance portability is not 
just an issue with respect to hardware, but also for compilers. Third, it is 
generally impossible for most application developers to clearly understand 
the choices made by compilers during optimization processes.

Using vectorization intrinsics in application source code is also problematic 
because different processors support different instruction set architectures
(ISAs) and so source code portability requires a mechanism that insulates it 
from architecture-specific code.

Writing GPU code makes a programmer be explicit about parallelization, and SIMD 
is really no different. RAJA enables single-source portable code across a 
variety of programming model back-ends. The RAJA vectorization abstractions
introduced here are an attempt to bring some convergence between SIMD 
and GPU programming by providing uniform access to hardware-specific 
acceleration.

.. important:: **Auto-vectorization is not a programming model.** --Tim Foley

---------------------
Register
---------------------

``RAJA::expt::Register<T, REGISTER_POLICY>`` is a class template with 
parameters for a data type ``T`` and a register policy ``REGISTER_POLICY``, 
which specifies the hardware register type. It is intended as a building block 
for higher level abstractions.  The ``RAJA::expt::Register`` interface provides
uniform access to register-level operations for different hardware features 
and ISA models. A ``RAJA::expt::Register`` type represents one SIMD register 
on a CPU architecture and 1 value/SIMT lane on a GPU architecture. 

``RAJA::expt::Register`` supports four scalar element types, ``int32_t``, 
``int64_t``, ``float``, and ``double``. These are the only types that are 
portable across all SIMD/SIMT architectures. ``Bfloat``, for example, is not 
portable, so we don't provide support for that type.

``RAJA::expt::Register`` supports the following SIMD/SIMT hardware-specific 
ISAs: AVX, AVX2, and AVX512 for SIMD CPU vectorization, and CUDA warp and
HIP wavefront for NVIDIA and AMD GPUs, respectively. Scalar support is 
provided for all hardware for portability and experimentation/analysis. 
Extensions to support other architectures may be forthcoming as they are 
needed and requested by users.

.. note:: One can use the ``RAJA::expt::Register`` type directly in her
          code. However, we do not recommend it. Instead, we want users to 
          employ higher level abstractions that RAJA provides.

Register Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``RAJA::expt::Register`` provides various operations which include:

  * Basic SIMD handling: get element, broadcast
  * Memory operations: load (packed, strided, gather) and store (packed, strided, scatter)
  * SIMD element-wise arithmetic: add, subtract, multiply, divide, vmin, vmax
  * Reductions: dot-product, sum, min, max
  * Special operations for matrix operations: permutations, segmented operations

.. note: All operations are provided for all hardware. Depending on hardware
         support, some operations may have slower serial performance; 
         e.g., gather/scatter.

Register DAXPY Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following code example shows how to use the ``RAJA::expt::Register`` 
class to perform a DAXPY kernel with AVX2 SIMD instructions.
While we do not recommend that you write code directly using the Register
class, but instead use the higher level VectorRegister abstraction, we use
the Register type here to illustrate the basics mechanics of SIMD 
vectorization::

  // Define array length
  int len = ...;

  // Define data used in kernel
  double a = ...;
  double const *X = ...; 
  double const *Y = ...; 
  double *Z = ...; 

  // Define an avx2 register, which has width of 4 doubles	
  using reg_t = RAJA::expt::Register<double, RAJA::expt::avx2_register>;
  int reg_width = reg_t::s_num_elem;

  // Compute daxpy in chunks of 4 values (register width) at a time
  for (int i = 0;i < len; i += reg_width){
    reg_t x, y;
    
    // Load 4 consecutive values of X, Y arrays into registers
    x.load_packed( X+i );
    y.load_packed( Y+i );

    // Perform daxpy on 4 values simultaneously and store in a register
    reg_t z = a * x + y;

    // Store register result in Z array
    z.store_packed( Z+i );
  }

  // Loop postamble code to complete daxpy operation when array length
  // is not an integer multiple of the register width
  int remainder = len % reg_width;
  if (remainder) {
    reg_t x, y;

    // 'i' is the starting array index of the remainder
    int i = len - remainder;
       
    // Load remainder values of X, Y arrays into registers 
    x.load_packed_n( X+i, remainder );
    y.load_packed_n( Y+i, remainder );

    // Perform daxpy on remainder values simultaneously and store in register
    reg_t z = a * x + y;

    // Store register result in Z array
    z.store_packed_n(Z+i, remainder);
  }

This code is guaranteed to vectorize since the ``RAJA::expt::Register`` 
operations insert the appropriate SIMD intrinsics into the operation 
calls. Since ``RAJA::expt::Register`` provides overloads of basic 
arithmetic operations, the SIMD DAXPY operation ``z = a * x + y`` looks 
like vanilla scalar code.

Because we are using bare pointers to the data, load and store 
operations are performed by explicit method calls in the code. Also, we must
write explicit *postamble* code to handle cases where the array length 
``len`` is not an integer multiple of the register width ``reg_width``. The 
postamble code performs the DAXPY operation on the *remainder* of the array 
that is excluded from the for-loop, which is strided by the register width.

**The need to write extra postamble code should make clear one reason why we 
do not recommend using ``RAJA::Register`` directly in application code.**

------------------
Vector Register
------------------

**To make code cleaner and more readable, the specific types are intended to
be used with ``RAJA::View`` and ``RAJA::expt::TensorIndex`` objects.**

``RAJA::expt::VectorRegister<T, REGISTER_POLICY, NUM_ELEM>`` provides an 
abstraction for a vector of arbitrary length. It is implemented using one or 
more ``RAJA::expt::Register`` objects. The vector length is independent of the 
underlying register width. The template parameters are: data type ``T``, 
vector register policy ``REGISTER_POLICY``, and ``NUM_ELEM`` which 
is the number of data elements of type ``T`` that fit in a register. The last 
two of these template parameters have defaults for all cases, so a user
need note provide them in most cases.

Recall that we said earlier that we do not recommended using 
``RAJA::expt::Register`` directly. One important reason for this is that 
decoupling the vector length from hardware register size allows one to write
simpler, more readable code that is easier to get correct. This should be 
clear from the code example below, when compared to the previous code example.

Vector Register DAXPY Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following code example shows the DAXPY computation discussed above,
but written using ``RAJA::expt::VectorRegister``, ``RAJA::expt::VectorIndex``, 
and ``RAJA::View`` types. Using these types, we can write cleaner, more 
concise code that is easier to get correct because it is simpler. For example,
we do not have to write the postamble code discussed earlier::

  // Define array length and data used in kernel (as before)
  int len = ...;
  double a = ...;
  double const *X = ...;
  double const *Y = ...;
  double *Z = ...;

  // Define vector register and index types
  using vec_t = RAJA::expt::VectorRegister<double, RAJA::expt::avx2_register>;
  using idx_t = RAJA::expt::VectorIndex<int, vec_t>;

  // Wrap array pointers in RAJA View objects   
  auto vX = RAJA::make_view( X, len );
  auto vY = RAJA::make_view( Y, len );
  auto vZ = RAJA::make_view( Z, len );

  // The 'all' variable gets the length of the arrays from the vX, vY, and 
  // vZ View objects and encodes the vector register type
  auto all = idx_t::all();

  // Compute the complete array daxpy in one line of code
  // this produces a vectorized loop and the loop postamble
  // in the executable
  vZ( all ) = a * vX( all ) + vY( all );

It should be clear that this code has several advantages over the previous 
code example. It is guaranteed to vectorize as before, but it is much easier 
to read, get correct, and maintain since the ``RAJA::View`` class handles the 
looping and postamble code automatically for arrays of arbitrary size. The 
``RAJA::View`` class provides overloads of the arithmetic operations based on 
the ``all`` variable and inserts the appropriate SIMD instructions and 
load/store operations to vectorize the operations that were explicit in the 
earlier example. It may be considered by some to be inconvenient to have to 
use the ``RAJA::View`` class, but it is easy to wrap bare pointers as is shown
here.

Expression Templates
^^^^^^^^^^^^^^^^^^^^^

The figure below shows the sequence of SIMD operations, as they are parsed to
form of an *abstract syntax tree (AST)*, for the DAXPY code in the vector 
register code example above.

.. figure:: ../figures/vectorET.png

   An AST illustration of the SIMD operations in the DAXPY code.

During compilation, a tree of *expression template* objects is constructed 
based on the order of operations that appear in the DAXPY kernel. Specifically, 
the operation sequence is the following:

  #. Load a chunk of values in 'vX' into a register.
  #. Broadcast the scalar value 'a' to each slot in a vector register.
  #. Load a chunk of values in 'vY' into a register.
  #. Multiply values in the 'a' register and 'vX' register and multiply
     by the values in the 'vY' register in a single vector FMA
     (Fused Multiply-Add) operation, storing the result in a register.
  #. Write the result in the register to the 'vZ' array.

``RAJA::View`` objects indexed by ``RAJA::TensorIndex`` objects 
(``RAJA::VectorIndex`` in this case) return *Load/Store* expression
template objects. Each expression template object is evaluated on assignment 
and a register chunk size of values is loaded into another register object.
Finally, the left-hand side of the expression is evaluated by storing the
chunk of values in the right-hand side result register into the array associated
with the view ``vZ`` on the left-hand side of the equal sign.


CPU/GPU Portability
^^^^^^^^^^^^^^^^^^^^^

It is important to note that the code in the example above can only run on a 
CPU; i.e., it is *not* portable to run on either a CPU or GPU because it does 
not include a way to launch a GPU kernel. The following code example shows 
how to enable the code to run on either a CPU or GPU via a run time choice::

  // array lengths and data used in kernel same as above

  // define vector register and index types
  using vec_t = RAJA::expt::VectorRegister<double>;
  using idx_t = RAJA::expt::VectorIndex<int, vec_t>;

  // array pointers wrapped in RAJA View objects as before
  // ...

  using cpu_launch = RAJA::expt::seq_launch_t;
  using gpu_launch = RAJA::expt::cuda_launch_t<false>; // false => launch
                                                       // CUDA kernel
                                                       // synchronously

  using pol_t = 
    RAJA::expt::LoopPolicy< cpu_launch, gpu_launch >;

  RAJA::expt::ExecPlace cpu_or_gpu = ...;

  RAJA::expt::launch<pol_t>( cpu_or_gpu, resources,

                             [=] RAJA_HOST_DEVICE (context ctx) {
                                 auto all = idx_t::all();
                                 vZ( all ) = a * vX( all ) + vY( all );
                             }
                           );

This version of the kernel can be run on a CPU or GPU depending on the run time
chosen value of the variable ``cpu_or_gpu``. When compiled, the code will 
generate versions of the kernel for a CPU and an CUDA GPU based on the 
parameters in the ``pol_t`` loop policy. The CPU version will be the same 
as the version described earlier. The GPU version is essentially the same 
but will run in a GPU kernel. Note that there is only one template argument 
passed to the register when ``vec_t`` is defined. 
``RAJA::expt::VectorRegister<double>`` uses defaults for the register policy, 
based on the system hardware, and number of data elements of type double that 
will fit in a register.

-------------------
Tensor Register
-------------------

``RAJA::expt::TensorRegister< >`` is a class template that provides a 
higher-level interface on top of ``RAJA::expt::Register``.
``RAJA::expt::TensorRegister< >`` wraps one or more 
``RAJA::expt::Register< >`` objects to create a tensor-like object.

.. note:: As with ``RAJA::expt::Register``, we don't recommend using 
          ``RAJA::expt::TensorRegister`` directly. Rather, we recommend using
          higher-level abstraction types that RAJA provides and which are 
          described below.

-----------------------
Matrix Registers
-----------------------

RAJA provides ``RAJA::expt::TensorRegister`` type aliases to support
matrices of arbitrary size and shape. These are:

  * ``RAJA::expt::SquareMatrixRegister<T, LAYOUT, REGISTER_POLICY>`` which
    abstracts operations on an N x N square matrix.
  * ``RAJA::expt::RectMatrixRegister<T, LAYOUT, ROWS, COLS, REGISTER_POLICY>`` 
    which abstracts operations on an N x M rectangular matrix.

Matrices are implemented using one or more ``RAJA::expt::Register`` 
objects. Data layout can be row-major or column major. Matrices are intended 
to be used with ``RAJA::View`` and ``RAJA::expt::TensorIndex`` objects,
similar to what was shown above in the ``RAJA::expt::VectorRegister`` example.

Matrix operations support matrix-matrix, matrix-vector, vector-matrix 
multiplication, and transpose operations. Rows or columns can be represented
with one or more registers, or a power-of-two fraction of a single register.
This is important for GPU warp/wavefront registers, which are 32-wide for
CUDA and 64-wide for HIP.

Here is a code example that performs the matrix-analogue of the 
vector DAXPY operation using square matrices::

  // Define matrix size and data used in kernel (similar to before)
  int N = ...;
  double a = ...;
  double const *X = ...;
  double const *Y = ...;
  double *Z = ...;

  // Define matrix register and row/column index types
  using mat_t = RAJA::expt::SquareMatrixRegister<double, 
                                                 RAJA::expt::RowMajorLayout>;
  using row_t = RAJA::expt::RowIndex<int, mat_t>;
  using col_t = RAJA::expt::ColIndex<int, mat_t>;

  // Wrap array pointers in RAJA View objects (similar to before)
  auto mX = RAJA::make_view( X, N, N );
  auto mY = RAJA::make_view( Y, N, N );
  auto mZ = RAJA::make_view( Z, N, N );

  using cpu_launch = RAJA::expt::seq_launch_t;
  using gpu_launch = RAJA::expt::cuda_launch_t<false>; // false => launch
                                                       // CUDA kernel
                                                       // synchronously
  using pol_t =
    RAJA::expt::LoopPolicy< cpu_launch, gpu_launch >;

  RAJA::expt::ExecPlace cpu_or_gpu = ...;

  RAJA::expt::launch<pol_t>( cpu_or_gpu, resources,

      [=] RAJA_HOST_DEVICE (context ctx) {
         auto rows = row_t::all();
         auto cols = col_t::all();
         mZ( rows, cols ) = a * mX( rows, cols ) + mY( rows, cols );
      }
    ); 

Conceptually, as well as implementation-wise, this is similar to the previous
vector example except the operations are on two-dimensional matrices. The 
kernel code is easy to read, it is guaranteed to vectorize, and iterating 
over the data is handled by RAJA view objects (register-width sized chunk, 
plus postamble scalar operations), and it can run on a CPU or NVIDIA GPU. As 
before, the ``RAJA::View`` arithmetic operation overloads insert the 
appropriate vector instructions in the code.

