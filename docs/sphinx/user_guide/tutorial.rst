.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _tutorial-label:

****************************
RAJA Tutorial and Examples
****************************

The following sections contain tutorial material and examples that describe
how to use RAJA features.

===============
RAJA Tutorial
===============

This section contains a self-paced tutorial that shows how to use many RAJA
features by way of a sequence of examples and exercises. Each exercise is
located in files in the ``RAJA/exercises`` directory, one *exercise* file with
code sections removed and comments containing instructions to fill in the
missing code parts and one *solution* file containing complete working code to
compare with and for guidance if you get stuck working on the exercise file.
You are encouraged to build and run the exercises and modify them to try out
different variations.

We also maintain a repository of tutorial slide presentations
`RAJA Tutorials Repo <https://github.com/LLNL/RAJA-tutorials>`_ which we use
when we give in-person or virtual online tutorials in various venues. The
presentations complement the material found here. The tutorial material
evolves as we add new features to RAJA, so refer to it periodically if you
are interested in learning about new things in RAJA.

To understand the GPU examples (e.g., CUDA), it is also important to know the
difference between CPU (host) and GPU (device) memory allocations and how
transfers between those memory spaces work. For a detailed discussion, see
`Device Memory <http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory>`_.

It is important to note that RAJA does not provide a memory model. This is by
design as application developers who use RAJA prefer to manage memory
in different ways. Thus, users are responsible for ensuring that data is
properly allocated and initialized on a GPU device when running GPU code.
This can be done using explicit host and device allocation and copying between
host and device memory spaces or via unified memory (UM), if available.
The RAJA Portability Suite contains other libraries, namely
`CHAI <https://github.com/LLNL/CHAI>`_ and
`Umpire <https://github.com/LLNL/Umpire>`_, that complement RAJA by
providing alternatives to manual programming model specific memory operations.

.. note:: Most of the CUDA GPU exercises use unified memory (UM) via a simple
          memory manager capability provided in a file in the ``RAJA/exercises``
          directory. HIP GPU exercises use explicit host and device memory
          allocations and explicit memory copy operations to move data between
          the two.

.. _tutorial-lambda-label:

===============================
A Little C++ Background
===============================

To understand the discussion and code examples, a working knowledge of C++
templates and lambda expressions is required. So, before we begin, we provide
a bit of background discussion of basic aspects of how RAJA use employs C++
templates and lambda expressions, which is essential to use RAJA successfully.

RAJA is almost an entirely header-only library that makes heavy use of
C++ templates. Using RAJA most easily and effectively is done by representing
the bodies of loop kernels as C++ lambda expressions. Alternatively, C++
functors can be used, but they make application source code more complex,
potentially placing a significant negative burden on source code readability
and maintainability.

-----------------------------------
C++ Templates
-----------------------------------

C++ templates enable one to write type-generic code and have the compiler
generate an implementation for each set of template parameter types specified.
For example, the ``RAJA::forall`` method to execute loop kernels is
essentially method defined as::

  template <typename ExecPol,
            typename IdxType,
            typename LoopBody>
  forall(IdxType&& idx, LoopBody&& body) {
     ...
  }

Here, "ExecPol", "IdxType", and "LoopBody" are C++ types that a user specifies
in her code and which are seen by the compiler when the code is built.
For example::

  RAJA::forall< RAJA::seq_exec >( RAJA::TypedRangeSegment<int>(0, N), [=](int i) {
    a[i] = b[i] + c[i];
  });

is a sequential CPU RAJA kernel that performs an element-by-element vector sum.
The C-style analogue of this kernel is::

  for (int i = 0; i < N; ++i) {
    a[i] = b[i] + c[i];
  }

The execution policy type ``RAJA::seq_exec`` template argument
is used to choose as specific implementation of the
``RAJA::forall`` method. The ``IdxType`` and ``LoopBody`` types are deduced by
the compiler based the arguments passed to the ``RAJA::forall`` method;
i.e., the ``IdxType`` is the stride-1 index range::

  RAJA::TypedRangeSegment<int>(0, N)

and the ``LoopBody`` type is the lambda expression::

  [=](int i) { a[i] = b[i] + c[i]; }

-----------------------------------
Elements of C++ Lambda Expressions
-----------------------------------

Here, we provide a brief description of the basic elements of C++ lambda
expressions. A more technical and detailed discussion is available here:
`Lambda Functions in C++11 - the Definitive Guide <https://www.cprogramming.com/c++11/c++11-lambda-closures.html>`_

Lambda expressions were introduced in C++ 11 to provide a lexical-scoped
name binding; specifically, a *closure* that stores a function with a data
environment. That is, a lambda expression can *capture* variables from an
enclosing scope for use within the local scope of the function expression.

A C++ lambda expression has the following form::

  [capture list] (parameter list) {function body}

The ``capture list`` specifies how variables outside the lambda scope are pulled
into the lambda data environment. The ``parameter list`` defines arguments
passed to the lambda function body -- for the most part, lambda arguments
are just like arguments in a regular C++ method. Variables in the capture list
are initialized when the lambda expression is created, while those in the
parameter list are set when the lambda expression is called. The body of a
lambda expression is similar to the body of an ordinary C++ method.
RAJA kernel execution templates, such as ``RAJA::forall`` and ``RAJA::kernel``
that we will describe in detail later, pass arguments
to lambdas based on usage and context such as loop iteration indices.

A C++ lambda expression can capture variables in the capture list *by value*
or *by reference*. This is similar to how arguments to C++ methods are passed;
i.e., *pass-by-reference* or *pass-by-value*. However, there are some subtle
differences between lambda variable capture rules and those for ordinary
methods. **Variables included in the capture list with no extra symbols are
captured by value.** Variables captured by value are effectively *const*
inside the lambda expression body and cannot be written to.
Capture-by-reference is accomplished by using the reference symbol '&' before
the variable name similar to C++ method arguments.  For example::

  int x;
  int y = 100;
  [&x, &y](){ x = y; };

generates a lambda expression that captures both 'x' and 'y' by reference
and assigns the value of 'y' to 'x' when called. The same outcome would be
achieved by writing::

  [&](){ x = y; };   // capture all lambda arguments by reference...

or::

  [=, &x](){ x = y; };  // capture 'x' by reference and 'y' by value...

Note that the following two attempts will generate compilation errors::

  [=](){ x = y; };      // error: all lambda arguments captured by value,
                        //        so cannot assign to 'x'.
  [x, &y](){ x = y; };  // error: cannot assign to 'x' since it is captured
                        //        by value.

.. note:: A variable that is captured by value in a lambda expression is
          **read-only.**

----------------------------------------
A Few Notes About Lambda Usage With RAJA
----------------------------------------

There are several issues to note about using C++ lambda expressions to
represent kernel bodies with RAJA. We describe them here.

 * **Prefer by-value lambda capture.**

   We recommend `capture by-value` for all lambda kernel bodies passed to
   RAJA execution methods. To execute a RAJA loop on a non-CPU device, such
   as a GPU, all variables accessed in the loop body must be passed into the
   GPU device data environment. Using capture by-value for all RAJA-based
   lambda usage will allow your code to be portable for either CPU or GPU
   execution. In addition, the read-only nature of variables captured
   by-value can help avoid incorrect CPU code since the compiler will report
   incorrect usage.

|br|

 * **The '__device__' annotation is required for device execution using CUDA or HIP.**

   Any lambda passed to a CUDA or HIP execution context (or function called from a
   device kernel, for that matter) must be decorated with
   the ``__device__`` annotation; for example::

     RAJA::forall<RAJA::cuda_exec<BLOCK_SIZE>>( range, [=] __device__ (int i) { ... } );

   Without this, the code will not compile and generate compiler errors
   indicating that a 'host' lambda cannot be called in 'device' code.

   RAJA provides the macro ``RAJA_DEVICE`` that can be used to help switch
   between host-only or device-only compilation.

|br|

 * **Use 'host-device' annotation on a lambda carefully.**

   RAJA provides the macro ``RAJA_HOST_DEVICE`` to support the dual
   annotation ``__ host__ __device__``, which makes a lambda or function
   callable from CPU or GPU device code. However, when CPU performance is
   important, **the host-device annotation should be applied carefully on a
   lambda that is used in a host (i.e., CPU) execution context**. Although
   compiler improvements in recent years have significantly
   improved support for host-device lambda expressions, a loop kernel
   containing a lambda annotated in this way may run noticeably slower on
   a CPU than the same lambda with no annotation depending on the version of
   the compiler (e.g., nvcc) you are using. To be sure that your code does not
   suffer in performance, we recommend comparing CPU execution timings of
   important kernels with and without the ``__host__ __device__`` annotation.

|br|

 * **Cannot use 'break' and 'continue' statements in a lambda.**

   In this regard, a lambda expression is similar to a function. So, if you
   have loops in your code with these statements, they should be rewritten.

|br|

 * **Global variables are not captured in a lambda.**

   This fact is due to the C++ standard. If you need access to a
   global variable inside a lambda expression, one solution is to make a local
   reference to it; for example::

     double& ref_to_global_val = global_val;

     RAJA::forall<RAJA::cuda_exec<BLOCK_SIZE>>( range, [=] __device__ (int i) {
       // use ref_to_global_val
     } );

|br|

 * **Local stack arrays may not be captured by CUDA device lambdas.**

   Although this is inconsistent with the C++ standard (local stack arrays
   are properly captured in lambdas for code that will execute on a CPU),
   attempting to access elements in a local stack array in a CUDA device
   lambda may generate a compilation error depending on the version of the
   device compiler you are using. One solution to this problem is to wrap the
   array in a struct; for example::

     struct array_wrapper {
       int[4] array;
     } bounds;

     bounds.array = { 0, 1, 8, 9 };

     RAJA::forall<RAJA::cuda_exec<BLOCK_SIZE>>(range, [=] __device__ (int i) {
       // access entries of bounds.array
     } );

   This issue was resolved in the 10.1 release of CUDA. If you are using an
   earlier version, an implementation similar to the one above will be required.

.. |br| raw:: html

   <br />

===========================
RAJA Examples and Exercises
===========================

The remainder of this tutorial illustrates how to use RAJA features with
working code examples and interactive exercises. Files containing the
exercise source code are located in  the ``RAJA/exercises`` directory.
Additional information about the RAJA features used can be found
in :ref:`features-label`.

The examples demonstrate CPU execution (sequential and OpenMP
multithreading) and GPU execution (CUDA and/or HIP). Examples that show how
to use RAJA with other parallel programming model back-ends will appear in
future RAJA releases. For adventurous users who wish to try experimental
RAJA back-end support, usage is similar to what is shown in the
examples here.

All RAJA programming model support features are enabled via CMake options,
which are described in :ref:`configopt-label`.

.. _tutorialbasic-label:

=====================================
Simple Loops and Basic RAJA Features
=====================================

The examples in this section illustrate how to use ``RAJA::forall`` methods
to execute simple loop kernels; i.e., non-nested loops. It also describes
iteration spaces, reductions, atomic operations, scans, sorts, and RAJA
data views.

.. toctree::
   :maxdepth: 1

   tutorial/add_vectors.rst
   tutorial/indexset_segments.rst
   tutorial/vertexsum_coloring.rst
   tutorial/dot_product.rst
   tutorial/reductions.rst
   tutorial/atomic_histogram.rst
   tutorial/scan.rst
   tutorial/sort.rst
   tutorial/view_layout.rst
   tutorial/permuted-layout-batch-matrix-multiply.rst

.. _tutorialcomplex-label:

=================================================================
Complex Loops and Advanced RAJA Features
=================================================================

RAJA provides two APIs for writing complex kernels involving nested
loops: ``RAJA::kernel`` that has been available for several years and
``RAJA::launch``, which is more recent. We briefly introduce both interfaces
here. The tutorial sections that follow provide much more detailed descriptions.

``RAJA::kernel`` is analogous to ``RAJA::forall`` in that it involves
kernel execution templates, execution policies, iteration spaces, and lambda
expression kernel bodies. The main differences between ``RAJA::kernel`` and
``RAJA::forall`` are:

  * ``RAJA::kernel`` requires a tuple of iteration spaces, one for each level
    in a loop nest, whereas ``RAJA::forall`` takes exactly one iteration
    space.
  * ``RAJA::kernel`` can accept multiple lambda expressions to express
    different parts of a kernel body, whereas ``RAJA::forall`` accepts
    exactly one lambda expression for a kernel body.
  * ``RAJA::kernel`` execution policies are more complicated than those
    for ``RAJA::forall``. ``RAJA::forall`` policies essentially represent
    the kernel execution back-end only. ``RAJA::kernel`` execution policies
    enable complex compile time algorithm transformations to be done without
    changing the kernel code.

The following exercises illustrate the common usage of ``RAJA::kernel``
and ````RAJA::launch``. Please see :ref:`loop_elements-kernelpol-label`
for more information about other execution policy constructs ``RAJA::kernel``
provides. ``RAJA::launch`` takes a ``RAJA::LaunchParams`` type argument for
representing a teams-thread launch configuration, and a lambda expression
which takes a ``RAJA::launchContext`` argument. ``RAJA::launch``
allows an optional run time choice of execution environment, either CPU or GPU.
Code written inside the lambda expression body will execute in the chosen
execution environment. Within that environment, a user executes
kernel operations using ``RAJA::loop<EXEC_POL>`` method calls, which
take lambda expressions to express loop body operations.

.. note:: A key difference between the ``RAJA::kernel`` and
          ``RAJA::launch`` approaches is that almost all of the
          kernel execution pattern is expressed in the execution policy
          when using ``RAJA::kernel``, whereas with ``RAJA::launch`` the
          kernel execution pattern is expressed mostly in the lambda
          expression kernel body.

One may argue that ``RAJA::kernel`` is more portable and flexible in that
the execution policy enables compile time code transformations without
changing kernel body code. On the other hand, ``RAJA::launch`` is
less opaque and more intuitive, but may require kernel body code changes for
algorithm changes. Which interface to use depends on personal preference
and other concerns, such as portability requirements, the need for run time
execution selection, etc. Kernel structure is more explicit in application
source code with ``RAJA::launch``, and more concise and arguably more
opaque with ``RAJA::kernel``. There is a large overlap of algorithms that can
be expressed with either interface. However, there are things that one can do
with one or the other but not both.

In the following sections, we introduce the basic mechanics and features
of both APIs with examples and exercises. We also present a sequence of
execution policy examples and matrix transpose examples using both
``RAJA::kernel`` and ``RAJA::launch`` to compare and contrast the
two interfaces.

===========================================================================
Nested Loops with ``RAJA::kernel``
===========================================================================

The examples in this section illustrate various features of the
``RAJA::kernel`` API used to execute nested loop kernels. It describes how to
construct kernel execution policies and use different view types and tiling
mechanisms to transform loop patterns. More information can be found in
:ref:`loop_elements-kernel-label`.

.. toctree::
   :maxdepth: 1

   tutorial/kernel_nested_loop_reorder.rst
   tutorial/kernel_exec_pols.rst
   tutorial/offset-layout-5pt-stencil.rst

=================================================================
Nested Loops with ``RAJA::launch``
=================================================================

The examples in this section illustrate how to use ``RAJA::launch``
to create an run time selectable execution space for expressing algorithms
as nested loops.

.. toctree::
   :maxdepth: 1

   tutorial/launch_basic.rst
   tutorial/launch_exec_pols.rst
   tutorial/launch_naming_kernels.rst

.. _tutorialmatrixtranspose-label:

===============================================================================
Comparing ``RAJA::kernel`` and ``RAJA::launch``: Matrix-Transpose
===============================================================================

In this section, we compare ``RAJA::kernel`` and ``RAJA::launch``
implementations of a matrix transpose algorithm. We illustrate
implementation differences of the two interfaces as we build upon each
example with more complex features.

.. toctree::
   :maxdepth: 1

   tutorial/matrix_transpose.rst
   tutorial/matrix_transpose_tiled.rst
   tutorial/matrix_transpose_local_array.rst

==========================================
Other RAJA Features and Usage Examples
==========================================

.. toctree::
   :maxdepth: 1

   tutorial/halo-exchange.rst
   tutorial/matrix_multiply.rst

.. important:: The `RAJA Performance Suite <https://github.com/LLNL/RAJAPerf>`
               is an excellent source of RAJA usage examples. The Suite
               contains many numerical kernels, each of which is implemented
               in a variety of RAJA and non-RAJA variants in OpenMP, CUDA, HIP,
               SYCL, etc. Comparing different variants of these kernels is
               instructive to understand how to use RAJA features and how they
               work.
