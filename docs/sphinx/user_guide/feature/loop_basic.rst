.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _loop_elements-label:

==============================================
Elements of Loop Execution
==============================================

The ``RAJA::forall``, ``RAJA::expt::dynamic_forall``, ``RAJA::kernel``, and ``RAJA::launch``
template methods comprise the RAJA interface for kernel
execution. ``forall`` methods execute simple, non-nested loops,
``RAJA::kernel`` methods support nested loops and other complex loop
kernels and transformations, and ``RAJA::launch`` creates an execution
space in which kernels are written in terms of nested loops using
the ``RAJA::loop`` method.

.. note:: The ``forall`` ,  ``kernel``, and ``launch`` methods are in the ``RAJA``
          namespace, while ``dynamic_forall`` is in the RAJA namespace for
          experimental features ``RAJA::expt``.  ``RAJA::expt::dynamic_forall``
          will be moved to the ``RAJA`` namespace in a future RAJA release.

For more information on RAJA execution policies and iteration space constructs,
see :ref:`feat-policies-label` and :ref:`feat-index-label`, respectively.

The following sections describe the basic aspects of these methods.
Detailed examples showing how to use ``RAJA::forall``, ``RAJA::kernel``, ``RAJA::launch`` methods may be found in the :ref:`tutorial-label`. Links to specific
RAJA tutorial sections are provided in the sections below.

.. _loop_elements-forall-label:

---------------------------
Simple Loops (RAJA::forall)
---------------------------

Consider a C-style loop that adds two vectors::

  for (int i = 0; i < N; ++i) {
    c[i] = a[i] + b[i];
  }

This may be written using ``RAJA::forall`` as::

  RAJA::forall<exec_policy>(RAJA::TypesRangeSegment<int>(0, N), [=] (int i) {
    c[i] = a[i] + b[i];
  });

A ``RAJA::forall`` loop execution method is a template that takes an
*execution policy* type template parameter. A ``RAJA::forall`` method takes
two arguments: an iteration space object, such as a contiguous range of loop
indices as shown here, and a single lambda expression representing the loop
kernel body.

Applying different loop execution policies enables the loop to run in
different ways; e.g., using different programming model back-ends. Different
iteration space objects enable the loop iterates to be partitioned, reordered,
run in different threads, etc. Please see :ref:`feat-index-label` for details
about RAJA iteration spaces.

.. note:: Changing loop execution policy types and iteration space constructs
          enables loops to run in different ways by recompiling the code and
          without modifying the loop kernel code.

As an extension of ``RAJA::forall``, the ``RAJA::expt::dynamic_forall`` method enables users
to compile using a list of execution policies and choose the execution policy at run-time.
For example, a user may want to have N policies available and at run-time choose which policy to use::

  using exec_pol_list = camp::list<pol_0,
				    ...
				   pol_N>;
  int pol = i; //run-time value

  RAJA::expt::dynamic_forall<exec_pol_list>(pol, RAJA::TypedRangeSegment<int>(0, N), [=] (int i) {
     c[i]  = a[i] + b[i];
  });


While static loop execution using ``forall`` methods is a subset of
``RAJA::kernel`` functionality, described next,
we maintain the ``forall`` interfaces for simple loop execution because the syntax is
simpler and less verbose for that use case.

.. note:: Data arrays in lambda expressions used with RAJA are typically
          RAJA Views (see :ref:`feat-view-label`) or bare pointers as shown in
          the code snippets above. Using something like 'std::vector' is
          non-portable (won't work in GPU kernels, generally) and would add
          excessive overhead for copying data into the lambda data environment
          when captured by value.

Please see the following tutorial sections for detailed examples that use
``RAJA::forall``:

 * :ref:`tut-addvectors-label`
 * :ref:`tut-dotproduct-label`
 * :ref:`tut-reduction-label`
 * :ref:`tut-atomichist-label`
 * :ref:`tut-indexset-label`
 * :ref:`tut-vertexsum-label`
 * :ref:`tut-permutedlayout-label`


.. _loop_elements-kernel-label:

----------------------------
Complex Loops (RAJA::kernel)
----------------------------

A ``RAJA::kernel`` template provides ways to compose and execute arbitrary
loop nests and other complex kernels.
The ``RAJA::kernel`` interface employs similar concepts to ``RAJA::forall``
but extends it to support much more complex kernel structures.
Each ``RAJA::kernel`` method is a template that takes an *execution policy*
type template parameter. The execution policy can be an arbitrarily complex
sequence of nested templates that define a kernel execution pattern.
In its simplest form, ``RAJA::kernel`` takes two arguments:
a *tuple* of iteration space objects, and a lambda expression representing
the kernel inner loop body. In more complex usage, ``RAJA::kernel`` can take
multiple lambda expressions representing different portions of the loop
kernel body.

To introduce the RAJA *kernel* interface, consider a (N+1)-level C-style loop
nest::

  for (int iN = 0; iN < NN; ++iN) {
    ...
       for (int i0 = 0; i0 < N0; ++i0) {s
         \\ inner loop body
       }
  }

It is important to note that we do not recommend writing a RAJA version of
this by nesting ``RAJA::forall`` statements. For example::

  RAJA::forall<exec_policyN>(IN, [=] (int iN) {
    ...
       RAJA::forall<exec_policy0>(I0, [=] (int i0)) {
         \\ inner loop body
       }
    ...
  }

This would work for some execution policy choices, but not in general.
Also, this approach treats each loop level as an independent entity, which
makes it difficult to parallelize the levels in the loop nest together. So it
may limit the amount of parallelism that can be exposed and the types of
parallelism that may be used. For example, if an OpenMP or CUDA
parallel execution policy is used on the outermost loop, then all inner loops
would be run sequentially in each thread. It also makes it difficult to perform
transformations like loop interchange and loop collapse without changing the
source code, which breaks RAJA encapsulation.

.. note:: **We do not recommend using nested ``RAJA::forall`` statements.**

The ``RAJA::kernel`` interface facilitates parallel execution and compile-time
transformation of arbitrary loop nests and other complex loop structures.
It can treat a complex loop structure as a single entity, which enables
the ability to transform and apply different parallel execution patterns by
changing the execution policy type and **not the kernel code**, in many cases.

The C-style loop above nest may be written using ``RAJA::kernel`` as::

    using KERNEL_POL =
      RAJA::KernelPolicy< RAJA::statement::For<N, exec_policyN,
                            ...
                              RAJA::statement::For<0, exec_policy0,
                                RAJA::statement::Lambda<0>
                              >
                            ...
                          >
                        >;

    RAJA::kernel< KERNEL_POL >(
      RAJA::make_tuple(RAJA::TypedRangeSegment<int>(0, NN),
                       ...,
                       RAJA::TypedRangeSegment<int>(0, N0),

      [=] (int iN, ... , int i0) {
         // inner loop body
      }

    );

In the case we discuss here, the execution policy contains a nested sequence
of ``RAJA::statement::For`` types, indicating an iteration over each level in
the loop nest.  Each of these statement types takes three template parameters:

  * an integral index parameter that binds the statement to the item
    in the iteration space tuple corresponding to that index
  * an execution policy type for the associated loop nest level
  * an *enclosed statement list* (described in :ref:`loop_elements-kernelpol-label`).

.. note:: The nesting of ``RAJA::statement::For`` types is analogous to the
          nesting of for-statements in the C-style version of the loop nest.
          One can think of the '<, >' symbols enclosing the template parameter
          lists as being similar to the curly braces in C-style code.

Here, the innermost type in the kernel policy is a
``RAJA::statement::Lambda<0>`` type indicating that the first lambda expression
(argument zero of a sequence of lambdas passed to the ``RAJA::kernel`` method)
will comprise the inner loop body. We only have one lambda in this example
but, in general, we can have any number of lambdas and we can use any subset
of them, with ``RAJA::statement::Lambda`` types placed appropriately in the
execution policy, to construct a loop kernel. For example, placing
``RAJA::statement::Lambda`` types between ``RAJA::statement::For`` statements
enables non-perfectly nested loops.

RAJA offers two types of ``RAJA::statement::Lambda`` statements. The simplest
form, shown above, requires that each lambda expression passed to a
``RAJA::kernel`` method **must take an index argument for each iteration
space.** With this type of lambda statement, the entire iteration space must
be active in a surrounding ``For`` construct.  A compile time ``static_assert``
will be triggered if any of the arguments are undefined, indicating that
something is not correct.

A second ``RAJA::statement::Lambda`` type, which is an extension of the first,
takes additional template parameters which specify which iteration spaces
are passed as lambda arguments. The result is that a kernel lambda only needs
to accept iteration space index arguments that are used in the lambda body.

The kernel policy list with lambda arguments may be written as::

    using KERNEL_POL =
      RAJA::KernelPolicy< RAJA::statement::For<N, exec_policyN,
                            ...
                              RAJA::statement::For<0, exec_policy0,
                                RAJA::statement::Lambda<0, RAJA::Segs<N,...,0>>
                              >
                            ...
                          >
                        >;

The template parameter ``RAJA::Segs`` is used to specify indices from which
elements in the segment tuple are passed as arguments to the lambda, and in
which argument order. Here, we pass all segment indices so the lambda kernel
body definition could be identical to on passed to the previous RAJA version.
RAJA offers other types such as ``RAJA::Offsets``, and ``RAJA::Params`` to
identify offsets and parameters in segments and parameter tuples that could be
passed to ``RAJA::kernel`` methods. See :ref:`tut-matrixmultiply-label`
for an example.

.. note:: Unless lambda arguments are specified in RAJA lambda statements,
          the loop index arguments for each lambda expression used in a RAJA
          kernel loop body **must match** the contents of the
          *iteration space tuple* in number, order, and type. Not all index
          arguments must be used in a lambda, but they **all must appear**
          in the lambda argument list and **all must be in active loops** to be
          well-formed. In particular, your code will not compile if this is
          not done correctly. If an argument is unused in a lambda expression,
          you may include its type and omit its name in the argument list to
          avoid compiler warnings just as one would do for a regular C++
          method with unused arguments.

For RAJA nested loops implemented with ``RAJA::kernel``, as shown here, the
loop nest ordering is determined by the order of the nested policies, starting
with the outermost loop and ending with the innermost loop.

.. note:: The integer value that appears as the first parameter in each
          ``RAJA::statement::For`` template indicates which iteration space
          tuple entry or lambda index argument it corresponds to. **This
          allows loop nesting order to be changed simply by changing the
          ordering of the nested policy statements**. This is analogous to
          changing the order of 'for-loop' statements in C-style nested loop
          code.

.. note:: In general, RAJA execution policies for ``RAJA::forall`` and
          ``RAJA::kernel`` are different. A summary of all RAJA execution
          policies that may be used with ``RAJA::forall`` or ``RAJA::kernel``
          may be found in :ref:`feat-policies-label`.

A discussion of how to construct ``RAJA::KernelPolicy`` types and
available ``RAJA::statement`` types can be found in
:ref:`loop_elements-kernelpol-label`.

Please see the following tutorial sections for detailed examples that use
``RAJA::kernel``:

 * :ref:`tut-kernelnestedreorder-label`
 * :ref:`tut-kernelexecpols-label`
 * :ref:`tut-matrixtranspose-label`
 * :ref:`tut-offsetlayout-label`
 * :ref:`tut-matrixmultiply-label`

------------------------------------------
Hierarchical loops (RAJA::launch)
------------------------------------------

The ``RAJA::launch`` template is an alternative interface to
``RAJA::kernel`` that may be preferred for certain types of complex kernels
or based on coding style preferences.

``RAJA::launch`` optionally allows either host or device execution
to be chosen at run time. The method takes an execution policy type that
will define the execution environment inside a lambda expression for a kernel
to be run on a host, device, or either. Kernel algorithms are written inside
main lambda expression using ``RAJA::loop`` methods.

The ``RAJA::launch`` framework aims to unify thread/block based
programming models such as CUDA/HIP/SYCL while maintaining portability on
host back-ends (OpenMP, sequential). As we showed earlier, when using the
``RAJA::kernel`` interface, developers express all aspects of nested loop
execution in an execution policy type on which the ``RAJA::kernel`` method
is templated.
In contrast, the ``RAJA::launch`` interface allows users to express
nested loop execution in a manner that more closely reflects how one would
write conventional nested C-style for-loop code. For example, here is an
example of a ``RAJA::launch`` kernel that copies values from an array in
into a *shared memory* array::

  RAJA::launch<launch_policy>(select_CPU_or_GPU)
  RAJA::LaunchParams(RAJA::Teams(NE), RAJA::Threads(Q1D)),
  [=] RAJA_HOST_DEVICE (RAJA::Launch ctx) {

    RAJA::loop<team_x> (ctx, RAJA::RAJA::TypedRangeSegment<int>(0, teamRange), [&] (int bx) {

      RAJA_TEAM_SHARED double s_A[SHARE_MEM_SIZE];

      RAJA::loop<thread_x> (ctx, RAJA::RAJA::TypedRangeSegment<int>(0, threadRange), [&] (int tx) {
        s_A[tx] = tx;
      });

        ctx.teamSync();

   )};

  });

The idea underlying ``RAJA::launch`` is to enable developers to express
hierarchical parallelism in terms of teams and threads. Similar to the CUDA
programming model, development is done using a collection of threads, and
threads are grouped into teams. Using the ``RAJA::loop`` methods
iterations of the loop may be executed by threads or teams depending on the
execution policy type. The launch context serves to synchronize threads within
the same team. The ``RAJA::launch`` interface has three main concepts:

  * ``RAJA::launch`` template. This creates an execution environment in
    which a kernel implementation is written using nested ``RAJA::loop``
    statements. The launch policy template parameter used with the
    ``RAJA::launch`` method enables specification of both a host and
    device execution environment, which enables run time selection of
    kernel execution.

  * ``RAJA::LaunchParams`` type. This type takes a number of teams, threads
    per team, and optionally the size of dynamic shared memory in bytes.

  * ``RAJA::loop`` template. These are used to define hierarchical
    parallel execution of a kernel. Operations within a loop are mapped to
    either teams or threads based on the execution policy template parameter
    provided.

Team shared memory can be allocated by using the ``RAJA_TEAM_SHARED`` macro on
statically sized arrays or via dynamic allocation in the ``RAJA::LaunchParams``
method. Team shared memory enables threads in a given team to have shared access to a shared memory buffer.
Loops are then assigned to either teams or threads based on the GPU execution
policy. Under the CUDA/HIP nomenclature, teams correspond to blocks while,
in SYCL nomenclature, teams correspond to workgroups.

In a host execution environment, team and thread parameters in the
``RAJA::LaunchParams`` struct have no effect in execution and may be
omitted if only running on the host.


Please see the following tutorial sections for detailed examples that use
``RAJA::launch``:

 * :ref:`tut-launchintro-label`
 * :ref:`tut-launchexecpols-label`
 * :ref:`tut-matrixtranspose-label`

.. _loop_elements-CombiningAdapter-label:

------------------------------------------------------------------------
Multi-dimensional loops using simple loop APIs (RAJA::CombiningAdapter)
------------------------------------------------------------------------

A ``RAJA::CombiningAdapter`` object provides ways to run perfectly nested loops
with simple loop APIs like ``RAJA::forall`` and those described in
:ref:`workgroup-label`.
To introduce the ``RAJA ::CombiningAdapter`` interface, consider a (N+1)-level
C-style loop nest::

  for (int iN = 0; iN < NN; ++iN) {
    ...
       for (int i0 = 0; i0 < N0; ++i0) {
         \\ inner loop body
       }
  }

We can use a ``RAJA::CombiningAdapter`` to combine the iteration spaces of the
loops and pass the adapter to a ``RAJA::forall`` statement to execute them::

  auto adapter = RAJA::make_CombingingAdapter(
      [=] (int iN, ..., int i0)) {
        \\ inner loop body
      }, IN, ..., I0);

  RAJA::forall<exec_policy>(adapter.getRange(), adapter);

A ``RAJA::CombiningAdapter`` object is a template combining a loop body and
iteration spaces. The ``RAJA::make_CombingingAdapter`` template method takes
a lambda expression for the loop body and an arbitrary number of index
arguments. It provides a *flattened* iteration space via the ``getRange``
method that can be passed as the iteration space to the ``RAJA::forall``
method, for example. The object's call operator does the conversion of the
flat single dimensional index into the multi-dimensional index space, calling
the provided lambda with the appropriate indices.

.. note:: CombiningAdapter currently only supports
          ``RAJA::TypedRangeSegment`` segments.
