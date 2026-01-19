.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _workgroup-label:

=========
WorkGroup
=========

In this section, we describe the basics of RAJA workgroups.
``RAJA::WorkPool``, ``RAJA::WorkGroup``, and ``RAJA::WorkSite`` class templates comprise the
RAJA interface for grouped loop execution. ``RAJA::WorkPool`` takes a set  of simple
loops (e.g., non-nested loops) and instantiates a ``RAJA::WorkGroup``. ``RAJA::WorkGroup``
represents an executable form of those loops and when run makes a ``RAJA::WorkSite``.
``RAJA::WorkSite`` holds all of the resources used for a single run of the loops. Be aware
that the RAJA workgroup constructs API is still being developed and may change in later RAJA
releases.

.. note:: * All workgroup constructs are in the namespace ``RAJA``.
          * The ``RAJA::WorkPool``, ``RAJA::WorkGroup``, and ``RAJA::WorkSite`` class templates are templated on:
              * a WorkGroup policy which is composed of:
                  * a work execution policy
                  * a work ordering policy
                  * a work storage policy
                  * a work dispatch policy
              * an index type that is the first argument to the loop bodies.
              * a list of extra argument types that are the rest of the arguments to the loop bodies.
              * an allocator type to be used for the memory used to store and
                manage the loop bodies.
          * The ``RAJA::WorkPool::enqueue`` method takes two arguments:
              * an iteration space object, and
              * a lambda expression representing the loop body.
          * Multi-dimensional loops can be used with ``RAJA::CombiningAdapter``
            see, :ref:`loop_elements-CombiningAdapter-label`.

Examples showing how to use RAJA workgroup methods may be found in
the :ref:`tutorial-label`.

For more information on RAJA work policies and iteration space constructs,
see :ref:`feat-policies-label` and :ref:`feat-index-label`, respectively.

.. _workgroup-Policies-label:

--------
Policies
--------

The behavior of the RAJA workgroup constructs is determined by a policy.
The ``RAJA::WorkGroupPolicy`` has four components, a work execution policy,
a work ordering policy, a work storage policy, and a work dispatch policy.
``RAJA::WorkPool``, ``RAJA::WorkGroup``, and ``RAJA::WorkSite`` class templates
all take the same policy and template arguments.  For example::

  using workgroup_policy = RAJA::WorkGroupPolicy <
                               RAJA::seq_work,
                               RAJA::ordered,
                               RAJA::ragged_array_of_objects,
                               RAJA::indirect_function_call_dispatch >;

is a workgroup policy that will run loops sequentially on the host in the order
they were enqueued, stores the loop bodies sequentially in single buffer in
memory, and dispatches each loop using a function pointer.

The work execution policy acts like the execution policies used with ``RAJA::forall``
and determines the backend used to run the loops and the parallelism within each
loop.

 ====================================== ========================================
 Work Execution Policies                Brief description
 ====================================== ========================================
 seq_work                               Execute loop iterations strictly
                                        sequentially.
 simd_work                              Execute loop iterations sequentially and
                                        try to force generation of SIMD
                                        instructions via compiler hints in RAJA
                                        internal implementation.
 seq_work                               Execute loop iterations sequentially and
                                        allow compiler to generate any
                                        optimizations.
 omp_work                               Execute loop iterations in parallel
                                        using OpenMP.
 cuda_work<BLOCK_SIZE>,                 Execute loop iterations in parallel
 cuda_work_async<BLOCK_SISZE>           using a CUDA kernel launched with given
                                        thread-block size.
 omp_target_work                        Execute loop iterations in parallel
                                        using OpenMP target.
 ====================================== ========================================

The work ordering policy acts like the segment iteration execution policies when
``RAJA::forall`` is used with a ``RAJA::IndexSet`` and determines the backend
used when iterating over the loops and the parallelism between each loop.

 ======================================================= ========================================
 Work Ordering Policies                                  Brief description
 ======================================================= ========================================
 ordered                                                 Execute loops sequentially in the order
                                                         they were enqueued using forall.
 reverse_ordered                                         Execute loops sequentially in the
                                                         reverse of the order order they were
                                                         enqueued using forall.
 unordered_cuda_loop_y_block_iter_x_threadblock_average  Execute loops in parallel by mapping
                                                         each loop to a set of cuda blocks with
                                                         the same index in the y direction in
                                                         a cuda kernel. Each loop is given a
                                                         number of threads over one of more
                                                         blocks in the x direction equal to the
                                                         average number of iterations of all the
                                                         loops rounded up to a multiple of the
                                                         block size.
 ======================================================= ========================================

The work storage policy determines the strategy used to allocate and layout the
storage used to store the ranges, loop bodies, and other data necessary to
implement the workstorage constructs.

 ====================================== ========================================
 Work Storage Policies                  Brief description
 ====================================== ========================================
 array_of_pointers                      Store loop data in individual
                                        allocations and keep an array of
                                        pointers to the individual loop data
                                        allocations.
 ragged_array_of_objects                Store loops sequentially in a single
                                        allocation, reallocating and moving the
                                        loop data items as needed, and keep an
                                        array of offsets to the individual loop
                                        data items.
 constant_stride_array_of_objects       Store loops sequentially in a single
                                        allocation with a consistent stride
                                        between loop data items, reallocating
                                        and/or changing the stride and moving
                                        the loop  data items as needed.
 ====================================== ========================================

The work dispatch policy determines the technique used to dispatch from type
erased storage to the loops or iterations of each range and loop body pair.

 ====================================== ========================================
 Work Dispatch Policies                 Brief description
 ====================================== ========================================
 indirect_function_call_dispatch        Dispatch using function pointers.
 indirect_virtual_function_dispatch     Dispatch using virtual functions in a
                                        class hierarchy.
 direct_dispatch<                       Dispatch using a switch statement like
     camp::list<Range, Callable>...>    coding to pick the right pair of
                                        Range and Callable types from the
                                        template parameter pack. You may only
                                        enqueue a range and callable pair that
                                        is in the list of types in the policy.
 ====================================== ========================================


.. _workgroup-Arguments-label:

---------
Arguments
---------

The next two template arguments to the workgroup constructs determine the
call signature of the loop bodies that may be added to the workgroup. The first
is an index type which is the first parameter in the call signature. Next is a
list of types called ``RAJA::xargs``, short for extra arguments, that gives the
rest of the types of the parameters in the call signature. The values of the
extra arguments are passed in when the loops are run, see :ref:`workgroup-WorkGroup-label`.
For example::

  int, RAJA::xargs<>

can be used with lambdas with the following signature::

  [=](int) { ... }

and::

  int, RAJA::xargs<int*, double>

can be used with lambdas with the following signature::

  [=](int, int*, double) { ... }


.. _workgroup-Allocators-label:

----------
Allocators
----------

The last template argument to the workgroup constructs is an allocator type
that conforms to the allocator named requirement used in the standard library.
This gives you control over how memory is allocated, for example with umpire,
and what memory space is used, both of which have poerformance implications.
Find the requirements for allocator types along with a simple example here
https://en.cppreference.com/w/cpp/named_req/Allocator. The default allocator
used by the standard template library may be used with ordered and non-GPU
policies::

  using Allocator = std::allocator<char>;

.. note:: * The allocator type must use template argument ``char``.
          * Allocators must provide memory that is accessible where it is used.
              * Ordered work order policies only require memory that is accessible
                where loop bodies are enqueued.
              * Unordered work order policies require memory that is accessible
                from both where the loop bodies are enqueued and from where the
                loop is executed based on the work execution policy.

                  For example, when using cuda work exeution policies with CUDA
                  unordered work order policies, pinned memory is a good choice
                  because it is always accessible on the host and device.


.. _workgroup-WorkPool-label:

--------
WorkPool
--------

The ``RAJA::WorkPool`` class template holds a set of simple (e.g., non-nested)
loops that are enqueued one at a time. Note that simple multi-dimensional loops
can be adapted into simple loops via ``RAJA::CombiningAdapter``, see
:ref:`loop_elements-CombiningAdapter-label`.
For example, to enqueue a C-style loop that adds two vectors, like::

  for (int i = 0; i < N; ++i) {
    c[i] = a[i] + b[i];
  }

is as simple as calling enqueue on a ``RAJA::WorkPool`` object and passing the
same arguments you would pass to ``RAJA::forall``.::

  using WorkPool_type = RAJA::WorkPool< workgroup_policy,
                                        int, RAJA::xargs<>,
                                        Allocator >;
  WorkPool_type workpool(Allocator{});

  workpool.enqueue(RAJA::RangeSegment(0, N), [=] (int i) {
    c[i] = a[i] + b[i];
  });

Note that WorkPool may have to allocate and reallocate multiple times to store
a set of loops depending on the work storage policy. Reallocation can be avoided
by reserving enough memory before adding any loops.::

  workpool.reserve(num_loops, storage_bytes);

Here ``num_loops`` is the number of loops to allocate space for and
``num_storage_bytes`` is the amount of storage to allocate. These may be used
differently depending on the work storage policy. The number of loops
enqueued in a ``RAJA::WorkPool`` and the amount of storage used may be queried
using::

  size_t num_loops     = workpool.num_loops();
  size_t storage_bytes = workpool.storage_bytes();

Storage will automatically reserved when reusing a `RAJA::WorkPool`` object
based on the maximum seen values for num_loops and storage_bytes.

When you've added all the loops you want to the set, you can call instantiate
on the ``RAJA::WorkPool`` to generate a ``RAJA::WorkGroup``.::

  WorkGroup_type workgroup = workpool.instantiate();

.. _workgroup-WorkGroup-label:

---------
WorkGroup
---------

The ``RAJA::WorkGroup`` class template is responsible for hanging onto the set
of loops and running the loops. The ``RAJA::WorkGroup`` owns its loops and must
not be destroyed before any loops run asynchronously using it have completed.
It is instantiated from a ``RAJA::WorkPool`` object which transfers ownership
of a set of loops to the ``RAJA::WorkGroup`` and prepares the loops to be run.
For example::

  using WorkGroup_type = RAJA::WorkGroup< workgroup_policy,
                                          int, RAJA::xargs<>,
                                          Allocator >;
  WorkGroup_type workgroup = workpool.instantiate();

creates a ``RAJA::WorkGroup`` ``workgroup`` from the loops in ``workpool`` and
leaves ``workpool`` empty and ready for reuse. When you want to run the loops
simply call run on ``workgroup`` and pass in the extra arguments::

  WorkSite_type worksite = workgroup.run();

In this case no extra arguments were passed to run because the ``RAJA::WorkGroup``
specified no extra arguments ``RAJA::xargs<>``. Passing extra arguments when the
loops are run lets you delay creation of those arguments until you plan to run
the loops. This lets the value of the arguments depend on the loops in the set.
A simple example of this may be found in the tutorial here :ref:`tutorial-label`.
Run produces a ``RAJA::WorkSite`` object.


.. _workgroup-WorkSite-label:

--------
WorkSite
--------

The ``RAJA::WorkSite`` class template is responsible for extending the lifespan
of objects used when running loops asynchronously. This means that the
``RAJA::WorkSite`` object must remain alive until the call to run has been
synchronized. For example the scoping here::

  {
    using WorkSite_type = RAJA::WorkSite< workgroup_policy,
                                          int, RAJA::xargs<>,
                                          Allocator >;
    WorkSite_type worksite = workgroup.run();

    // do other things

    synchronize();
  }

ensures that ``worksite`` survives until after synchronize is called.
