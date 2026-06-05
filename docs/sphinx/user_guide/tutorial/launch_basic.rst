.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _tut-launchintro-label:

------------------------------
``RAJA::Launch`` Basics
------------------------------

There are no exercise files to work through for this section. Instead, there
is an example source file ``RAJA/examples/tut_launch_basic.cpp`` which
contains complete code examples of the concepts described here.

Key RAJA features shown in the following examples are:

  * ``RAJA::launch`` method to create a run-time
    selectable host/device execution space.
  * ``RAJA::loop`` methods to express algorithms
    in terms of nested for loops.

In this example, we introduce the RAJA Launch framework and discuss
hierarchical loop-based parallelism. Kernel execution details 
with RAJA Launch occur inside the lambda expression
passed to the ``RAJA::launch`` method, which defines an execution
space::

  RAJA::launch<launch_policy>(RAJA::ExecPlace ,
  RAJA::LaunchParams(RAJA::Teams(Nteams,Nteams),
                     RAJA::Threads(Nthreads,Nthreads)),
  [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx) {

    /* Kernel code goes here */

  });

The ``RAJA::launch`` method accepts a ``RAJA::LaunchPolicy``
template parameter that can be defined using up to two policies 
(host and device). For example, the following creates an execution space 
for a sequential and CUDA kernel dispatch::

  using launch_policy = RAJA::LaunchPolicy
    <RAJA::seq_launch_t, RAJA::cuda_launch_t<false>>;

Whether a kernel executes on the host or device is determined by the first 
argument passed to the ``RAJA::launch`` method, which is a 
``RAJA::ExecPlace`` enum value, either ``HOST`` or ``DEVICE``.
Similar to GPU thread and block programming models, RAJA Launch carries out
computation in a predefined compute grid made up of threads which are
then grouped into teams when executing on the device. The execution space is 
then enclosed by a host/device lambda which takes a 
``RAJA::LaunchContext`` object, which may be used to control the flow 
within the kernel, for example by creating thread-team synchronization points.

Inside the execution space, developers write a kernel using nested
``RAJA::loop`` methods. The manner in which each loop is executed 
is determined by a template parameter type, which
indicates how the corresponding iterates are mapped to the Teams/Threads
configuration defined by the ``RAJA::LaunchParams`` type passed as the second
argument to the ``RAJA::launch`` method. Following the CUDA and HIP 
programming models, this defines an hierarchical structure in which outer loops 
are executed by thread-teams and inner loops are executed by threads in a team.

.. literalinclude:: ../../../../examples/tut_launch_basic.cpp
   :start-after: // _team_loops_start
   :end-before: // _team_loops_end
   :language: C++

The mapping between teams and threads to the underlying programming 
model depends on how the ``RAJA::loop`` template parameter types are
defined. For example, we may define host and device mapping strategies as::

  using teams_x = RAJA::LoopPolicy< RAJA::seq_exec,
                                    RAJA::cuda_block_x_direct >;
  using thread_x = RAJA::LoopPolicy< RAJA::seq_exec,
                                     RAJA::cuda_block_x_direct >;

Here, the ``RAJA::LoopPolicy`` type holds both the host (CPU) and 
device (CUDA GPU) loop mapping strategies. On the host, both the team/thread 
strategies expand out to standard C-style loops for execution:

.. literalinclude:: ../../../../examples/tut_launch_basic.cpp
   :start-after: // _c_style_loops_start
   :end-before: // _c_style_loops_end
   :language: C++

On the device the ``teams_x/y`` policies will map loop iterations directly to
CUDA (or HIP) thread blocks, while the ``thread_x/y`` policies will map loop 
iterations directly to threads in a CUDA (or HIP) thread block. The direct CUDA 
equivalent of the kernel body using the policy shown above is:

.. literalinclude:: ../../../../examples/tut_launch_basic.cpp
   :start-after: // _device_loop_start
   :end-before: // _device_loop_end
   :language: C++
