.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _tut-launchperfectloop-label:

-----------------------------------------------------------
``RAJA::launch`` Perfectly Nested Loops
-----------------------------------------------------------

This section introduces a compact way to express perfectly nested launch loops
with a single lambda body. The example source file
``RAJA/examples/tut_launch_perfect_loop.cpp`` shows both host and device
dispatch through the run-time ``RAJA::ExecPlace`` API.

Key RAJA features shown in this section are:

  * ``RAJA::PerfectLoopPolicy`` to group many loop mappings into one policy.
  * ``RAJA::perfect_loop`` to execute a perfectly nested loop tree with a
    single lambda.
  * ``RAJA::make_multi_range`` when you want to package the ranges explicitly.

The basic pattern is::

  using perfect_loop_policy = RAJA::PerfectLoopPolicy<
      teams_y,
      teams_x,
      threads_y,
      threads_x>;

  RAJA::launch<launch_policy>(place, params, [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
    RAJA::perfect_loop<perfect_loop_policy>(
        ctx,
        team_y_range,
        team_x_range,
        thread_y_range,
        thread_x_range,
        [=](int team_y, int team_x, int thread_y, int thread_x) {
          /* loop body */
        });
  });

Compared to spelling out each nested ``RAJA::loop`` call directly, this form
keeps the full loop mapping in one policy alias and moves the algorithm body
into a single lambda. That is especially useful when the loop nest is deep or
when the same loop structure is used across multiple kernels.

If you need to pass the ranges around as one object, ``RAJA::make_multi_range``
is still available. ``RAJA::PerfectLoopPolicy`` accepts the matching list of
loop policies, in the same order. RAJA also provides
``RAJA::perfect_loop_icount`` when the body needs both the value and the loop
counter for each level.

.. literalinclude:: ../../../../examples/tut_launch_perfect_loop.cpp
   :start-after: // _launch_perfect_loop_start
   :end-before: // _launch_perfect_loop_end
   :language: C++

Loop interchange is available through ``RAJA::PerfectLoopInterchange``. The
interchange controls execution order while the lambda argument order remains
aligned with the original segment list::

  using interchange = RAJA::PerfectLoopInterchange<1, 2, 0>;

  RAJA::perfect_loop_icount<perfect_loop_policy, interchange>(
      ctx,
      i_range,
      j_range,
      k_range,
      [=](int i, int j, int k, int ii, int jj, int kk) { /* body */ });

.. literalinclude:: ../../../../examples/tut_launch_perfect_loop_interchange.cpp
   :start-after: // _launch_perfect_loop_interchange_start
   :end-before: // _launch_perfect_loop_interchange_end
   :language: C++
