.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _feat-jit-label:

===============================
JIT Compilation (Proteus + RAJA)
===============================

RAJA can optionally integrate with `Proteus <https://github.com/Olympus-HPC/proteus>`_
to *just-in-time (JIT) compile* specialized variants of kernels. This is useful when
some performance-critical values are only known at runtime.  For example, propagating
loop bounds as runtime constants can enhance loop analysis and scheduling.  Propagating
other values can enable optimizations like branch elimination, etc.

.. warning:: This capability is new and should be considered experimental.

-----------------------------
Enabling JIT in a RAJA build
-----------------------------

JIT support is enabled at configuration time:

.. code-block:: bash

  cmake -DRAJA_ENABLE_JIT=On ...

Enabling ``RAJA_ENABLE_JIT`` requires an LLVM-based (Clang-family) compiler.
If you enable JIT and configure with a non-Clang compiler, configuration will
fail. See :ref:`configopt-label` for the CMake options described here.

Proteus dependency
^^^^^^^^^^^^^^^^^^

When ``RAJA_ENABLE_JIT=On``, RAJA needs Proteus headers and build integration:

* If you provide ``-DPROTEUS_INSTALL_DIR=<prefix>``, RAJA will use
  ``find_package(proteus ...)`` using that prefix.
* Otherwise, RAJA will attempt to fetch Proteus via CMake ``FetchContent`` at
  configure time.

LLVM installation requirement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unless you provide a Proteus installation that is statically linked with LLVM,
Proteus support requires an LLVM 18, 19, or 20 installation that you must point
RAJA/Proteus at via ``LLVM_INSTALL_DIR``:

.. code-block:: bash

  cmake -DRAJA_ENABLE_JIT=On -DLLVM_INSTALL_DIR=/path/to/llvm-19 ...
  
An example of how to configure a JIT build of RAJA with HIP on LC machines is included
in ``scripts/toss4_amdclang_proteus.sh``.

--------------------------
Marking a kernel for JIT
--------------------------

The user-facing interface shown in ``examples/forall-jit.cpp`` consists of:

* ``RAJA_JIT_COMPILE``: annotate a lambda or function so Proteus can identify it
  as a JIT compilation candidate.
* ``RAJA_JIT_VARIABLE``: wrap runtime values that should be treated as
  constants for specialization.

For example, specializing loop bounds and a branch condition:

.. literalinclude:: ../../../../examples/forall-jit.cpp
    :start-after: _raja_jit_mark_start
    :end-before: _raja_jit_mark_end
    :language: C++

When JIT is disabled (``RAJA_ENABLE_JIT=Off``), ``RAJA_JIT_COMPILE`` expands to
nothing. Similarly, RAJA_JIT_VARIABLE expands to ``proteus::jit_variable`` when JIT
is enabled, but simply expands to its single argument with ``RAJA_ENABLE_JIT=Off``.
``proteus::enable()`` and ``proteus::disable()`` manually enable/disable Proteus within a
region of source code.  Currently, with ``RAJA_ENABLE_JIT=On``, all RAJA kernels
will be JIT compiled unless ``proteus::disable()`` is specified.

Building and running the example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example ``examples/forall-jit.cpp`` is built from the RAJA source tree when:

* ``ENABLE_EXAMPLES=On``
* ``RAJA_ENABLE_JIT=On``

The example takes four command-line arguments:

.. code-block:: bash

  ./bin/forall-jit <a> <b> <N> <accum>

where ``a`` and ``b`` are matrix dimensions, ``N`` is the problem size, and
``accum`` is the branch condition (0/1) that is specialized with JIT. 
The example performs N-many matrix multiplications (1 per thread).  Each multiplication
is [a x b] [b x a].  The result is either set or added into the output, depending
on the boolean flag the user provides.  By forcing each thread to perform serialized 
arithmetic with a simple branch condition, we show how JIT compilation can improve both 
serial loop scheduling (per-thread) and branch elimination.

Specializing argument indices (advanced)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RAJA also provides ``RAJA_JIT_COMPILE_ARGS(...)`` to annotate functions and
specify which 1-indexed arguments should be treated as specialization inputs:

.. code-block:: c++

  __global__ RAJA_JIT_COMPILE_ARGS(3) void my_kernel(int x, int y, int z) { ... }

This is used internally by RAJA's GPU back-ends; most users only need
``RAJA_JIT_COMPILE`` on their lambdas and ``proteus::jit_variable`` for values
captured into the lambda.
