.. ##
.. ## Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _configopt-label:

****************************
Build Configuration Options
****************************

RAJA uses `BLT <https://github.com/LLNL/blt>`_, a CMake-based build system.
In :ref:`getting_started-label`, we described how to run CMake to configure
RAJA with its default option settings. In this section, we describe RAJA
configuration options that are most useful for users to know about and
their defaults.

=============================
RAJA Option Types
=============================

RAJA contains two types of options, those that exist in 
RAJA only and those that are similar to standard CMake options or options 
provided by BLT; i.e., *dependent options* in CMake terminology. RAJA 
dependent option names are the same as the associated CMake and BLT option 
names, but with the ``RAJA_`` prefix added.

.. note:: RAJA uses a mix of RAJA-only options and CMake-dependent
          options that can be controlled with CMake or BLT variants. 

            * Dependent options are typically used for *disabling* features.
              For example, when the CMake option ``-DENABLE_TESTS=On`` is
              used to enable tests in the build of an application that includes
              multiple CMake-based package builds, providing the CMake option 
              ``-DRAJA_ENABLE_TESTS=Off`` will disable compilation of RAJA 
              tests, while compiling them for other packages.

            * We recommend using the option names without the ``RAJA_`` prefix,
              when available, to enable features at compile time to avoid 
              potential undesired behavior. For example, passing the option
              ``-DRAJA_ENABLE_CUDA=On`` to CMake will not enable CUDA because
              ``ENABLE_CUDA`` is off by default. So to enable CUDA, you need
              to pass the ``-DENABLE_CUDA=On`` option to CMake.

=======================
Setting Options
=======================

The RAJA configuration can be set using standard CMake variables along with
BLT and RAJA-specific variables. For example, to make a release build with 
some system default GNU compiler and then install the RAJA header files and
libraries in a specific directory location, you could do the following in 
the top-level RAJA directory::

    $ mkdir build-gnu-release
    $ cd build-gnu-release
    $ cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_INSTALL_PREFIX=../install-gnu-release ../
    $ make
    $ make install

Following CMake conventions, RAJA supports three build types: ``Release``, 
``RelWithDebInfo``, and ``Debug``. With CMake, compiler flags for each of
these build types are applied automatically and so you do not have to 
specify them. However, if you want to apply other compiler flags, you will
need to do that using appropriate CMake variables.

All RAJA options are set like regular CMake variables. RAJA settings for 
default options, compilers, flags for optimization, etc. can be found in files 
in the ``RAJA/cmake`` directory and top-level ``CMakeLists.txt`` file. 
Configuration variables can be set by passing arguments to CMake on the 
command line when calling CMake. For example, to enable RAJA OpenMP 
functionality, pass the following argument to CMake::

    cmake ... \
    -DENABLE_OPENMP=On \
    ...

Alternatively, CMake options may be set in a CMake *cache file* and passing 
that file to CMake using the CMake ``-C`` option; for example::

    cmake ... \
    -C my_cache_file.cmake \
    ...

The directories ``RAJA/scripts/*-builds`` contain scripts that run CMake for
various build configurations. These contain cmake invocations that use CMake 
cache files (we call them *host-config* files) and may be used as a guide for 
users trying to set their own options. 

Next, we summarize RAJA CMake options and their defaults.


.. _configopt-raja-features-label:

==========================================
Available RAJA CMake Options and Defaults
==========================================

RAJA uses a variety of custom variables to control how it is compiled. Many 
of these are used internally to control RAJA compilation and do 
not need to be set by users. Others can be used to enable or disable certain 
RAJA features. Most variables get translated to 
compiler directives and definitions in the RAJA ``config.hpp`` file that is 
generated when CMake runs. The ``config.hpp`` header file is included in other 
RAJA headers so all options propagate consistently through the 
build process for all of the code. 

The following tables describe which variables set RAJA options and 
and their default settings. 

.. note:: Items marked with a double asterisk (**) indicate variables that 
          are supported directly in BLT/CMake and are also supported in RAJA as
          CMake dependent options, for finer-grained configuration control.
          The RAJA CMake dependent variable form adds the prefix ``RAJA_``.

Examples, tests, warnings, etc.
--------------------------------

CMake variables can be used to control whether RAJA tests, examples, 
tutorial exercises, etc. are built when RAJA is compiled.

      =========================  =========================================
      Variable                   Default
      =========================  =========================================
      **ENABLE_TESTS             On 
      **ENABLE_EXAMPLES          On 
      RAJA_ENABLE_EXERCISES      On 
      **ENABLE_BENCHMARKS        Off
      RAJA_ENABLE_REPRODUCERS    Off 
      **ENABLE_COVERAGE          Off (supported for GNU compilers only)
      =========================  =========================================

Other configuration options are available to specialize how RAJA is compiled:

      ==================================   =========================
      Variable                             Default
      ==================================   =========================
      RAJA_ENABLE_WARNINGS_AS_ERRORS       Off
      RAJA_ENABLE_FORCEINLINE_RECURSIVE    On (Intel compilers only)
      RAJA_ALLOW_INCONSISTENT_OPTIONS      Off 
      ==================================   =========================

RAJA Views/Layouts may be configured to check for out of bounds 
indexing at run time:

      =========================   ======================
      Variable                    Default
      =========================   ======================
      RAJA_ENABLE_BOUNDS_CHECK    Off
      =========================   ======================

.. note:: RAJA bounds checking is a run time check and will add considerable 
          execution time overhead. Thus, this feature should only be used for 
          debugging and correctness checking and should be disabled for 
          production builds.
    
RAJA Features
-------------------

Some RAJA features are enabled by RAJA-specific CMake variables.

      ===========================   =======================================
      Variable                      Meaning
      ===========================   =======================================
      RAJA_ENABLE_RUNTIME_PLUGINS   Enable support for dynamically loaded
                                    RAJA plugins. Default is off.
      RAJA_ENABLE_DESUL_ATOMICS     Replace RAJA atomic implementations
                                    with Desul variants at compile-time.
                                    Default is off.
      RAJA_ENABLE_VECTORIZATION     Enable SIMD/SIMT intrinsics support.
                                    Default is on.
      ===========================   =======================================
 
Programming model back-end support
-------------------------------------

Variables that control which RAJA programming model back-ends are enabled
are as follows (names are descriptive of what they enable):

      ==========================   ============================================
      Variable                     Default
      ==========================   ============================================
      **ENABLE_OPENMP              Off
      **ENABLE_CUDA                Off
      **ENABLE_HIP                 Off
      RAJA_ENABLE_TARGET_OPENMP    Off (when on, ENABLE_OPENMP must also be on)
      RAJA_ENABLE_SYCL             Off
      ==========================   ============================================

Other programming model specific compilation options are also available:

      ======================================   =================================
      Variable                                 Default
      ======================================   =================================
      **ENABLE_CLANG_CUDA                      Off (if on, ENABLE_CUDA 
                                               must be on too!)
      RAJA_ENABLE_EXTERNAL_CUB                 Off
      RAJA_ENABLE_NVTX                         Off
      RAJA_ENABLE_EXTERNAL_ROCPRIM             Off
      RAJA_ENABLE_ROCTX                        Off
      ======================================   =================================

Turning the ``(RAJA_)ENABLE_CLANG_CUDA`` variable on will build CUDA 
code with the native support in the Clang compiler.

.. note:: See :ref:`getting_started-label` for more information about
          using the ``RAJA_ENABLE_EXTERNAL_CUB`` and 
          ``RAJA_ENABLE_EXTERNAL_ROCPRIM`` variables, as well other 
          RAJA back-ends.

Timer Options
--------------

RAJA provides a simple portable timer class that is used in RAJA
example codes to determine execution timing and can be used in other apps
as well. This timer can use any of three internal timers depending on
your preferences, and one should be selected by setting the 'RAJA_TIMER'
variable. 

      ======================   ======================
      Variable                 Values
      ======================   ======================
      RAJA_TIMER               chrono (default),
                               gettime,
                               clock
      ======================   ======================

What these variables mean:

      =============================   ========================================
      Value                           Meaning
      =============================   ========================================
      chrono                          Use the std::chrono library from the 
                                      C++ standard library
      gettime                         Use `timespec` from the C standard 
                                      library time.h file
      clock                           Use `clock_t` from time.h
      =============================   ========================================

Data types, sizes, alignment, etc.
-------------------------------------

The options discussed in this section are typically not needed by users.
They are provided for special cases when users want to parameterize floating 
point types in applications, which makes it easier to switch between types.

.. note:: RAJA data types in this section are provided as a convenience to 
          users if they wish to use them. They are not used within RAJA 
          implementation code directly.

The following variables are used to set the data type for the type
alias ``RAJA::Real_type``:

      ======================   ======================
      Variable                 Default
      ======================   ======================
      RAJA_USE_DOUBLE          On (type is double)
      RAJA_USE_FLOAT           Off 
      ======================   ======================

     Similarly, the ``RAJA::Complex_type`` can be enabled to support complex 
     numbers if needed:

      ======================   ======================
      Variable                 Default
      ======================   ======================
      RAJA_USE_COMPLEX         Off 
      ======================   ======================

When turned on, the ``RAJA::Complex_type`` is an alias to 
``std::complex<Real_type>``.

There are several variables to control the definition of the RAJA 
floating-point data pointer type ``RAJA::Real_ptr``. The base data type
is always ``Real_type``. When RAJA is compiled for CPU execution 
only, the defaults are:

      =============================   ======================
      Variable                        Default
      =============================   ======================
      RAJA_USE_BARE_PTR               Off
      RAJA_USE_RESTRICT_PTR           On
      RAJA_USE_RESTRICT_ALIGNED_PTR   Off
      RAJA_USE_PTR_CLASS              Off
      =============================   ======================

When RAJA is compiled with CUDA enabled, the defaults are:

      =============================   ======================
      Variable                        Default
      =============================   ======================
      RAJA_USE_BARE_PTR               On
      RAJA_USE_RESTRICT_PTR           Off
      RAJA_USE_RESTRICT_ALIGNED_PTR   Off
      RAJA_USE_PTR_CLASS              Off
      =============================   ======================

The meaning of these variables is:

      =============================   ========================================
      Variable                        Meaning
      =============================   ========================================
      RAJA_USE_BARE_PTR               Use standard C-style pointer
      RAJA_USE_RESTRICT_PTR           Use C-style pointer with restrict
                                      qualifier
      RAJA_USE_RESTRICT_ALIGNED_PTR   Use C-style pointer with restrict
                                      qualifier and alignment attribute 
                                      (see RAJA_DATA_ALIGN below)
      RAJA_USE_PTR_CLASS              Use pointer class with overloaded `[]` 
                                      operator that applies restrict and 
                                      alignment intrinsics. This is useful 
                                      when a compiler does not support 
                                      attributes in a typedef.
      =============================   ========================================

RAJA internally uses a parameter to define platform-specific constant
data alignment. The variable that control this is:

      =============================   ======================
      Variable                        Default
      =============================   ======================
      RAJA_DATA_ALIGN                 64
      =============================   ======================

This variable is used to specify data alignment used in intrinsics and typedefs
in units of **bytes**.

For details on the options in this section are used, please see the 
header file ``RAJA/include/RAJA/util/types.hpp``.

Other RAJA Features
-------------------
   
RAJA contains some features that are used mainly for development or may
not be of general interest to RAJA users. These are turned off be default.
They are described here for reference and completeness.

      ===========================   =======================================
      Variable                      Meaning
      ===========================   =======================================
      RAJA_ENABLE_FT                Enable/disable RAJA experimental
                                    loop-level fault-tolerance mechanism
      RAJA_REPORT_FT                Enable/disable a report of fault-
                                    tolerance enabled run (e.g., number of 
                                    faults detected, recovered from, 
                                    recovery overhead, etc.)
      ===========================   =======================================


.. _configopt-raja-backends-label:

===============================
Setting RAJA Back-End Features
===============================

Various `ENABLE_*` options are listed above for enabling RAJA back-ends,
such as OpenMP and CUDA. To access compiler and hardware optimization features,
it may be necessary to pass additional options to CMake. Please see
:ref:`getting_started-label` for more information. 
