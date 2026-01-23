.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##


.. _getting_started-label:

*************************
Getting Started With RAJA
*************************

This section should help get you up and running with RAJA quickly.

============
Requirements
============

The primary requirement for using RAJA is a C++17 standard compliant compiler.
Certain features, such as programming model back-ends like CUDA or HIP,
must be supported by the compiler you chose to use them. Available RAJA
configuration options and how to enable or disable features are described 
in :ref:`configopt-label`. 

To build RAJA and use its most basic features, you will need:

- C++ compiler with C++17 support
- `CMake <https://cmake.org/>`_ version 3.23 or greater.


==================
Get the Code
==================

The RAJA project is hosted on GitHub: 
`GitHub RAJA project <https://github.com/LLNL/RAJA>`_. To get the code, clone 
the repository into a local working space using the command::

   $ git clone --recursive https://github.com/LLNL/RAJA.git

The ``--recursive`` option above is used to pull RAJA Git *submodules*, on 
which RAJA depends, into your local copy of the RAJA repository.

After running the clone command, a copy of the RAJA repository will reside in
the ``RAJA`` subdirectory where you ran the clone command. You will be on the 
``develop`` branch, which is the default RAJA branch.

If you do not pass the ``--recursive`` argument to the ``git clone``
command, you can also type the following commands after cloning::

  $ cd RAJA
  $ git submodule update --init --recursive

Either way, the end result is the same and you should be good to configure the
code and build it.

.. note:: * If you switch branches in a RAJA repo (e.g., you are on a branch,
            with everything up-to-date, and you run the command 
            ``git checkout <different branch name>``, you may need to run 
            the command ``git submodule update`` to set the Git submodule 
            versions to what is used by the new branch.
          * If the set of submodules in a new branch is different than the
            previous branch you were on, you may need to run the command
            ``git submodule update --init --recursive`` to pull in the 
            correct set of submodule and versions.

.. _getting_started_depend-label:

==================
Dependencies
==================

RAJA has several dependencies that are required based on how you want to
build and use it. The RAJA Git repository has submodules that contain 
most of these dependencies.

RAJA includes other submodule dependencies, which are used to support our 
Gitlab CI testing. These are described in the RAJA Developer Guide. 

Dependencies that are required to build the RAJA code are:

- A C++ 17 standard compliant compiler
- `BLT build system <https://github.com/LLNL/blt>`_
- `CMake <https://cmake.org/>`_ version 3.23 or greater when building the HIP back-end, and version 3.20 or greater otherwise.
- `Camp compiler agnostic metaprogramming library  <https://github.com/LLNL/camp>`_

Other dependencies that users should be aware of that support certain 
features are:

- `CUB CUDA utilities library <https://github.com/NVlabs/cub>`_, which is required for using the RAJA CUDA back-end.
- `rocPRIM HIP parallel primitives library <https://github.com/ROCmSoftwarePlatform/rocPRIM.git>`_, which is required for using the RAJA HIP back-end.
- `Desul <https://github.com/desul/desul>`_, which is required if you want to use Desul atomics in RAJA instead of our current default atomics. Note that we plan to switch over to Desul atomics exclusively at some point.

.. note:: You may want or need to use external versions of camp, CUB, or 
          rocPRIM instead of the RAJA submodules. This is usually the case
          when you are using RAJA along with some other library that also
          needs one of these. To do so, you need to use CMake variables to 
          pass a path to a valid installation of each library. Specifically:

            * External camp::

                cmake \
                ... \
                -Dcamp_DIR=path/to/camp/install \
                ...

            * External CUB::

                cmake \
                ... \ 
                -DRAJA_ENABLE_EXTERNAL_CUB=On \
                -DCUB_DIR=path/to/cub \
                ...

            * External rocPRIM:: 

                cmake \
                ... \
                -DRAJA_ENABLE_EXTERNAL_ROCPRIM=On \
                -DROCPRIM_DIR=path/to/rocPRIM \
                ... 

More information about configuring GPU builds with CUDA or HIP is provided
in :ref:`getting_started_build_gpu-label`

Additional discussion of these dependencies, with respect to building RAJA, is 
provided in :ref:`getting_started_build-label`. Other than that, you probably 
don't need to know much about them. If you are curious and want to know more, 
please click on the link to the library you want to know about in the above 
list.

.. _getting_started_build-label:

==================
Build and Install
==================

The complexity of building and installing RAJA depends on which features you 
want to use and how easy it is to do this on your system.

.. note:: RAJA builds must be *out-of-source*. In particular, RAJA does not 
          allow building in its source directory. You must create a build 
          directory and run CMake in it.

RAJA uses CMake to configure a build. To create a "bare bones" configuration, 
build, and install it, you can do the following::

  $ mkdir build-dir && cd build-dir
  $ cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ../
  $ make  (or make -j <N> for a parallel build)
  $ make install

Running ``cmake`` generates the RAJA build configuration. Running ``make``
compiles the code. Running ``make install`` copies RAJA header files 
to an ``include`` directory and installs the RAJA library in a ``lib`` 
directory, both in the directory location specified with the
``-DCMAKE_INSTALL_PREFIX`` CMake option.

Other build configurations are accomplished by passing other options to CMake.
For example, if you want to use a C++ compiler other than the default on 
your system, you would pass a path to the compiler using the standard
CMake option ``-DCMAKE_CXX_COMPILER=path/to/compiler``.
When you run CMake, it will generate output about the build configuration 
(compiler and version, options, etc.), which is helpful to make sure CMake
is doing what you want. For a summary of RAJA configuration 
options, please see :ref:`configopt-label`.

.. note:: RAJA is configured to build its tests, examples, and tutorial
          exercises by default. If you do not disable them with the 
          appropriate CMake option (see :ref:`configopt-label`), 
          you can run them after the build completes to check if everything 
          is built properly.

          The easiest way to run the full set of RAJA tests is to type::

             $ make test

          in the build directory after the build completes.

          You can also run individual tests by invoking the corresponding
          test executables directly. They will be located in the ``test`` 
          subdirectory in your build space. RAJA tests use the 
          `Google Test framework <https://github.com/google/googletest>`_, 
          so you can also run and filter tests via Google Test commands.

          The source files for RAJA examples and exercises are located in 
          the ``RAJA/examples`` and ``RAJA/exercises`` directories, 
          respectively. When built, the executables for the examples and 
          exercises will be located in the ``bin`` subdirectory in your build
          space.

.. _getting_started_build_gpu-label:

-------------------------------------------
Additional RAJA Back-end Build Information
-------------------------------------------

Configuring a RAJA build to support a GPU back-end, such as CUDA, HIP, or 
OpenMP target offload, typically requires additional CMake options, which 
we describe next. 

CUDA
^^^^^^

To run RAJA code on NVIDIA GPUs, one typically must have a CUDA compiler 
installed on the system, in addition to a host code compiler. You may need 
to specify both when you run CMake. The host compiler is specified using the 
``CMAKE_CXX_COMPILER`` CMake variable as described earlier. The CUDA software
stack and compiler are specified using the following CMake options:

  * -DCUDA_TOOLKIT_ROOT_DIR=path/to/cuda/toolkit
  * -DCMAKE_CUDA_COMPILER=path/to/nvcc

When using the NVIDIA nvcc compiler for RAJA CUDA functionality, the variables:

  * CMAKE_CUDA_FLAGS_RELEASE
  * CMAKE_CUDA_FLAGS_DEBUG
  * CMAKE_CUDA_FLAGS_RELWITHDEBINFO

correspond to the standard CMake build types and are used to pass additional
compiler options to nvcc.

.. note:: Often, nvcc must pass options to the host compiler, the arguments
          can be included using the ``CMAKE_CUDA_FLAGS...`` CMake variables
          listed above. Host compiler options must be prepended with the 
          ``-Xcompiler`` directive to properly propagate.

To set the CUDA compute architecture, which should be chosen based on the 
NVIDIA GPU hardware you are using, you can use the ``CMAKE_CUDA_ARCHITECTURES`` 
CMake variable. For example, the CMake option 
``-DCMAKE_CUDA_ARCHITECTURES=70`` will tell the 
compiler to use the `sm_70` SASS architecture in its second stage of 
compilation. The compiler will pick the PTX architecture to use in the first 
stage of compilation that is suitable for the SASS architecture you specify.

Alternatively, you may specify the PTX and SASS architectures, using
appropriate nvcc options in the ``CMAKE_CUDA_FLAGS_*`` variables.

.. note:: **RAJA requires a minimum CUDA architecture level of `sm_35` to use
          all supported CUDA features.** Mostly, the architecture level affects
          which RAJA CUDA atomic operations are available and how they are
          implemented inside RAJA. This is described in 
          :ref:`feat-atomics-label`.

          * If you do not specify a value for ``CMAKE_CUDA_ARCHITECTURES``, 
            it will be set to `35` by default and CMake will emit a status 
            message indicating this choice was made.

          * If you give a ``CMAKE_CUDA_ARCHITECTURES`` value less than `35` 
            (e.g., `30`), CMake will report this as an error and stop 
            processing.

Also, RAJA relies on the CUB CUDA utilities library, mentioned earlier, for 
some CUDA back-end functionality. The CUB version included in the CUDA toolkit 
installation is used by default when available. This is the case for CUDA 
version 11 and later. RAJA includes a CUB submodule that is used by default
with older versions of CUDA. To use an external CUB installation, provide the 
following options to CMake:: 

  cmake \
  ... \
  -DRAJA_ENABLE_EXTERNAL_CUB=On \
  -DCUB_DIR=<path/to/cub> \
  ...

.. note:: The CUDA toolkit version of CUB is
          required for compatibility with the CUDA toolkit version of thrust
          starting with CUDA version 11.0.0. So, if you build
          RAJA with CUDA version 11 or higher, you should use the version of
          CUB contained in the CUDA toolkit version you are using to use 
          Thrust and to be compatible with libraries that use Thrust.

.. note:: The version of Googletest that
          is used in RAJA version v0.11.0 or newer requires CUDA version
          9.2.x or newer when compiling with nvcc. Thus, if you build
          RAJA with CUDA enabled and want to also enable RAJA tests, you
          must use CUDA version 9.2.x or newer.

HIP
^^^^

To run RAJA code on AMD GPUs, one typically uses a ROCm compiler and tool 
chain (which can also be used to compile code for NVIDIA GPUs, which is not
covered in detail in RAJA user documentation).

.. note:: RAJA requires version 3.5 or newer of the ROCm software stack to 
          use the RAJA HIP back-end.

Unlike CUDA, you do not specify a host compiler and a device compiler when 
using the AMD ROCm software stack. Typical CMake options to use when building 
with a ROCm stack are:

  * -DROCM_ROOT_DIR=path/to/rocm
  * -DHIP_ROOT_DIR=path/to/hip
  * -DHIP_PATH=path/to/hip/binaries
  * -DCMAKE_CXX_COMPILER=path/to/rocm/compiler 

Additionally, you use the CMake variable ``CMAKE_HIP_ARCHITECTURES`` to set
the target compute architecture. For example::

  -DCMAKE_HIP_ARCHITECTURES=gfx908

RAJA relies on the rocPRIM HIP utilities library for some HIP
functionality. The rocPRIM included in the ROCm install is used by default if
available. RAJA includes a rocPRIM submodule that is used if it is not
available. To use an external rocPRIM install provide the following options
to CMake::

  cmake \
  ... \
  -DRAJA_ENABLE_EXTERNAL_ROCPRIM=On \
  -DROCPRIM_DIR=<pat/to/rocPRIM> \
  ...

.. note:: When using HIP and targeting NVIDIA GPUs, RAJA uses CUB instead of
          rocPRIM. In this case, you must configure with an external CUB 
          install using the CMake variables described in the CUDA section above.

OpenMP
^^^^^^^

To use OpenMP target offload GPU execution, additional options may need to be
passed to the compiler. BLT variables are used for this. Option syntax follows 
the CMake *list* pattern. For example, to specify OpenMP target options for 
NVIDIA GPUs using a clang-based compiler, one may do something like::

   cmake \
     ... \
     -DBLT_OPENMP_COMPILE_FLAGS="-fopenmp;-fopenmp-targets=nvptx64-nvidia-cuda" \
     -DBLT_OPENMP_LINK_FLAGS="-fopenmp;-fopenmp-targets=nvptx64-nvidia-cuda" \
     ...

Compiler flags are passed to other compilers similarly, using flags specific to
the compiler. Typically, the compile and link flags are the same as shown here.

----------------------------------------
RAJA Example Build Configuration Files
----------------------------------------

The RAJA repository has subdirectories ``RAJA/scripts/*-builds`` that contain
a variety of build scripts we use to build and test RAJA on various platforms 
with various compilers. These scripts pass files (*CMake cache files*) 
located in the ``RAJA/host-configs`` directory to CMake using the '-C' option.
These files serve as useful examples of how to configure RAJA prior to
compilation.

======================
Learning to Use RAJA
======================

The RAJA repository contains a variety of example source codes that you are 
encouraged to view and run to learn about how to use RAJA:

  * The ``RAJA/examples`` directory contains various examples that illustrate
    algorithm patterns.
  * The ``RAJA/exercises`` directory contains exercises for users to work 
    through along with complete solutions. These are described in detail
    in the :ref:`tutorial-label` section.
  * Other examples can also be found in the ``RAJA/test`` directories.

We mentioned earlier that RAJA examples, exercises, and tests are built by
default when RAJA is compiled. So, unless you explicitly disable them when 
you run CMake to configure a RAJA build, you can run them after compiling RAJA.
Executables for the examples and exercises will be located in the
``<build-dir>/bin`` directory in your build space. Test executables will
be located in the ``<build-dir>/test`` directory.

For an overview of all the main RAJA features, see :ref:`features-label`.
A full tutorial with a variety of examples showing how to use RAJA features
can be found in :ref:`tutorial-label`.

.. important:: The `RAJA Performance Suite <https://github.com/LLNL/RAJAPerf>`
               is an excellent source of RAJA usage examples. The Suite
               contains many numerical kernels, each of which is implemented
               in a variety of RAJA and non-RAJA variants in OpenMP, CUDA, HIP,
               SYCL, etc. Comparing different variants of these kernels is
               instructive to understand how to use RAJA features and how they
               work.
