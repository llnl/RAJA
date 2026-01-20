.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _using-raja-label:

******************************
Using RAJA in Your Application
******************************

Using RAJA in an application requires two things: ensuring the RAJA header files
are visible, and linking against the RAJA library. We maintain a 
`RAJA Template Project <https://github.com/LLNL/RAJA-project-template>`_
that shows how to use RAJA in a project that uses CMake or make, either as a 
Git submodule or as an externally installed library that you link your 
application against.

========================
CMake Configuration File
========================

As part of the RAJA installation, we provide a ``RAJA-config.cmake`` file. If
your application uses CMake, this can be used with CMake's find_package
capability to import RAJA into your CMake project.

To use the configuration file, you can add the following command to your CMake
project::

  find_package(RAJA)

Then, pass the path of RAJA to CMake when you configure your code::

  cmake -DRAJA_DIR=<path-to-raja-install>/lib/cmake/raja/

The ``RAJA-config.cmake`` file provides a ``RAJA`` target, that can be used
natively by CMake to add a dependency on RAJA. For example::

  add_executable(my-app.exe
                 my-app.cpp)

  target_link_libraries(my-app.exe PUBLIC RAJA)
