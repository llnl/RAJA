.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _tut-offsetlayout-label:

------------------------------------------------
OffsetLayout: Five-point Stencil
------------------------------------------------

This section contains an exercise file ``RAJA/exercises/offset-layout-stencil.cpp``
for you to work through if you wish to get some practice with RAJA. The
file ``RAJA/exercises/offset-layout-stencil.cpp`` contains
complete working code for the examples discussed in this section. You can use
the solution file to check your work and for guidance if you get stuck. To build
the exercises execute ``make offset-layout-stencil`` and ``make offset-layout-stencil_solution``
from the build directory.

Key RAJA features shown in the following example:

  * ``RAJA::kernel`` loop execution template and execution policies
  * ``RAJA::View`` multi-dimensional data access
  * ``RAJA:make_offset_layout`` method to create an offset Layout

The examples in this section apply a five-point stencil to the interior cells 
of a two-dimensional lattice and store a resulting sum in a second 
lattice of equal size. The five-point stencil associated with a lattice cell
accumulates the value in the cell and each of its four neighbors. We use 
``RAJA::View`` and ``RAJA::OffsetLayout`` constructs to simplify 
the multi-dimensional indexing so that we can write the stencil operation 
naturally, as such::

  output(row, col) = input(row, col) + 
                     input(row - 1, col) + input(row + 1, col) + 
                     input(row, col - 1) + input(row, col + 1)

A lattice is assumed to have :math:`N_r \times N_c` interior cells with unit 
values surrounded by a halo of cells containing zero values for a total 
dimension of :math:`(N_r + 2) \times (N_c + 2)`. For example, when
:math:`N_r = N_c = 3`, the input lattice and values are:

  +---+---+---+---+---+
  | 0 | 0 | 0 | 0 | 0 |
  +---+---+---+---+---+
  | 0 | 1 | 1 | 1 | 0 |
  +---+---+---+---+---+
  | 0 | 1 | 1 | 1 | 0 |
  +---+---+---+---+---+
  | 0 | 1 | 1 | 1 | 0 |
  +---+---+---+---+---+
  | 0 | 0 | 0 | 0 | 0 |
  +---+---+---+---+---+

After applying the stencil, the output lattice and values are:

  +---+---+---+---+---+
  | 0 | 0 | 0 | 0 | 0 |
  +---+---+---+---+---+
  | 0 | 3 | 4 | 3 | 0 |
  +---+---+---+---+---+
  | 0 | 4 | 5 | 4 | 0 |
  +---+---+---+---+---+
  | 0 | 3 | 4 | 3 | 0 |
  +---+---+---+---+---+
  | 0 | 0 | 0 | 0 | 0 |
  +---+---+---+---+---+

For this :math:`(N_r + 2) \times (N_c + 2)` lattice case, here is our 
(row, col) indexing scheme.

  +----------+---------+---------+---------+---------+
  | (-1, 3)  | (0, 3)  | (1, 3)  | (2, 3)  | (3, 3)  |
  +----------+---------+---------+---------+---------+
  | (-1, 2)  | (0, 2)  | (1, 2)  | (2, 2)  | (3, 2)  |
  +----------+---------+---------+---------+---------+
  | (-1, 1)  | (0, 1)  | (1, 1)  | (2, 1)  | (3, 1)  |
  +----------+---------+---------+---------+---------+
  | (-1, 0)  | (0, 0)  | (1, 0)  | (2, 0)  | (3, 0)  |
  +----------+---------+---------+---------+---------+
  | (-1, -1) | (0, -1) | (1, -1) | (2, -1) | (3, -1) |
  +----------+---------+---------+---------+---------+

Notably, :math:`[0, N_r) \times [0, N_c)` corresponds to the interior index
range over which we apply the stencil, and :math:`[-1,N_r+1) \times [-1, N_c+1)`
is the full lattice index range.

For reference and comparison to the ``RAJA::kernel`` implementations 
described below, we begin by walking through a C-style version of the stencil 
computation. First, we define the size of our lattice:

.. literalinclude:: ../../../../exercises/offset-layout-stencil_solution.cpp
   :start-after: _stencil_define_start
   :end-before: _stencil_define_end
   :language: C++

Then, after allocating input and output arrays, we initialize the input:

.. literalinclude:: ../../../../exercises/offset-layout-stencil_solution.cpp
   :start-after: _stencil_input_init_start
   :end-before: _stencil_input_init_end
   :language: C++

and compute the reference output solution:

.. literalinclude:: ../../../../exercises/offset-layout-stencil_solution.cpp
   :start-after: _stencil_output_ref_start
   :end-before: _stencil_output_ref_end
   :language: C++


^^^^^^^^^^^^^^^^^^^
RAJA Offset Layouts
^^^^^^^^^^^^^^^^^^^

We use the ``RAJA::make_offset_layout`` method to construct a 
``RAJA::OffsetLayout`` object that we use to create ``RAJA::View`` objects
for our input and output data arrays:

.. literalinclude:: ../../../../exercises/offset-layout-stencil_solution.cpp
   :start-after: _offsetlayout_views_start
   :end-before: _offsetlayout_views_end
   :language: C++

Here, the row index range is :math:`[-1, N_r+1)`, and the column index 
range is :math:`[-1, N_c+1)`. The first argument to each call to the 
``RAJA::View`` constructor is the pointer to the array that holds the View
data. The second argument is the ``RAJA::OffsetLayout`` object.

``RAJA::OffsetLayout`` objects allow us to write loops over
data arrays using non-zero based indexing and without having to manually 
compute offsets into the arrays.

For more information about RAJA View and Layout types, please see 
:ref:`feat-view-label`.

^^^^^^^^^^^^^^^^^^^^^^^^^^^
RAJA Kernel Variants
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the RAJA implementations of the stencil computation, we use two 
``RAJA::TypedRangeSegment`` objects to define the row and column iteration 
spaces for the interior cells:

.. literalinclude:: ../../../../exercises/offset-layout-stencil_solution.cpp
   :start-after: _offsetlayout_ranges_start
   :end-before: _offsetlayout_ranges_end
   :language: C++

Now, we have all the ingredients to implement the stencil computation using
``RAJA::kernel``. Here is a sequential CPU variant:

.. literalinclude:: ../../../../exercises/offset-layout-stencil_solution.cpp
   :start-after: _offsetlayout_rajaseq_start
   :end-before: _offsetlayout_rajaseq_end
   :language: C++

This RAJA variant does the computation as the C-style variant 
introduced above.

Since the input and output arrays are distinct, the stencil computation is
data parallel. Thus, we can use ``RAJA::kernel`` and an appropriate 
execution policy to run the computation in parallel. Here is an OpenMP
collapse variant that maps the row-column product index space to OpenMP
threads:

.. literalinclude:: ../../../../exercises/offset-layout-stencil_solution.cpp
   :start-after: _offsetlayout_rajaomp_start
   :end-before: _offsetlayout_rajaomp_end
   :language: C++

Note that the lambda expression representing the kernel body is identical to
the ``RAJA::kernel`` sequential version. 

Here are variants for CUDA

.. literalinclude:: ../../../../exercises/offset-layout-stencil_solution.cpp
   :start-after: _offsetlayout_rajacuda_start
   :end-before: _offsetlayout_rajacuda_end
   :language: C++

and HIP

.. literalinclude:: ../../../../exercises/offset-layout-stencil_solution.cpp
   :start-after: _offsetlayout_rajahip_start
   :end-before: _offsetlayout_rajahip_end
   :language: C++

The only difference between the CPU and GPU variants is that the RAJA macro
``RAJA_DEVICE`` is used to decorate the lambda expression with the 
``__device__`` annotation, which is required when capturing a lambda for use
in a GPU device environment as we have discussed in other examples in this
tutorial.

One other point to note is that the CUDA variant in the exercise files 
uses Unified Memory and the HIP variant uses distinct host and device memory
arrays, with explicit host-device data copy operations. Thus, new 
``RAJA::View`` objects were created for the HIP variant to wrap the 
device data pointers used in the HIP kernel. Please see the exercise files
for this example for details.
