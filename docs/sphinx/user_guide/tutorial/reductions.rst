.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _tut-reduction-label:

-----------------------------------------------------
Reduction Types and Kernels with Multiple Reductions
-----------------------------------------------------

This section contains an exercise file ``RAJA/exercises/reductions.cpp``
for you to work through if you wish to get some practice with RAJA. The
file ``RAJA/exercises/reductions_solution.cpp`` contains complete
working code for the examples discussed in this section. You can use the
solution file to check your work and for guidance if you get stuck. To build
the exercises execute ``make reductions`` and ``make reductions_solution``
from the build directory.

Key RAJA features shown in this section are:

  * ``RAJA::forall`` loop execution template and execution policies
  * ``RAJA::TypedRangeSegment`` iteration space construct
  * RAJA reduction types and reduction policies

In the :ref:`tut-dotproduct-label` exercise, we showed how to use the RAJA sum 
reduction type. The following example uses all supported RAJA reduction types: 
min, max, sum, min-loc, max-loc.

.. note:: RAJA 'min-loc' and 'max-loc' reductions determine the min and max 
          reduction value, respectively, along with an iteration index at 
          which the main/max value is found. 

.. note:: Multiple RAJA reductions can be combined in any RAJA loop kernel 
          execution method, and reduction operations can be combined with 
          any other kernel operations. 

.. note:: Each RAJA reduction type requires a reduction policy that must 
          be compatible with the execution policy for the kernel in which 
          it is used.

We start by allocating an array and initializing its values in a 
manner that makes the example mildly interesting and able to show what the 
different reduction types do. Specifically, the array is initialized to
a sequence of alternating values ('1' and '-1'). Then, two values near
the middle of the array are set to '-100' and '100':

.. literalinclude:: ../../../../exercises/reductions_solution.cpp
   :start-after: _reductions_array_init_start
   :end-before: _reductions_array_init_end
   :language: C++

We also define a range segment to iterate over the array:

.. literalinclude:: ../../../../exercises/reductions_solution.cpp
   :start-after: _reductions_range_start
   :end-before: _reductions_range_end
   :language: C++

With these parameters and data initialization, the code example 
presented below will generate the following results:

 * the sum will be zero
 * the min will be -100
 * the max will be 100
 * the min loc will be N/2
 * the max loc will be N/2 + 1

A sequential kernel that exercises all RAJA sequential reduction types 
along with operations after the kernel to print the reduced values is:
 
.. literalinclude:: ../../../../exercises/reductions_solution.cpp
   :start-after: _reductions_raja_seq_start
   :end-before: _reductions_raja_seq_end
   :language: C++

Note that each reduction object takes an initial value at construction. Also,
within the kernel, updating each reduction is done via an operator or method
that is basically what you would expect for the type of reduction 
(e.g., '+=' for sum, 'min()' for min, etc.). After the kernel executes, the 
reduced value computed by each reduction 
object is retrieved after the kernel by calling a 'get()' method on the 
reduction object. The min-loc/max-loc index values are obtained using 
'getLoc()' methods.

For parallel multithreading execution via OpenMP, the exercise can be run with
the execution and reduction policies:

.. literalinclude:: ../../../../exercises/reductions_solution.cpp
   :start-after: _reductions_raja_omppolicy_start
   :end-before: _reductions_raja_omppolicy_end
   :language: C++

Similarly, the kernel containing the reductions can be run in parallel
on a GPU using CUDA policies:

.. literalinclude:: ../../../../exercises/reductions_solution.cpp
   :start-after: _reductions_raja_cudapolicy_start
   :end-before: _reductions_raja_cudapolicy_end
   :language: C++

or HIP policies:

.. literalinclude:: ../../../../exercises/reductions_solution.cpp
   :start-after: _reductions_raja_hippolicy_start
   :end-before: _reductions_raja_hippolicy_end
   :language: C++

