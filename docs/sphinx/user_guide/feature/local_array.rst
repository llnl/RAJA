.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _feat-local_array-label:

===========
Local Array
===========

This section introduces RAJA *local arrays*. A ``RAJA::LocalArray`` is an
array object with one or more dimensions whose memory is allocated when a 
RAJA kernel is executed and only lives within the scope of the kernel 
execution. To motivate the concept and usage, consider a simple C example
in which we construct and use two arrays in nested loops::

           for(int k = 0; k < 7; ++k) { //k loop

            int a_array[7][5];
            int b_array[5];

             for(int j = 0; j < 5; ++j) { //j loop
               a_array[k][j] = 5*k + j;
               b_array[j] = 7*j + k;
             }

             for(int j = 0; j < 5; ++j) { //j loop
               printf("%d %d \n",a_array[k][j], b_array[j]);
             }

           }

Here, two stack-allocated arrays are defined inside the outer 'k' loop and 
used in both inner 'j' loops. 

This loop pattern may be also be written using RAJA local arrays in a 
``RAJA::kernel_param`` kernel. We show this next, and then discuss 
its constituent parts::

  // 
  // Define two local arrays
  // 

  using RAJA_a_array = RAJA::LocalArray<int, RAJA::Perm<0, 1>, RAJA::SizeList<5,7> >;
  RAJA_a_array kernel_a_array;

  using RAJA_b_array = RAJA::LocalArray<int, RAJA::Perm<0>, RAJA::SizeList<5> >;
  RAJA_b_array kernel_b_array;


  // 
  // Define the kernel execution policy
  // 

  using POL = RAJA::KernelPolicy<
                RAJA::statement::For<1, RAJA::seq_exec,
                  RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<0, 1>,
                    RAJA::statement::For<0, RAJA::seq_exec,
                      RAJA::statement::Lambda<0>
                    >,
                    RAJA::statement::For<0, RAJA::seq_exec,
                      RAJA::statement::Lambda<1>
                    >
                  >
                >
              >;


  // 
  // Define the kernel
  // 

  RAJA::kernel_param<POL> ( RAJA::make_tuple(RAJA::TypedRangeSegment<int>(0,5),
                                             RAJA::TypedRangeSegment<int<(0,7)),
                            RAJA::make_tuple(kernel_a_array, kernel_b_array),

    [=] (int j, int k, RAJA_a_array& kernel_a_array, RAJA_b_array& kernel_b_array) {
      a_array(k, j) = 5*k + j;
      b_array(j) = 5*k + j;
    },

    [=] (int j, int k, RAJA_a_array& a_array, RAJA_b_array& b_array) {
      printf("%d %d \n", kernel_a_array(k, j), kernel_b_array(j));
    }

  );

The RAJA version defines two ``RAJA::LocalArray`` types, one 
two-dimensional and one one-dimensional and creates an instance of each type. 
The template arguments for the ``RAJA::LocalArray`` types are:

  * Array data type
  * Index striding order (see :ref:`feat-view-label` for details)
  * Array dimensions

The local array instances are passed to the kernel in a tuple after the 
iteration space tuple. 

The kernel policy is a two-level nested loop policy (see 
:ref:`loop_elements-kernel-label` for information about RAJA kernel policies) 
with a statement type ``RAJA::statement::InitLocalMem`` inserted between the 
nested 'For' statements, which allocates the memory for the local arrays when 
the kernel executes. The ``InitLocalMem`` statement type has two parameters.
One for the memory type ``RAJA::cpu_tile_mem``, and one for specifying which
parameter tuple entries correspond to the local arrays 
``RAJA::ParamList<0, 1>``. The local array initialization is done in the first 
lambda expression, and the local array values are printed in the second lambda 
expression.

.. note:: ``RAJA::LocalArray`` types support arbitrary dimensions and extents
          in each dimension.

-------------------
Memory Policies
-------------------

``RAJA::LocalArray`` supports CPU stack-allocated memory and CUDA or HIP GPU 
shared memory and thread private memory. See :ref:`localarraypolicy-label` 
for a discussion of available memory policies.
