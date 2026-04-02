.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _tut-dotproduct-label:

-----------------------------------
Sum Reduction: Vector Dot Product
-----------------------------------

This section contains an exercise file ``RAJA/exercises/dot-product.cpp``
for you to work through if you wish to get some practice with RAJA. The
file ``RAJA/exercises/dot-product_solution.cpp`` contains complete
working code for the examples discussed in this section. You can use the
solution file to check your work and for guidance if you get stuck. To build
the exercises execute ``make dot-product`` and ``make dot-product_solution``
from the build directory.

Key RAJA features shown in this example are:

  * ``RAJA::forall`` loop execution template and execution policies
  * ``RAJA::TypedRangeSegment`` iteration space construct
  * ``RAJA::ReduceSum`` sum reduction template and reduction policies

In the example, we compute a vector dot product, 'dot = (a,b)', where 
'a' and 'b' are two vectors of length N and 'dot' is a scalar. Typical
C-style code to compute the dot product and print its value afterward is: 

.. literalinclude:: ../../../../exercises/dot-product_solution.cpp
   :start-after: _csytle_dotprod_start
   :end-before: _csytle_dotprod_end
   :language: C++

Although this kernel is serial, it is representative of a *reduction* 
operation which is a common algorithm pattern that
produces a single result from a set of values. Reductions present a variety
of issues that must be addressed to operate properly in parallel. 

^^^^^^^^^^^^^^^^^^^^^
RAJA Variants
^^^^^^^^^^^^^^^^^^^^^

Different programming models support parallel reduction operations differently.
Some models, such as CUDA, do not provide direct support for reductions and 
so such operations must be explicitly coded by users. It can be challenging 
to generate a correct and high performance implementation. RAJA provides 
portable reduction types that make it easy to perform reduction operations
in kernels. The RAJA variants of the dot product computation show how 
to use the ``RAJA::ReduceSum`` sum reduction template type. RAJA provides
other reduction types and allows multiple reduction operations to be
performed in a single kernel alongside other computations. Please see 
:ref:`feat-reductions-label` for more information.

Each RAJA reduction type takes a `reduce policy` template argument, which
**must be compatible with the execution policy** applied to the kernel
in which the reduction is used. Here is the RAJA sequential variant of the dot 
product computation:

.. literalinclude:: ../../../../exercises/dot-product_solution.cpp
   :start-after: _rajaseq_dotprod_start
   :end-before: _rajaseq_dotprod_end
   :language: C++

The sum reduction object is defined by specifying the reduction 
policy ``RAJA::seq_reduce`` matching the kernel execution policy
``RAJA::seq_exec``, and a reduction value type (i.e., 'double'). An initial 
value of zero for the sum is passed to the reduction object constructor. After 
the kernel executes, we use the 'get' method to retrieve the reduced value.

The OpenMP multithreaded variant of the kernel is implemented similarly:

.. literalinclude:: ../../../../exercises/dot-product_solution.cpp
   :start-after: _rajaomp_dotprod_start
   :end-before: _rajaomp_dotprod_end
   :language: C++

Here, we use the ``RAJA::omp_reduce`` reduce policy to match the OpenMP
kernel execution policy.

The RAJA CUDA variant is achieved by using appropriate kernel execution and 
reduction policies:

.. literalinclude:: ../../../../exercises/dot-product_solution.cpp
   :start-after: _rajacuda_dotprod_start
   :end-before: _rajacuda_dotprod_end
   :language: C++

Here, the CUDA reduce policy ``RAJA::cuda_reduce`` matches the CUDA 
kernel execution policy. Note that the CUDA thread block size is not 
specified in the reduce policy as it will use the same value as the
loop execution policy.

Similarly, for the RAJA HIP variant:

.. literalinclude:: ../../../../exercises/dot-product_solution.cpp
   :start-after: _rajahip_dotprod_start
   :end-before: _rajahip_dotprod_end
   :language: C++

It is worth repeating how similar the code looks for each of these variants.
The loop body is identical for each and only the loop execution policy
and reduce policy types change.

.. note:: Currently available reduction capabilities in RAJA require a
          *reduction policy* type that is compatible with the execution
          policy for the kernel in which the reduction is used. We
          are developing a new reduction interface for RAJA that will
          provide an alternative for which the reduction policy is not 
          required.
