.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _feat-atomics-label:

===================
Atomic Operations
===================

RAJA provides portable atomic operations that can be used to update values
at arbitrary memory locations while avoiding data races. They are described
in this section.

.. note:: All RAJA atomic operations are in the namespace ``RAJA``.

.. note:: Each RAJA atomic operation is templated on an *atomic policy*
          type, which **must be compatible with the execution policy used by 
          the kernel in which it is used.** For example, in 
          a CUDA kernel, a CUDA atomic policy type must be used.

For more information about available RAJA atomic policies, please see
:ref:`atomicpolicy-label`.

.. note:: RAJA support for CUDA atomic operations may be specific to
          the compute architecture for which the code is compiled. Please 
          see :ref:`cudaatomics-label` for more information.

RAJA currently supports two different implementations of atomic operations
via the same basic interface. The default implementation is the original one
developed in RAJA and which has been available for several years. Alternatively,
one can choose an implementation based on 
`DESUL <https://github.com/desul/desul>`_ at compile time. Please see 
:ref:`desul-atomics-label` for more information. Eventually, we plan to 
deprecate the original RAJA implementation and provide only the DESUL 
implementation. The RAJA atomic interface is expected to change when we switch
over to DESUL atomic support. Specifically, the atomic policy noted above will
no longer be used.

Please see the following tutorial sections for detailed examples that use
RAJA atomic operations:

 * :ref:`tut-atomichist-label`.

.. _atomic-ops:

-----------------
Atomic Operations
-----------------

RAJA atomic support the most common atomic operations.

.. note:: Each atomic method described below returns the value of 
          the potentially modified argument (i.e., \*acc) immediately before 
          the atomic operation is applied, in case a user requires it.

^^^^^^^^^^^
Arithmetic
^^^^^^^^^^^

* ``atomicAdd< atomic_policy >(T* acc, T value)`` - Add ``value`` to ``\*acc``.

* ``atomicSub< atomic_policy >(T* acc, T value)`` - Subtract ``value`` from ``\*acc``.

^^^^^^^^^^^
Min/max
^^^^^^^^^^^

* ``atomicMin< atomic_policy >(T* acc, T value)`` - Set ``\*acc`` to min of ``\*acc`` and ``value``.

* ``atomicMax< atomic_policy >(T* acc, T value)`` - Set ``\*acc`` to max of ``\*acc`` and ``value``.

^^^^^^^^^^^^^^^^^^^^
Increment/decrement
^^^^^^^^^^^^^^^^^^^^

* ``atomicInc< atomic_policy >(T* acc)`` - Add 1 to ``\*acc``.

* ``atomicDec< atomic_policy >(T* acc)`` - Subtract 1 from ``\*acc``.

* ``atomicInc< atomic_policy >(T* acc, T compare)`` - Add 1 to ``\*acc`` if ``\*acc`` < ``compare``, else set ``\*acc`` to zero.

* ``atomicDec< atomic_policy >(T* acc, T compare)`` - Subtract 1 from ``\*acc`` if ``\*acc`` != 0 and ``\*acc`` <= ``compare``, else set ``\*acc`` to ``compare``.

^^^^^^^^^^^^^^^^^^^^
Bitwise operations
^^^^^^^^^^^^^^^^^^^^

* ``atomicAnd< atomic_policy >(T* acc, T value)`` - Bitwise 'and' equivalent: Set ``\*acc`` to ``\*acc`` & ``value``. Only works with integral data types.

* ``atomicOr< atomic_policy >(T* acc, T value)`` - Bitwise 'or' equivalent: Set ``\*acc`` to ``\*acc`` | ``value``. Only works with integral data types.

* ``atomicXor< atomic_policy >(T* acc, T value)`` - Bitwise 'xor' equivalent: Set ``\*acc`` to ``\*acc`` ^ ``value``. Only works with integral data types.

^^^^^^^^^^^^^^^^^^^^
Replace
^^^^^^^^^^^^^^^^^^^^

* ``atomicExchange< atomic_policy >(T* acc, T value)`` - Replace ``\*acc`` with ``value``.

* ``atomicCAS< atomic_policy >(T* acc, Tcompare, T value)`` - Compare and swap: Replace ``\*acc`` with ``value`` if and only if ``\*acc`` is equal to ``compare``.

Here is a simple example that shows how to use an atomic operation to compute
an integral sum on a CUDA GPU device::

  //
  // Use CUDA UM to share data pointer with host and device code.
  // RAJA mechanics work the same way if device data allocation
  // and host-device copies are done with traditional cudaMalloc
  // and cudaMemcpy.
  //
  int* sum = nullptr;
  cudaMallocManaged((void **)&sum, sizeof(int));
  cudaDeviceSynchronize();
  *sum = 0;

  RAJA::forall< RAJA::cuda_exec<BLOCK_SIZE> >(RAJA::TypedRangeSegment<int>(0, N), 
    [=] RAJA_DEVICE (int i) {

    RAJA::atomicAdd< RAJA::cuda_atomic >(sum, 1);

  });

After this kernel executes, the value reference by 'sum' will be 'N'.

^^^^^^^^^^^^^^^^^^^^
AtomicRef
^^^^^^^^^^^^^^^^^^^^

RAJA also provides an interface similar to the C++20 ``std::atomic_ref``, 
but which works for arbitrary memory locations. The class 
``RAJA::AtomicRef`` provides an object-oriented interface to the 
atomic methods described above. For example, after the following operations:: 

  double val = 2.0;
  RAJA::AtomicRef<double,  RAJA::omp_atomic > sum(&val);

  sum++;
  ++sum;
  sum += 1.0; 

the value of 'val' will be 5.

.. _cudaatomics-label:

---------------------------------------
CUDA Atomics Architecture Dependencies
---------------------------------------

The implementations for RAJA atomic operations may vary depending
on which CUDA architecture is available and/or specified when RAJA
is configured for compilation. The following rules apply when the following
CUDA architecture level is chosen:

  * **CUDA architecture is lower than `sm_35`** 

    * Certain atomics will be implemented using CUDA `atomicCAS` 
      (Compare and Swap).

  * **CUDA architecture is `sm_35` or higher**   

    * CUDA native 64-bit unsigned atomicMin, atomicMax, atomicAnd, atomicOr,
      atomicXor are used.

  * **CUDA architecture is `sm_60` or higher** 

    * CUDA native 64-bit double `atomicAdd` is used.

.. _desul-atomics-label:

---------------------
DESUL Atomics Support
---------------------

RAJA provides the ability to use 
`DESUL Atomics <https://github.com/desul/desul>`_ as
an alternative to the default implementation of RAJA atomics. DESUL atomics 
are considered an **experimental** feature in RAJA at this point and may
impact the performance of some atomic functions. While DESUL atomics typically 
yields better or similar performance to RAJA default atomics, some atomic 
operations may perform worse when using DESUL.

To enable DESUL atomics, pass the option to CMake when configuring a RAJA
build: ``-DRAJA_ENABLE_DESUL_ATOMICS=On``.

Enabling DESUL atomics alters RAJA atomic functions to be wrapper-functions 
for their DESUL counterparts. This removes the need for user code changes to 
switch between DESUL and RAJA implementations for the most part. The exception 
to this is when RAJA atomic helper functions are used instead of the 
backward-compatible API functions specified by :ref:`atomic-ops`. By 
*helper functions*, we mean the RAJA atomic methods which take an atomic
policy object as the first argument, instead of specifying the atomic policy 
type as a template parameter. 

DESUL atomic functions are compiled with the proper back-end implementation 
based on the scope in which they are called, which removes the need to specify 
atomic policies for target back-ends. As a result, atomic policies such as 
``RAJA::cuda_atomic`` or ``RAJA::omp_atomic`` are ignored when DESUL is 
enabled, but are still necessary to pass in as parameters to the RAJA API. 
This will likely change in the future when we switch to use DESUL atomics
exclusively and remove the default RAJA atomic operations.
