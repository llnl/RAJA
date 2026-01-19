.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _feat-reductions-label:

====================
Reduction Operations
====================

RAJA does not provide separate loop execution methods for loops containing
reduction operations like some other C++ loop programming abstraction models.
Instead, RAJA provides reduction types that allow users to perform reduction
operations in kernels launched using ``RAJA::forall``, ``RAJA::kernel``,
and ``RAJA::launch`` methods in a portable, thread-safe manner. Users may
use as many reduction objects in a loop kernel as they need. If a runtime number
of reductions is required in a loop kernel, then multi-reductions can be used.
Available RAJA reduction types are described in this section.

.. note:: All RAJA reduction types are located in the namespace ``RAJA``.

Also

.. note:: * Each RAJA reduction type is templated on a **reduction policy**
            and a **reduction value type** for the reduction variable. The
            **reduction policy type must be compatible with the execution
            policy used by the kernel in which it is used.** For example, in
            a CUDA kernel, a CUDA reduction policy must be used.
          * Each RAJA reduction type accepts an **initial reduction value or
            values** at construction (see below).
          * Each RAJA reduction type has a 'get' method to access reduced
            values after kernel execution completes.

Please see the following tutorial sections for detailed examples that use
RAJA reductions:

 * :ref:`tut-reduction-label`.

Please see the following sections for a description of multi-reducers:

 * :ref:`feat-multi-reductions-label`.

Please see the following cook book sections for guidance on policy usage:

 * :ref:`cook-book-reductions-label`.


----------------
Reduction Types
----------------

RAJA supports five common reduction types:

* ``ReduceSum< reduce_policy, data_type >`` - Sum of values.

* ``ReduceMin< reduce_policy, data_type >`` - Min value.

* ``ReduceMax< reduce_policy, data_type >`` - Max value.

* ``ReduceMinLoc< reduce_policy, data_type >`` - Min value and a loop index where the minimum was found.

* ``ReduceMaxLoc< reduce_policy, data_type >`` - Max value and a loop index where the maximum was found.

and two less common bitwise reduction types:

* ``ReduceBitAnd< reduce_policy, data_type >`` - Bitwise 'and' of values (i.e., ``a & b``).

* ``ReduceBitOr< reduce_policy, data_type >`` - Bitwise 'or' of values (i.e., ``a | b``).

.. note:: * When ``RAJA::ReduceMinLoc`` and ``RAJA::ReduceMaxLoc`` are used
            in a sequential execution context, the loop index of the
            min/max is the first index where the min/max occurs.
          * When these reductions are used in a parallel execution context,
            the loop index computed for the reduction value may be any index
            where the min or max occurs.

.. note:: ``RAJA::ReduceBitAnd`` and ``RAJA::ReduceBitOr`` reduction types are designed to work on integral data types because **in C++, at the language level, there is no such thing as a bitwise operator on floating-point numbers.**

-------------------
Reduction Examples
-------------------

Next, we provide a few examples to illustrate basic usage of RAJA reduction
types.

Here is a simple RAJA reduction example that shows how to use a sum reduction
type and a min-loc reduction type::

  const int N = 1000;

  //
  // Initialize array of length N with all ones. Then, set some other
  // values in the array to make the example mildly interesting...
  //
  int vec[N] = {1};
  vec[100] = -10; vec[500] = -10;

  // Create a sum reduction object with initial value of zero
  RAJA::ReduceSum< RAJA::omp_reduce, int > vsum(0);

  // Create a min-loc reduction object with initial min value of 100
  // and initial location index value of -1
  RAJA::ReduceMinLoc< RAJA::omp_reduce, int > vminloc(100, -1);

  // Run a kernel using the reduction objects
  RAJA::forall<RAJA::omp_parallel_for_exec>( RAJA::RangeSegment(0, N),
    [=](RAJA::Index_type i) {

    vsum += vec[i];
    vminloc.minloc( vec[i], i );

  });

  // After kernel is run, extract the reduced values
  int my_vsum = static_cast<int>(vsum.get());

  int my_vmin = static_cast<int>(vminloc.get());
  int my_vminloc = static_cast<int>(vminloc.getLoc());

The results of these operations will yield the following values:

 * my_vsum == 978 (= 998 - 10 - 10)
 * my_vmin == -10
 * my_vminloc == 100 or 500

Note that the location index for the minimum array value can be one of two
values depending on the order of the reduction finalization since the loop
is run in parallel. Also, note that the reduction objects are created using
a ``RAJA::omp_reduce`` reduction policy, which is compatible with the
OpenMP execution policy used in the kernel.

Here is an example of a bitwise or reduction::

  const int N = 100;

  //
  // Initialize all entries in array of length N to the value '9'
  //
  int vec[N] = {9};

  // Create a bitwise or reduction object with initial value of '5'
  RAJA::ReduceBitOr< RAJA::omp_reduce, int > my_or(5);

  // Run a kernel using the reduction object
  RAJA::forall<RAJA::omp_parallel_for_exec>( RAJA::RangeSegment(0, N),
    [=](RAJA::Index_type i) {

    my_or |= vec[i];

  });

  // After kernel is run, extract the reduced value
  int my_or_reduce_val = static_cast<int>(my_or.get());

The result of the reduction is the value '13'. In binary representation
(i.e., bits), :math:`9 = ...01001` (the vector entries) and
:math:`5 = ...00101` (the initial reduction value).
So :math:`9 | 5 = ...01001 | ...00101 = ...01101 = 13`.

-------------------
Reduction Policies
-------------------

For more information about available RAJA reduction policies and guidance
on which to use with RAJA execution policies, please see
:ref:`reducepolicy-label`.

--------------------------------
Experimental Reduction Interface
--------------------------------

An experimental reduction interface is now available that offers several
usability and performance advantages over the current reduction model in RAJA.
The new interface allows ``RAJA::forall`` to take optional "plugin-like"
objects to extend the execution behavior of a ``RAJA::forall`` execution
context.

The new interface passes ``RAJA::expt::Reduce<OP_TYPE>`` objects as function
arguments to ``RAJA::forall`` and provides users with thread-local variables
of the reduction data type to be updated inside the lambda. This differs
from the current reduction model in which ``RAJA::ReduceOP<REDUCE_POL, T>``
objects are captured by the user-supplied kernel body lambda expression.


RAJA::expt::Reduce
..................
::

  using VALOP_DOUBLE_SUM = RAJA::expt::ValOp<double, RAJA::operators::plus>;
  using VALOP_DOUBLE_MIN = RAJA::expt::ValOp<double, RAJA::operators::minimum>;

  double* a = ...;

  double rs = 0.0;
  double rm = 1e100;

  RAJA::forall<EXEC_POL> ( Res, Seg,
  RAJA::expt::Reduce<RAJA::operators::plus>(&rs),
  RAJA::expt::Reduce<RAJA::operators::minimum>(&rm),
  [=] (int i, VALOP_DOUBLE_SUM& _rs, VALOP_DOUBLE_MIN& _rm) {
    _rs += a[i];
    _rm.min(a[i]);
  }
  );

  std::cout << rs ...
  std::cout << rm ...

* Each ``RAJA::expt::Reduce`` argument to ``RAJA::forall`` is templated on
  a reduction operator, and takes a pointer to a target variable to write
  the final reduction result to, ``&rs`` and ``&rm`` in the example code
  above. The reduction operation will include the existing value of
  the given target variable.
* The kernel body lambda expression passed to ``RAJA::forall`` must have a
  ``RAJA::expt::ValOp`` parameter corresponding to each ``RAJA::expt::Reduce``
  argument, ``_rs`` and ``_rm`` in the example code. These parameters refer to a
  local target for each reduction operation. Each ``ValOp`` needs to be templated
  on the underlying data type (``double`` for ``_rs`` and ``_rm``), and the operator
  being used. It is important to note that the parameters follow the kernel iteration
  variable, ``i`` in this case, and appear in the same order as the corresponding
  ``RAJA::expt::Reduce`` arguments to ``RAJA::forall``. The ``ValOp`` parameters must
  be references to the objects instantiated by the ``RAJA::expt::Reduce`` arguments.
* The local variables referred to by ``_rs`` and ``_rm`` are initialized with
  the *identity* of the reduction operation to be performed.
* The local variables are updated in the user supplied lambda.
* The local variables are reduced to a single value, combining their values
  across all threads participating in the ``RAJA::forall`` execution.
* Finally, the target variable is updated with the result of the
  ``RAJA::forall`` reduction by performing the reduction operation to combine
  the existing value of the target variable and the result of the
  ``RAJA::forall`` reduction.
* The final reduction value is accessed by referencing the target variable
  passed to ``RAJA::expt::Reduce`` in the ``RAJA::forall`` method.

.. note:: In the above example ``Res`` is a resource object that must be
          compatible with the ``EXEC_POL``. ``Seg`` is the iteration space
          object for ``RAJA::forall``.

.. important:: * ``RAJA::expt::Reduce`` arguments must be passed to the forall.
                 These arguments are templated on the reduction operator, and take
                 a pointer to the target reduction variable that was declared outside
                 of the forall.
               * The local reduction arguments to the lambda expression must be
                 ``RAJA::expt::ValOp`` references. Each ``ValOp`` reference
                 corresponds to a ``RAJA::expt::Reduce`` argument within the forall.
               * The ordering of the ``ValOp`` references must correspond to the
                 ordering of the ``RAJA::expt::Reduce`` arguments to ensure that the
                 correct result is obtained.
               * Each ``ValOp`` reduction data type and RAJA operator need to match
                 the data type referenced, and operator template argument in the
                 corresponding ``RAJA::expt::Reduce`` argument.

RAJA::expt::ValLoc
..................

As with the current RAJA reduction interface, the new interface supports *loc*
reductions, which provide the ability to get a kernel/loop index at which the
final reduction value was found. With this new interface, *loc* reductions
are performed using ``ValLoc<T,I>`` types, where ``T`` is the underlying data type,
and ``I`` is the index type. Users must use the ``getVal()`` and ``getLoc()``
methods to access the reduction results after the kernel completes.

In the lambda expression, a ``ValLoc<T,I>`` must be wrapped in a
``ValOp`` type, and passed to the lambda in the same order as the corresponding 
``RAJA::expt::Reduce`` arguments, e.g. ``ValOp<ValLoc<T,I>, Op>``. In the example
below, ``VALOPLOC_DOUBLE_MIN`` represents a wrapped ``ValLoc`` usable within the
lambda.

For convenience, an alias of ``RAJA::expt::ValLocOp<T,I,Op>`` is provided.
Within the lambda, this ``ValLocOp`` object provides ``minloc``, and ``maxloc``
functions. In the example below, ``VALOPLOC_DOUBLE_MAX`` represents a wrapped
``ValLoc`` using the ``ValLocOp`` alias::

  double* a = ...;

  using VALOPLOC_DOUBLE_MIN = RAJA::expt::ValOp<ValLoc<double, RAJA::Index_type>,
                                                       RAJA::operators::minimum>;
  using VALOPLOC_DOUBLE_MAX = RAJA::expt::ValLocOp<double, RAJA::Index_type,
                                                   RAJA::operators::minimum>;

  using VL_DOUBLE = RAJA::expt::ValLoc<double>;
  VL_DOUBLE rmin_loc;
  VL_DOUBLE rmax_loc;

  RAJA::forall<EXEC_POL> ( Res, Seg,
  RAJA::expt::Reduce<RAJA::operators::minimum>(&rmin_loc),
  RAJA::expt::Reduce<RAJA::operators::maximum>(&rmax_loc),
  [=] (int i, VALOPLOC_DOUBLE_MIN& _rmin_loc, VALOPLOC_DOUBLE_MAX& _rmax_loc) {
    _rmin_loc.minloc(a[i], i);
    _rmax_loc.minloc(a[i], i);
  }
  );

  std::cout << rmin_loc.getVal() ...
  std::cout << rmin_loc.getLoc() ...
  std::cout << rmax_loc.getVal() ...
  std::cout << rmax_loc.getLoc() ...

Alternatively, *loc* reductions can be performed on separate reduction data, and
location variables without a ``ValLoc`` object, seen in the next example below.
To use this capability, a ``RAJA::expt::ReduceLoc`` argument must be passed to the
``RAJA::forall``, templated on the reduction operation, and passing in references to
the data and location. This is illustrated in the example below, with references to
``rm`` and ``loc`` being passed into the ``ReduceLoc`` argument in the forall. The
data and location can be accessed outside of the forall directly without
``getVal()`` or ``getLoc()`` functions.
:: 

  double* a = ...;

  using VALOPLOC_DOUBLE_MIN = RAJA::expt::ValLocOp<double, RAJA::Index_type,
                                                   RAJA::operators::minimum>;

  // No ValLoc needed from the user here.
  double rm;
  RAJA::Index_type loc;

  RAJA::forall<EXEC_POL> ( Res, Seg,
  RAJA::expt::ReduceLoc<RAJA::operators::minimum>(&rm, &loc), // --> 1 double & 1 index added
  [=] (int i, VALOPLOC_DOUBLE_MIN& _rm_loc) {
    _rm_loc.minloc(a[i], i);
  }
  );

  // No getVal() or getLoc() required. Access results in their original form.
  std::cout << rm ...
  std::cout << loc ...


Lambda Arguments
................

This interface takes advantage of C++ parameter packs to allow users to pass
any number of ``RAJA::expt::Reduce`` arguments to the ``RAJA::forall`` method::

  double* a = ...;

  using VALOP_DOUBLE_SUM = RAJA::expt::ValOp<double, RAJA::operators::plus>;
  using VALOP_DOUBLE_MIN = RAJA::expt::ValOp<double, RAJA::operators::minimum>;
  using VALOPLOC_DOUBLE_MIN = RAJA::expt::ValLocOp<double, RAJA::Index_type, RAJA::operators::minimum>;

  using VL_DOUBLE = RAJA::expt::ValLoc<double>;
  VL_DOUBLE rm_loc;
  double rs;
  double rm;

  RAJA::forall<EXEC_POL> ( Res, Seg,
    RAJA::expt::Reduce<RAJA::operators::plus>(&rs),        // --> 1 double added
    RAJA::expt::Reduce<RAJA::operators::minimum>(&rm),     // --> 1 double added
    RAJA::expt::Reduce<RAJA::operators::minimum>(&rm_loc), // --> 1 VL_DOUBLE added
    RAJA::Name("MyFirstRAJAKernel"),           // --> NO args added
    [=] (int i,
         VALOP_DOUBLE_SUM& _rs,
         VALOP_DOUBLE_MIN& _rm,
         VALOPLOC_DOUBLE_MIN& _rm_loc) {
      _rs += a[i];
      _rm.min(a[i]);
      _rm_loc.minloc(a[i], i);
    }
  );

  std::cout << rs ...
  std::cout << rm ...
  std::cout << rm_loc.getVal() ...
  std::cout << rm_loc.getLoc() ...

Again, the lambda expression parameters are in the same order as
the ``RAJA::expt::Reduce`` arguments to ``RAJA::forall``. The ``ValOp`` underlying
data types and operators, and order of the ``ValOp`` parameters must match 
the corresponding ``RAJA::expt::Reduce`` types to get correct results and to
compile successfully. Otherwise, a static assertion will be triggered::

  LAMBDA Not invocable w/ EXPECTED_ARGS. Ordering and types must match between RAJA::expt::Reduce() and ValOp arguments.

.. note:: This static assert is only enabled when passing an undecorated C++
          lambda. Meaning, this check will not happen when passing
          extended-lambdas (i.e. DEVICE tagged lambdas) or other functor like
          objects.

.. note:: The experimental ``RAJA::forall`` interface is more flexible than the
          current implementation, other optional arguments besides
          ``RAJA::expt::Reduce`` can be passed to a ``RAJA::forall`` to extend
          its behavior. In the above example we demonstrate using
          ``RAJA::Name``, which wraps a ``RAJA::forall`` executing
          under a ``HIP`` or ``CUDA`` policy in a named region. Use of
          ``RAJA::Name`` does not require an additional
          parameter in the lambda expression.


Experimental reduction support in Launch
........................................

The experimental reduction interface is also supported with the ``RAJA::launch`` API.
The usage of the experiemental reductions is similar to the forall example as illustrated below::

  double* a = ...;

  using VALOP_DOUBLE_SUM = RAJA::expt::ValOp<double, RAJA::operators::plus>;
  using VALOP_DOUBLE_MIN = RAJA::expt::ValOp<double, RAJA::operators::minimum>;

  double rs = 0.0;
  double rm = 1e100;

  RAJA::launch<EXEC_POL> ( Res,
    RAJA::expt::Reduce<RAJA::operators::plus>(&rs),
    RAJA::expt::Reduce<RAJA::operators::minimum>(&rm),
    RAJA::Name("LaunchReductionKernel"),
    [=] RAJA_HOST_DEVICE (int i, VALOP_DOUBLE_SUM& _rs, VALOP_DOUBLE_MIN& _rm) {

      RAJA::loop<loop_pol>(ctx, Seg, [&] (int i) {

        _rs += a[i];
        _rm.min(a[i], _rm);

        }
      );

    }
  );

  std::cout << rs ...
  std::cout << rm ...

All experimental reduction operators are supported within ``RAJA::launch``. Kernel naming in
launch does currently employ a different strategy as illustrated above but will be unified in a future
release. A wide range of examples may be found under ``examples/launch-param-reductions.cpp``.
