.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _feat-view-label:

===============
View and Layout
===============

Matrices and tensors, which are common in scientific computing applications,
are naturally expressed as multi-dimensional arrays. However, for efficiency
in C and C++, they are usually allocated as one-dimensional arrays.
For example, a matrix :math:`A` of dimension :math:`N_r \times N_c` is
typically allocated as::

   double* A = new double [N_r * N_c];

Using a one-dimensional array makes it necessary to convert
two-dimensional indices (rows and columns of a matrix) to a one-dimensional
pointer offset to access the corresponding array memory location. One
could use a macro such as::

   #define A(r, c) A[c + N_c * r]

to access a matrix entry in row `r` and column `c`. However, this solution has
limitations; e.g., additional macro definitions may be needed when adopting a
different matrix data layout or when using other matrices. To facilitate
multi-dimensional indexing and different indexing layouts, RAJA provides
``RAJA::View``, ``RAJA::Layout``, and ``RAJA::OffsetLayout`` classes.

Please see the following tutorial sections for detailed examples that use
RAJA Views and Layouts:

 * :ref:`tut-view_layout-label`
 * :ref:`tut-offsetlayout-label`
 * :ref:`tut-permutedlayout-label`
 * :ref:`tut-kernelexecpols-label`
 * :ref:`tut-launchexecpols-label`

----------
RAJA Views
----------

A ``RAJA::View`` object wraps a pointer and enables indexing into the data
referenced via the pointer based on a ``RAJA::Layout`` object. We can
create a ``RAJA::View`` for a matrix with dimensions :math:`N_r \times N_c`
using a RAJA View and a default RAJA two-dimensional Layout as follows::

   double* A = new double [N_r * N_c];

   const int DIM = 2;
   RAJA::View<double, RAJA::Layout<DIM> > Aview(A, N_r, N_c);

The ``RAJA::View`` constructor takes a pointer to the matrix data and the
extent of each matrix dimension as arguments. The template parameters to
the ``RAJA::View`` type define the pointer type and the Layout type; here,
the Layout just defines the number of index dimensions. Using the resulting
view object, one may access matrix entries in a row-major fashion (the
default RAJA layout follows the C and C++ standards for multi-dimensional
arrays) through the view *parenthesis operator*::

   // r - row index of matrix
   // c - column index of matrix
   // equivalent to indexing as A[c + r * N_c]
   Aview(r, c) = ...;

A ``RAJA::View`` can support any number of index dimensions::

   const int DIM = n+1;
   RAJA::View< double, RAJA::Layout<DIM> > Aview(A, N0, ..., Nn);

By default, entries corresponding to the right-most index are contiguous
in memory; i.e., unit-stride access. Each other index is offset by the
product of the extents of the dimensions to its right. For example, the loop::

   // iterate over index n and hold all other indices constant
   for (int in = 0; in < Nn; ++in) {
     Aview(i0, i1, ..., in) = ...
   }

accesses array entries with unit stride. The loop::

   // iterate over index j and hold all other indices constant
   for (int j = 0; j < Nj; ++j) {
     Aview(i0, i1, ..., j, ..., iN) = ...
   }

access array entries with stride N :subscript:`n` * N :subscript:`(n-1)` * ... * N :subscript:`(j+1)`.

MultiView
^^^^^^^^^^^^^^^^

Using numerous arrays with the same size and Layout, where each needs
a View, can be cumbersome. Developers need to create a View object for
each array, and when using the Views in a kernel, they require redundant
pointer offset calculations. ``RAJA::MultiView`` solves these problems by
providing a way to create many Views with the same Layout in one instantiation,
and operate on an array-of-pointers that can be used to succinctly access
data.

A ``RAJA::MultiView`` object wraps an array-of-pointers,
or a pointer-to-pointers, whereas a ``RAJA::View`` wraps a single
pointer or array. This allows a single ``RAJA::Layout`` to be applied to
multiple arrays associated with the MultiView, allowing the arrays to share
indexing arithmetic when their access patterns are the same.

The instantiation of a MultiView works exactly like a standard View,
except that it takes an array-of-pointers. In the following example, a MultiView
applies a 1-D layout of length 4 to 2 arrays in ``myarr``.

.. literalinclude:: ../../../../examples/multiview.cpp
   :start-after: _multiview_example_1Dinit_start
   :end-before: _multiview_example_1Dinit_end
   :language: C++

The default MultiView accesses individual arrays via the 0-th position of the
MultiView.

.. literalinclude:: ../../../../examples/multiview.cpp
   :start-after: _multiview_example_1Daccess_start
   :end-before: _multiview_example_1Daccess_end
   :language: C++

The index into the array-of-pointers can be moved to different argument
positions of the MultiView ``()`` access operator, rather than the default
0-th position. For example, by passing a third template argument to the
MultiView constructor in the previous example, the internal array index and
the integer indicating which array to access can be reversed.

.. literalinclude:: ../../../../examples/multiview.cpp
   :start-after: _multiview_example_1Daopindex_start
   :end-before: _multiview_example_1Daopindex_end
   :language: C++

With higher dimensional Layouts, the index into the array-of-pointers can be
moved to other positions in the MultiView ``()`` access operator. Here is an
example that compares the accesses of a 2-D layout on a normal ``RAJA::View``
with a ``RAJA::MultiView`` with the array-of-pointers index set to the 2nd
position.

.. literalinclude:: ../../../../examples/multiview.cpp
   :start-after: _multiview_example_2Daopindex_start
   :end-before: _multiview_example_2Daopindex_end
   :language: C++


------------
RAJA Layouts
------------

``RAJA::Layout`` objects support other indexing patterns with different
striding orders, offsets, and permutations. In addition to layouts created
using the default Layout constructor, as shown above, RAJA provides other
methods to generate layouts for different indexing patterns. We describe
them here.

.. _permuted-layout-label:

Permuted Layout
^^^^^^^^^^^^^^^^

The ``RAJA::make_permuted_layout`` method creates a ``RAJA::Layout`` object
with permuted index strides. That is, the indices with shortest to
longest stride are permuted. For example,::

  std::array< RAJA::idx_t, 3> perm {{1, 2, 0}};
  RAJA::Layout<3> layout =
    RAJA::make_permuted_layout( {{5, 7, 11}}, perm );

creates a three-dimensional layout with index extents 5, 7, 11 with
indices permuted so that the first index (index 0 - extent 5) has unit
stride, the third index (index 2 - extent 11) has stride 5, and the
second index (index 1 - extent 7) has stride 55 (= 5*11).

.. note:: If a permuted layout is created with the *identity permutation*
          (e.g., {0,1,2}), the layout is the same as if it were created by
          calling the Layout constructor directly with no permutation.

The first argument to ``RAJA::make_permuted_layout`` is a C++ array whose
entries define the extent of each index dimension. **The double braces are
required to properly initialize the internal sub-object which holds the
extents.** The second argument is the striding permutation and similarly
requires double braces.

In the next example, we create the same permuted layout as above, then create
a ``RAJA::View`` with it in a way that tells the view which index has
unit stride::

  const int s0 = 5;  // extent of dimension 0
  const int s1 = 7;  // extent of dimension 1
  const int s2 = 11; // extent of dimension 2

  double* B = new double[s0 * s1 * s2];

  std::array< RAJA::idx_t, 3> perm {{1, 2, 0}};
  RAJA::Layout<3> layout =
    RAJA::make_permuted_layout( {{s0, s1, s2}}, perm );

  // The Layout template parameters are dimension, 'linear index' type used
  // when converting an index triple into the corresponding pointer offset
  // index, and the index with unit stride
  RAJA::View<double, RAJA::Layout<3, int, 0> > Bview(B, layout);

  // Equivalent to indexing as: B[i + j * s0 * s2 + k * s0]
  Bview(i, j, k) = ...;

.. note:: Telling a view which index has unit stride makes the
          multi-dimensional index calculation more efficient by avoiding
          multiplication by '1' when it is unnecessary. **The layout
          permutation and unit-stride index specification
          must be consistent to prevent incorrect indexing.**

Offset Layout
^^^^^^^^^^^^^^^^

The ``RAJA::make_offset_layout`` method creates a ``RAJA::OffsetLayout`` object
with offsets applied to the indices. For example,::

  double* C = new double[10];

  RAJA::Layout<1> layout = RAJA::make_offset_layout<1>( {{-5}}, {{5}} );

  RAJA::View<double, RAJA::OffsetLayout<1> > Cview(C, layout);

creates a one-dimensional view with a layout that allows one to index into
it using indices in :math:`[-5, 5)`. In other words, one can use the loop::

  for (int i = -5; i < 5; ++i) {
    CView(i) = ...;
  }

to initialize the values of the array. Each 'i' loop index value is converted
to an array offset index by subtracting the lower offset from it; i.e., in
the loop, each 'i' value has '-5' subtracted from it to properly access the
array entry. That is, the sequence of indices generated by the for-loop::

  -5 -4 -3 ... 4

will index into the data array as::

  0 1 2 ... 9

The arguments to the ``RAJA::make_offset_layout`` method are C++ arrays that
hold the begin-end values of indices in the half-open interval
:math:[begin, end)`. RAJA offset layouts support any number of dimensions;
for example::

  RAJA::OffsetLayout<2> layout =
     RAJA::make_offset_layout<2>({{-1, -5}}, {{2, 5}});

defines a two-dimensional layout that enables one to index into a view using
indices :math:`[-1, 2)` in the first dimension and indices :math:`[-5, 5)` in
the second dimension. As noted earlier, double braces are needed to
properly initialize the internal data in the layout object.

Permuted Offset Layout
^^^^^^^^^^^^^^^^^^^^^^^^

The ``RAJA::make_permuted_offset_layout`` method creates a
``RAJA::OffsetLayout`` object with permutations and offsets applied to the
indices. For example,::

  std::array< RAJA::idx_t, 2> perm {{1, 0}};
  RAJA::OffsetLayout<2> layout =
    RAJA::make_permuted_offset_layout<2>( {{-1, -5}}, {{2, 5}}, perm );

Here, the two-dimensional index space is :math:`[-1, 2) \times [-5, 5)`, the
same as above. However, the index strides are permuted so that the first
index (index 0) has unit stride and the second index (index 1) has stride 3,
which is the extent of the first index (:math:`[-1, 2)`).

.. note:: It is important to note some facts about RAJA layout types.
          All layouts have a permutation. So a permuted layout and
          a "non-permuted" layout (i.e., default permutation) has the
          type ``RAJA::Layout``. Any layout with an offset has the
          type ``RAJA::OffsetLayout``. The ``RAJA::OffsetLayout`` type has
          a ``RAJA::Layout`` and offset data. This was an intentional design
          choice to avoid the overhead of offset computations in the
          ``RAJA::View`` data access operator when they are not needed.

Complete examples illustrating ``RAJA::Layouts`` and ``RAJA::Views``  may
be found in the :ref:`tut-offsetlayout-label` and :ref:`tut-permutedlayout-label`
tutorial sections.

Typed Layouts
^^^^^^^^^^^^^

RAJA provides typed variants of ``RAJA::Layout`` and ``RAJA::OffsetLayout``
that enable users to specify integral index types. Usage requires
specifying types for the linear index and the multi-dimensional indicies.
The following example creates two two-dimensional typed layouts where the
linear index is of type TIL and the '(x, y)' indices for accessing the data
have types TIX and TIY::

   RAJA_INDEX_VALUE(TIX, "TIX");
   RAJA_INDEX_VALUE(TIY, "TIY");
   RAJA_INDEX_VALUE(TIL, "TIL");

   RAJA::TypedLayout<TIL, RAJA::tuple<TIX,TIY>> layout(10, 10);
   RAJA::TypedOffsetLayout<TIL, RAJA::tuple<TIX,TIY>> offLayout(10, 10);;

.. note:: Using the ``RAJA_INDEX_VALUE`` macro to create typed indices
          is helpful to prevent incorrect usage by detecting at compile
          when, for example, indices are passes to a view parenthesis
          operator in the wrong order.

Shifting Views
^^^^^^^^^^^^^^

RAJA views include a shift method enabling users to generate a new view with
offsets to the base view layout. The base view may be templated with either a
standard layout or offset layout and their typed variants. The new view will
use an offset layout or typed offset layout depending on whether the base
view employed a typed layout. The example below illustrates shifting view
indices by :math:`N`, ::

  int N_r = 10;
  int N_c = 15;
  int *a_ptr = new int[N_r * N_c];

  RAJA::View<int, RAJA::Layout<DIM>> A(a_ptr, N_r, N_c);
  RAJA::View<int, RAJA::OffsetLayout<DIM>> Ashift = A.shift( {{N,N}} );

  for(int y = N; y < N_c + N; ++y) {
    for(int x = N; x < N_r + N; ++x) {
      Ashift(x,y) = ...
    }
  }

Index Layout
^^^^^^^^^^^^

``RAJA::IndexLayout`` is a layout that can use an index list to map input
indices to an entry within a view.  Each dimension of the layout is required to
have its own indexing strategy to determine this mapping.

Three indexing strategies are natively supported in RAJA: ``RAJA::DirectIndex``,
``RAJA::IndexList``, and ``RAJA::ConditionalIndexList``.  ``DirectIndex``
maps an input index to itself, and does not take any  arguments in its
constructor.  The ``IndexList`` strategy takes a pointer  to an array of
indices.  With this strategy, a given input index is mapped to  the entry in its
list corresponding to that index.  Lastly, the
``ConditionalIndexStrategy`` takes a pointer to an array of indices. When
the pointer is not a null pointer, the ``ConditionalIndex`` strategy is
equivalent to that of the ``IndexList``.  If the index list provided to
the constructor is a null pointer, the ``ConditionalIndexList`` is
identical to the ``DirectIndex`` strategy.  The
``ConditionalIndexList`` strategy is useful when the index list is not
initialized for some situations.

A simple illustrative example is shown below::

  int data[2][3];

  for (int i = 0; i < 2; i ++ ) {
    for (int j = 0; j < 3; j ++ ) {
      // fill data[i][j]...
    }
  }

  int index_list[2] = {1,2};

  auto index_tuple = RAJA::tuple<RAJA::DirectIndex<>, RAJA::IndexList<>>(
                      RAJA::DirectIndex<>(), RAJA::IndexList<>{&index_list[0]});

  auto index_layout = RAJA::make_index_layout(index_tuple, 2, 3);
  auto view = RAJA::make_index_view(&data[0][0], index_layout);

  assert( view(1,0) == data[1][1] );
  assert( &view(1,1) == &data[1][2] );

In the above example, a two-dimensional index layout is created with extents 2
and 3 for the first and second dimension, respectively.  A ``DirectIndex``
strategy is implemented for the first dimension and ``IndexList`` is used
with the entries for the second dimension with the list {1,2}.  With this
layout, the view created above will choose the entry along the first dimension
based on the first input index provided, and the second provided index will be
mapped to that corresponding entry of the index_list for the second dimension.

.. note::  There is currently no bounds checking implemented for
	   ``IndexLayout``.  When using the ``IndexList`` or
	   ``ConditionalIndexList``  strategies, it is the user's
	   responsibility to know the extents of the index lists when accessing
	   data from a view.  It is also the  user's responsibility to ensure
	   the index lists being used reside in  the same memory space as the
	   data stored in the view.

------------------
Make Permuted View
------------------

The ``RAJA::make_permuted_view`` function takes a pointer to data and extents in each dimension as arguments and returns a ``RAJA::View`` object.
The function is specialized on a `meta-layout` template parameter, which specifies the stride ordering for the dimensions.

The first meta-layout is based on a right-most layout which follows standard C indexing (right most index has unit stride).
For example:

.. literalinclude:: ../../../../examples/make_permuted_view.cpp
   :start-after: _Make_Permuted_View_right_start
   :end-before: _Make_Permuted_View_right_end
   :language: C++

In the example above a ``RAJA::View`` is returned with right most indexing having unit stride.

The second supported meta-layout is the ``RAJA::layout_left`` which will follow a Fortran style indexing
in which the left most index has unit stride and the right most has the longest stride. The configuration
is demonstrated in the following example.

.. literalinclude:: ../../../../examples/make_permuted_view.cpp
   :start-after: _Make_Permuted_View_left_start
   :end-before: _Make_Permuted_View_left_end
   :language: C++

Users may also specify a custom index ordering by providing a ``std::index_sequence`` type as a
template argument to the ``RAJA::make_permuted_view`` function. The members of the ``std::index_sequence`` enumerate
the striding order of the arguments of the ``RAJA::View`` parenthesis operator. Using the ``std::index_sequence<1U, 2U, 0U>``
in a ``RAJA::make_permuted_view`` function will construct the same ``RAJA::View`` object as described in Section :ref:`permuted-layout-label`.

As an additional example to use the C index ordering (``RAJA::layout_right``) with a 3-dimensional ``RAJA::View`` use the
following  sequence type ``std::index_sequence<0U,1U,2U>``.  In our final example, we consider a permutation
in which index 1 is placed in the right most position of the sequence so it has unit stride. Finally, since index 2 is placed in
the left most position of the sequence it has the longest stride.

.. literalinclude:: ../../../../examples/make_permuted_view.cpp
   :start-after: _Make_Permuted_View_custom_start
   :end-before: _Make_Permuted_View_custom_end
   :language: C++

The file ```RAJA/examples/make_permuted_view.cpp`` contains full working code demonstrating configuring and using
the ``RAJA::make_permuted_view`` function to generate a ``RAJA::View``.

-------------------
RAJA Index Mapping
-------------------

``RAJA::Layout`` objects can also be used to map multi-dimensional indices
to *linear indices* (i.e., pointer offsets) and vice versa. This
section describes basic Layout methods that are useful for converting between
such indices. Here, we create a three-dimensional layout
with dimension extents 5, 7, and 11 and illustrate mapping between a
three-dimensional index space to a one-dimensional linear space::

   // Create a 5 x 7 x 11 three-dimensional layout object
   RAJA::Layout<3> layout(5, 7, 11);

   // Map from 3-D index (2, 3, 1) to the linear index
   // Note that there is no striding permutation, so the rightmost index is
   // stride-1
   int lin = layout(2, 3, 1); // lin = 188 (= 1 + 3 * 11 + 2 * 11 * 7)

   // Map from linear index to 3-D index
   int i, j, k;
   layout.toIndices(lin, i, j, k); // i,j,k = {2, 3, 1}

RAJA layouts also support *projections*, where one or more dimension
extent is zero. In this case, the linear index space is invariant for
those index entries; thus, the 'toIndicies(...)' method will always return
zero for each dimension with zero extent. For example::

   // Create a layout with second dimension extent zero
   RAJA::Layout<3> layout(3, 0, 5);

   // The second (j) index is projected out
   int lin1 = layout(0, 10, 0);   // lin1 = 0
   int lin2 = layout(0, 5, 1);    // lin2 = 1

   // The inverse mapping always produces zero for j
   int i,j,k;
   layout.toIndices(lin2, i, j, k); // i,j,k = {0, 0, 1}

-------------------
RAJA Atomic Views
-------------------

Any ``RAJA::View`` object can be made *atomic* so that any update to a
data entry accessed via the view can only be performed one thread (CPU or GPU)
at a time. For example, suppose you have an integer array of length N, whose
element values are in the set {0, 1, 2, ..., M-1}, where M < N. You want to
build a histogram array of length M such that the i-th entry in the array is
the number of occurrences of the value i in the original array. Here is one
way to do this in parallel using OpenMP and a RAJA atomic view::

  using EXEC_POL = RAJA::omp_parallel_for_exec;
  using ATOMIC_POL = RAJA::omp_atomic

  int* array = new double[N];
  int* hist_dat = new double[M];

  // initialize array entries to values in {0, 1, 2, ..., M-1}...
  // initialize hist_dat to all zeros...

  // Create a 1-dimensional view for histogram array
  RAJA::View<int, RAJA::Layout<1> > hist_view(hist_dat, M);

  // Create an atomic view into the histogram array using the view above
  auto hist_atomic_view = RAJA::make_atomic_view<ATOMIC_POL>(hist_view);

  RAJA::forall< EXEC_POL >(RAJA::RangeSegment(0, N), [=] (int i) {
    hist_atomic_view( array[i] ) += 1;
  } );

Here, we create a one-dimensional view for the histogram data array. Then,
we create an atomic view from that, which we use in the RAJA loop to
compute the histogram entries. Since the view is atomic, only one OpenMP
thread can write to each array entry at a time.

------------------------------------
RAJA View/Layouts Bounds Checking
------------------------------------

The RAJA CMake variable ``RAJA_ENABLE_BOUNDS_CHECK`` may be used to turn on/off
runtime bounds checking for RAJA views. This may be a useful debugging aid for
users. When attempting to use an index value that is out of bounds,
RAJA will abort the program and print the index that is out of bounds and
the value of the index and bounds for it. Since the bounds checking is a runtime
operation, it incurs non-negligible overhead. When bounds checking is turned
off (default case), there is no additional run time overhead incurred.
