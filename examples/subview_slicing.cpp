//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cassert>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/SubView.hpp"

/*
  Simple, presentation-friendly RAJA subview examples.

  Goal: you can reason about everything by inspection.
  - Use a tiny 2D parent array (3x4) written out explicitly.
  - Values are NOT the linear index, so you can clearly distinguish:
      - a mapping result (a linear offset / "index"), from
      - a view result    (the value stored at that offset).

  We show the two equivalent “styles” that RAJA supports:

  Style A: "View with a sublayout"
    - Build a SubRegion from the PARENT LAYOUT and slice descriptors.
    - Use that SubRegion as the layout type in a normal RAJA::View.

  Style B: "Subview wrapper"
    - Build a SubRegion from the PARENT VIEW and slice descriptors.
    - When you index the subview, it computes parent indices then calls the parent.

  Under the hood, both styles use the same implementation, but you will see
  them in user code as:
    - RAJA::SubLayout<...> : "sliced mapping object" (parent is a layout)
    - RAJA::SubView<...>   : "sliced view wrapper"  (parent is a view)

  Slice specifiers (in `include/RAJA/util/SubView.hpp`):
    - RangeSlice{start,end}: keeps dimension, map_index(sub) = start + sub
    - FixedSlice{idx}      : reduces dimension, map_index()    = idx
*/

namespace
{

constexpr int R = 3;
constexpr int C = 4;

}  // namespace

// These helpers mirror `test/unit/view-layout/test-subview.cpp` so the example
// reads like the unit tests.
//
// A) Create a normal RAJA View whose layout type is a sliced SubLayout.
template <typename ViewType, typename... Slices>
RAJA_HOST_DEVICE auto make_view_with_sublayout(ViewType& view, Slices... slices)
{
  using SubLayoutType =
      RAJA::SubLayout<typename ViewType::layout_type, camp::list<Slices...>>;
  return RAJA::View<typename ViewType::value_type, SubLayoutType>(
      view.get_data(), SubLayoutType(view.get_layout(), slices...));
}

// B) Create a sliced SubView wrapper without modifying the underlying layout.
template <typename ViewType, typename... Slices>
RAJA_HOST_DEVICE auto make_subview_with_layout(ViewType& view, Slices... slices)
{
  using SubViewType = RAJA::SubView<ViewType, camp::list<Slices...>>;
  return SubViewType(view, slices...);
}

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv))
{
  // Parent data written out explicitly (this is what the underlying memory
  // looks like).
  //
  // Note: the RAJA::Layout<2>(R,C) used below is row-major, so the linear
  // memory order is:
  //   a[0][0], a[0][1], a[0][2], a[0][3], a[1][0], ..., a[2][3]
  RAJA::Index_type a[R][C] = {
      {10, 11, 12, 13},
      {20, 21, 22, 23},
      {30, 31, 32, 33},
  };

  // Parent view (2D):
  //   parent(i,j) returns the stored value a[i][j] (not the linear index).
  //
  // As a grid:
  //   row 0: 10 11 12 13
  //   row 1: 20 21 22 23
  //   row 2: 30 31 32 33
  using ParentLayout = RAJA::Layout<2>;
  using ParentView = RAJA::View<RAJA::Index_type, ParentLayout>;
  ParentView parent(&a[0][0], ParentLayout(R, C));

  //===========================================================================
  // Example 1: 2D -> 2D (RangeSlice + RangeSlice), same result in both styles
  //===========================================================================
  {
    // Slice: rows [1,3), cols [1,4)
    //
    // Expected values (2x3):
    //   row 0: 21 22 23   (this is parent row=1, cols 1..3)
    //   row 1: 31 32 33   (this is parent row=2, cols 1..3)

    // Construct slices inline (like the unit tests), so the slice specification
    // is right at the call site.
    auto A = make_view_with_sublayout(parent, RAJA::RangeSlice<>{1, 3}, RAJA::RangeSlice<>{1, 4});
    auto B = make_subview_with_layout(parent, RAJA::RangeSlice<>{1, 3}, RAJA::RangeSlice<>{1, 4});

    // What the SubView/SubLayout internals precompute (compile-time constants)
    // for slices = (RangeSlice, RangeSlice):
    //
    //   s_num_slices = 2
    //   n_dims       = 2   (no rank reduction)
    //
    //   s_slice_to_parent_map (length 2):
    //     "for each slice index, which subview index should be used?"
    //     both slices keep dimension -> {0, 1}
    //
    //   s_parent_to_slice_map (length n_dims=2):
    //     "for each subview dim, which slice provides its size/stride?"
    //     -> {0, 1}
    //
    // Conceptually:
    //   B(sub_i, sub_j):
    //     parent_i = rows.map_index(sub_i) = 1 + sub_i
    //     parent_j = cols.map_index(sub_j) = 1 + sub_j
    //     return parent(parent_i, parent_j)
    //
    // Concrete mapping point (use this in slides):
    //   sub(1,2) -> parent_indices {2,3}
    //
    // Mapping result (linear offset / "index"):
    //   parent_layout(2,3) = 2*C + 3 = 2*4 + 3 = 11
    //
    // Stored value at that offset:
    //   a[2][3] = 33
    //
    // Key point:
    //   - `A.get_layout()(...)` returns an index (because layouts are mappings).
    //   - `A(...)` and `B(...)` return values (because they are views).
    assert(A.get_layout()(1, 2) == 11);  // index
    assert(A(1, 2) == 33);          // value
    assert(B(1, 2) == 33);          // value
  }

  //===========================================================================
  // Example 2: 2D -> 1D (FixedSlice + RangeSlice), rank reduction
  //===========================================================================
  {
    // Slice: fix row = 1, keep cols [0,4)
    //
    // This is rank reduction (2D -> 1D) because FixedSlice reduces a dimension.
    //
    // Expected values (length 4):
    //   parent(1,0..3) = 20 21 22 23

    // subview = view[1,0:4]
    auto A = make_view_with_sublayout(parent, RAJA::FixedSlice<>{1}, RAJA::RangeSlice<>{0, 4});
    auto B = make_subview_with_layout(parent, RAJA::FixedSlice<>{1}, RAJA::RangeSlice<>{0, 4});

    // What the SubView/SubLayout internals precompute for slices = (FixedSlice, RangeSlice):
    //
    //   s_num_slices = 2
    //   n_dims       = 1   (FixedSlice reduces rank)
    //
    //   s_slice_to_parent_map (length 2):
    //     "for each slice index, which subview index should be used?"
    //     slice 0 reduces -> stored as 0 (placeholder; unused for FixedSlice)
    //     slice 1 keeps   -> uses subview idx 0
    //     -> {0, 0}
    //
    //   s_parent_to_slice_map (length n_dims=1):
    //     "for each subview dim, which slice provides its size/stride?"
    //     the only remaining subview dim corresponds to slice index 1
    //     -> {1}

    // Internals (what gets computed) for B(sub_j):
    //   parent_i = fixed_row.map_index() = 1
    //   parent_j = cols.map_index(sub_j) = 0 + sub_j = sub_j
    //   return parent(parent_i, parent_j)
    //
    // Concrete mapping point:
    //   sub(2) -> parent_indices {1,2}
    //
    // Mapping result (linear offset / "index"):
    //   parent_layout(1,2) = 1*C + 2 = 1*4 + 2 = 6
    //
    // Stored value:
    //   a[1][2] = 22
    assert(A.get_layout()(2) == 6);  // index
    assert(A(2) == 22);         // value
    assert(B(2) == 22);         // value
  }

  return 0;
}

/*
RangeSlice<>     r{1,4};      // [1,4)
RangeStartSlice<>rs{2};       // [2,end)
NoSlice<>        all{};       // [:]
StridedSlice<>   s{1,6,2};    // [1,6) step 2
FixedSlice<>     f{1};        // [1] (rank-reducing)
*/
