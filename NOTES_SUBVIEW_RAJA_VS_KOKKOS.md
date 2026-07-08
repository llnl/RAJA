# RAJA SubView/SubLayout vs. Kokkos subview/submdspan

This note summarizes how the current RAJA `SubView`/`SubLayout` implementation
works, how it relates to Kokkos `subview` and `Experimental::submdspan`, and the
practical tradeoffs between the approaches.

The goal is to make it easier to reason about:
- *Logical extents* (how many index combinations exist),
- *Physical strides* (how index increments change the underlying linear index),
- and how “slicing” composes with an existing layout/view mapping.

## Terminology

### Parent mapping
A multi-dimensional view/span is essentially:
- a data handle (pointer),
- plus a mapping from indices to a linear offset.

For a 2D row-major layout with extents `(R, C)`, the parent mapping is commonly:

```
offset(i, j) = i*C + j
```

and the physical strides are:
- `stride_row = C`
- `stride_col = 1`

### Slice specifiers (common concepts)
Across RAJA/Kokkos/mdspan, you typically slice each dimension using one of:
- **Fixed index**: pick a single index (rank-reducing)
- **Range**: `[begin, end)` (keeps rank, changes extent)
- **Strided range**: `[begin, end)` with step `s` (keeps rank, changes extent and physical stride)
- **Full extent**: keep the whole dimension

## RAJA: two ways to represent a slice

RAJA currently supports two closely related patterns (both appear in the unit
tests under `test/unit/view-layout/test-subview.cpp`):

### A) “View with a SubLayout”

You can construct a normal RAJA `View` whose layout type is a `SubLayout`:

- The result is still a `View` (normal `View` API).
- The layout object is a “mapping object” representing the slice.

This is conceptually similar to mdspan/Kokkos results: “a new view/span object
with a mapping that describes the slice”.

### B) “SubView with a normal layout”

You can construct a `SubView` that stores:
- the parent view/layout (`m_parent`), and
- the slice descriptors (`m_slices`),

and computes parent indices when you call `operator()`.

This is more of a “composed mapping wrapper” than “a new view with a rewritten
mapping”.

## Kokkos: two APIs (similar semantics, different types)

### `Kokkos::subview(View, ...)`
Produces a new `Kokkos::View`-like object that aliases the original allocation:
- the data handle refers to the same allocation (often with an adjusted base
  pointer to the slice origin),
- the mapping/extents/strides reflect the slice.

Kokkos often tries to preserve a more specialized layout type (e.g. `LayoutLeft`
or `LayoutRight`) when slicing allows it; otherwise it falls back to a more
general strided mapping.

### `Kokkos::Experimental::submdspan(mdspan, ...)`
Produces a new `mdspan`-like object:
- new data handle (possibly pointer-adjusted),
- new mapping that composes parent mapping with slice transforms,
- rank reduction for fixed-index slices.

This closely matches the mdspan slicing model/specifiers and is generally
“standard-library shaped”.

## Worked example: range + strided range (2D)

Parent extents: `(3, 6)`

Assume a row-major-like mapping:
```
offset(i, j) = i*6 + j
```

Slice:
- rows: `[1, 3)`  → 2 rows (parent `i = 1, 2`)
- cols: `[1, 6)` step 2 → 3 cols (parent `j = 1, 3, 5`)

### What Kokkos/mdspan compute (conceptually)

**New logical extents**
- `sub_extent_row = 2`
- `sub_extent_col = 3`

**Slice origin (base pointer offset)**
- base offset = `offset(1, 1) = 1*6 + 1 = 7`

**New physical strides**
- sub-row step: `+1` in sub-row means `+1` in parent row → `+6` in linear offset
- sub-col step: `+1` in sub-col means `+2` in parent col → `+2` in linear offset

So the sub-mapping looks like:
```
sub_offset(r, c) = 7 + r*6 + c*2
```

### What RAJA `SubRegion` does

RAJA computes parent indices from the slice descriptors:
- parent row = `1 + r` (range slice)
- parent col = `1 + 2*c` (strided slice)

then calls the parent mapping with those parent indices.

Algebraically, it’s the same:
```
offset(1+r, 1+2c) = (1+r)*6 + (1+2c) = 7 + r*6 + c*2
```

### Similarities (important)
- Same logical extents.
- Same physical strides.
- Same underlying data, no copy.
- Same rank (2D in → 2D out).

### Where RAJA differs
- In the “SubView wrapper” form, RAJA does not necessarily precompute a base
  pointer offset; instead, it composes the mapping at call time by transforming
  indices and delegating to the parent.
- In the “View with SubLayout” form, RAJA is closer to mdspan/Kokkos in spirit:
  you have a normal view whose layout/mapping object embodies the slice.

## Worked example: fixed-index (rank reduction)

Parent extents: `(3, 6)`, mapping `offset(i, j) = i*6 + j`

Slice:
- row fixed: `i = 1` (rank reduces)
- cols: `[2, 5)` → 3 columns

Result:
- rank: 1D
- logical extent: 3
- base offset: `offset(1, 2) = 8`
- physical stride along remaining dim: 1

Kokkos/mdspan represent this as a 1D view/span.

RAJA `SubRegion` also reduces `n_dims` (it excludes reducing slices from the
subview rank).

## Logical size vs physical stride (and “logical stride”)

### Logical size (`size()`)
“How many index combinations exist in the resulting (sub)view?”

Example above (range + strided range):
- logical extents: `(2, 3)`
- logical size: `2*3 = 6`

This does **not** mean contiguous storage; it just counts valid index tuples.

### Physical stride (`get_parent_dim_stride`)
“If you increment subview index `d` by 1, how much does the *parent linear index*
change?”

In the same example:
- row physical stride: `6`
- col physical stride: `2`

This is the stride that matters for pointer arithmetic, contiguity checks, and
vectorization decisions.

### Logical (packed) stride (`get_dim_stride`)
Sometimes you also want the stride of a *hypothetical dense packed layout* of
the subview’s logical extents (e.g., for packing/copying into a contiguous
buffer). That’s what `SubRegion::get_dim_stride<D>()` provides:
- for a 2D `(rows, cols)` subview, logical strides are typically:
  - `logical_stride_row = cols`
  - `logical_stride_col = 1`

This is intentionally different from physical strides when the subview is
strided/gappy in the parent.

## RAJA-only: “projection” (dimension size 0)

RAJA `Layout` supports “projection” by allowing a dimension size of `0`. This
means that dimension does not contribute to uniqueness in the linear index.

That’s why RAJA has:
- `size()` (treat projected dim size 0 as size 1 when multiplying)
- `size_noproj()` (raw product; becomes 0 if any dimension size is 0)

Kokkos/mdspan typically represent degeneracy differently (often extent `1` or
rank reduction), and do not generally encode “projection” via extent `0` with
special rules.

## Summary of pros/cons

### RAJA (SubView wrapper)
Pros:
- Very general: composes with any “layout-like callable” parent.
- Easy to nest slices (subview-of-subview is composition).
- Keeps logic localized: slice descriptors + parent mapping.

Cons:
- More template/lookup sensitivity: the “tuple-like” machinery must find the
  right `camp::tuple_size`/`camp::get` for index containers.
- The “mapping” is not fully materialized as a simple stride/base-offset pair
  unless you add/compute those explicitly.

### RAJA (View with SubLayout)
Pros:
- Result is a normal `View` with a mapping object describing the slice.
- Conceptually closer to mdspan/Kokkos outcomes.

Cons:
- Still requires careful definition of what stride/size mean for the sublayout.
- May require more work to “canonicalize” into specialized layout forms.

### Kokkos/mdspan
Pros:
- Clear model: result is a view/span with adjusted handle + mapping.
- Strongly consistent semantics for extents/strides/rank reduction.
- Tooling/ecosystem around mdspan-style mappings is well understood.

Cons:
- More machinery to preserve specialized layout forms under slicing.
- Potentially higher compile-time complexity, especially for many slice variants.
