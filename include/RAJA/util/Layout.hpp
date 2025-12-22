/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining Layout, a N-dimensional index calculator
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_LAYOUT_HPP
#define RAJA_LAYOUT_HPP

#include "RAJA/config.hpp"

#include <cassert>
#include <iostream>
#include <limits>

#include "RAJA/index/IndexValue.hpp"

#include "RAJA/internal/foldl.hpp"

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/Permutations.hpp"

namespace RAJA
{

namespace detail
{


template<typename Range,
         typename IdxLin        = Index_type,
         ptrdiff_t StrideOneDim = -1>
struct LayoutBase_impl;

/*!
 * Helper function to compute the strides
 */

template<size_t j, size_t n_dims, typename IdxLin = Index_type>
struct stride_calculator
{
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      IdxLin cur_stride,
      IdxLin const (&sizes)[n_dims]) const
  {
    return stride_calculator<j + 1, n_dims, IdxLin> {}(
        cur_stride * (sizes[j] ? sizes[j] : 1), sizes);
  }
};

template<size_t n_dims, typename IdxLin>
struct stride_calculator<n_dims, n_dims, IdxLin>
{
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      IdxLin cur_stride,
      IdxLin const (&)[n_dims]) const
  {
    return cur_stride;
  }
};

template<camp::idx_t... RangeInts, typename IdxLin, ptrdiff_t StrideOneDim>
struct LayoutBase_impl<camp::idx_seq<RangeInts...>, IdxLin, StrideOneDim>
{
public:
  using IndexLinear = IdxLin;
  using IndexRange  = camp::make_idx_seq_t<sizeof...(RangeInts)>;

  static inline constexpr size_t n_dims = sizeof...(RangeInts);
  static inline constexpr IdxLin limit = RAJA::operators::limits<IdxLin>::max();
  static inline constexpr ptrdiff_t stride_one_dim = StrideOneDim;

  IdxLin sizes[n_dims]       = {0};
  IdxLin strides[n_dims]     = {0};
  IdxLin inv_strides[n_dims] = {0};
  IdxLin inv_mods[n_dims]    = {0};


  /*!
   * Default constructor with zero sizes and strides.
   */
  RAJA_INLINE constexpr LayoutBase_impl()                       = default;
  RAJA_INLINE constexpr LayoutBase_impl(LayoutBase_impl const&) = default;
  RAJA_INLINE constexpr LayoutBase_impl(LayoutBase_impl&&)      = default;
  RAJA_INLINE constexpr LayoutBase_impl& operator=(LayoutBase_impl const&) =
      default;
  RAJA_INLINE constexpr LayoutBase_impl& operator=(LayoutBase_impl&&) = default;

  /*!
   * Construct a layout given the size of each dimension.
   */
  template<typename... Types>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr LayoutBase_impl(Types... ns)
      : sizes {static_cast<IdxLin>(stripIndexType(ns))...},
        strides {(detail::stride_calculator<RangeInts + 1, n_dims, IdxLin> {}(
            sizes[RangeInts] ? IdxLin(1) : IdxLin(0),
            sizes))...},
        inv_strides {(strides[RangeInts] ? strides[RangeInts] : IdxLin(1))...},
        inv_mods {(sizes[RangeInts] ? sizes[RangeInts] : IdxLin(1))...}
  {
    static_assert(n_dims == sizeof...(Types),
                  "number of dimensions must match");
  }

  /*!
   *  Templated copy ctor from simillar layout.
   */
  template<typename CIdxLin, ptrdiff_t CStrideOneDim>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr LayoutBase_impl(
      const LayoutBase_impl<camp::idx_seq<RangeInts...>,
                            CIdxLin,
                            CStrideOneDim>& rhs)
      : sizes {static_cast<IdxLin>(rhs.sizes[RangeInts])...},
        strides {static_cast<IdxLin>(rhs.strides[RangeInts])...},
        inv_strides {static_cast<IdxLin>(rhs.inv_strides[RangeInts])...},
        inv_mods {static_cast<IdxLin>(rhs.inv_mods[RangeInts])...}
  {}

  /*!
   *  Construct a Layout given the size and stride of each dimension
   */
  template<typename... Types>
  RAJA_INLINE constexpr LayoutBase_impl(
      const std::array<IdxLin, n_dims>& sizes_in,
      const std::array<IdxLin, n_dims>& strides_in)
      : sizes {sizes_in[RangeInts]...},
        strides {strides_in[RangeInts]...},
        inv_strides {(strides[RangeInts] ? strides[RangeInts] : IdxLin(1))...},
        inv_mods {(sizes[RangeInts] ? sizes[RangeInts] : IdxLin(1))...}
  {}

  /*!
   * Methods to performs bounds checking in layout objects
   */
  template<camp::idx_t N, typename Idx>
  RAJA_INLINE RAJA_HOST_DEVICE void BoundsCheckError(Idx idx) const
  {
    printf("Error at index %d, value %ld is not within bounds [0, %ld] \n",
           static_cast<int>(N), static_cast<long int>(idx),
           static_cast<long int>(sizes[N] - 1));
    RAJA_ABORT_OR_THROW("Out of bounds error \n");
  }

  template<camp::idx_t N>
  RAJA_INLINE RAJA_HOST_DEVICE void BoundsCheck() const
  {}

  template<camp::idx_t N, typename Idx, typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE void BoundsCheck(Idx idx,
                                                Indices... indices) const
  {
    if (sizes[N] > 0 && !(0 <= idx && idx < static_cast<Idx>(sizes[N])))
    {
      BoundsCheckError<N>(idx);
    }
    RAJA_UNUSED_VAR(idx);
    BoundsCheck<N + 1>(indices...);
  }

  /*!
   * Computes a linear space index from specified indices.
   * This is formed by the dot product of the indices and the layout strides.
   *
   * @param indices  Indices in the n-dimensional space of this layout
   * @return Linear space index.
   */

  template<typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE RAJA_BOUNDS_CHECK_constexpr IdxLin
  operator()(Indices... indices) const
  {
#if defined(RAJA_BOUNDS_CHECK_INTERNAL)
    BoundsCheck<0>(indices...);
#endif
    // dot product of strides and indices
    return sum<IdxLin>((RangeInts == stride_one_dim
                            ?  // Is this dimension stride-one?
                            indices
                            :  // it's stride one, so dont bother with multiply
                            strides[RangeInts] * indices  // it's not stride one
                        )...);
  }

  /*!
   * Given a linear-space index, compute the n-dimensional indices defined
   * by this layout.
   *
   * Note that this operation requires 2n integer divide instructions
   *
   * @param linear_index  Linear space index to be converted to indices.
   * @param indices  Variadic list of indices to be assigned, number must match
   *                 dimensionality of this layout.
   */
  template<typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE void toIndices(IdxLin linear_index,
                                              Indices&&... indices) const
  {
#if defined(RAJA_BOUNDS_CHECK_INTERNAL)
    IdxLin totSize = size_noproj();
    if (totSize > 0 && (linear_index < 0 || linear_index >= totSize))
    {
      printf("Error! Linear index %ld is not within bounds [0, %ld]. \n",
             static_cast<long int>(linear_index),
             static_cast<long int>(totSize - 1));
      RAJA_ABORT_OR_THROW("Out of bounds error \n");
    }
#endif

    camp::sink((indices = (camp::decay<Indices>)((linear_index /
                                                  inv_strides[RangeInts]) %
                                                 inv_mods[RangeInts]))...);
  }

  /*!
   * Computes a size of the layout's space with projections as size 1.
   * This is the produce of each dimensions size or 1 if projected.
   *
   * @return Total size spanned by indices
   */
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin size() const
  {
    // Multiply together all of the sizes,
    // replacing 1 for any zero-sized dimensions
    return foldl(
        RAJA::operators::multiplies<IdxLin>(),
        (sizes[RangeInts] == IdxLin(0) ? IdxLin(1) : sizes[RangeInts])...);
  }

  /*!
   * Computes a total size of the layout's space.
   * This is the produce of each dimensions size.
   *
   * @return Total size spanned by indices
   */
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin size_noproj() const
  {
    // Multiply together all of the sizes
    return foldl(RAJA::operators::multiplies<IdxLin>(), sizes[RangeInts]...);
  }

  template<camp::idx_t DIM>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexLinear get_dim_stride() const
  {
    return strides[DIM];
  }

  template<camp::idx_t DIM>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexLinear get_dim_size() const
  {
    return sizes[DIM];
  }

  template<camp::idx_t DIM>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexLinear get_dim_begin() const
  {
    return 0;
  }
};

}  // namespace detail

/*!
 * @brief A mapping of n-dimensional index space to a linear index space.
 *
 * This is particularly useful for creating multi-dimensional arrays with the
 * RAJA::View class, and other tasks such as mapping logical i,j,k zone
 * indices to a flattened zone id.
 *
 * For example:
 *
 *     // Create a layout object
 *     Layout<3> layout(5,7,11);
 *
 *     // Map from 3d index space to linear
 *     int lin = layout(2,3,1);   // lin=198
 *
 *     // Map from linear space to 3d indices
 *     int i, j, k;
 *     layout.toIndices(lin, i, j, k); // i,j,k = {2, 3, 1}
 *
 *
 * The above example creates a 3-d layout object with dimension sizes 5, 7,
 * and 11.  So the total index space covers 5*7*11=385 unique indices.
 * The operator() provides a mapping from 3-d indices to linear, and the
 * toIndices provides the inverse.
 *
 * The default striding has the first index (left-most)as the longest stride,
 * and the last (right-most) index with stride-1.
 *
 * To achieve other striding, see the RAJA::make_permuted_layout function
 * which can a permutation to the default striding.
 *
 * Layout supports projections, 0 or more dimensions may be of size zero.
 * In this case, the linear index space is invariant for those dimensions,
 * and toIndicies(...) will always produce a zero for that dimensions index.
 *
 * An example of a "projected" Layout:
 *
 *     // Create a layout with a degenerate dimensions
 *     Layout<3> layout(3, 0, 5);
 *
 *     // The second (J) index is projected out
 *     int lin1 = layout(0, 10, 0);   // lin1 = 0
 *     int lin2 = layout(0, 5, 1);    // lin2 = 1
 *
 *     // The inverse mapping always produces a 0 for J
 *     int i,j,k;
 *     layout.toIndices(lin2, i, j, k); // i,j,k = {0, 0, 1}
 *
 */
template<size_t n_dims, typename IdxLin = Index_type, ptrdiff_t StrideOne = -1>
using Layout =
    detail::LayoutBase_impl<camp::make_idx_seq_t<n_dims>, IdxLin, StrideOne>;

template<typename IdxLin, typename DimTuple, ptrdiff_t StrideOne = -1>
struct TypedLayout;

template<typename IdxLin, typename... DimTypes, ptrdiff_t StrideOne>
struct TypedLayout<IdxLin, camp::tuple<DimTypes...>, StrideOne>
    : public Layout<sizeof...(DimTypes), strip_index_type_t<IdxLin>, StrideOne>
{

  using StrippedIdxLin = strip_index_type_t<IdxLin>;
  using Self   = TypedLayout<IdxLin, camp::tuple<DimTypes...>, StrideOne>;
  using Base   = Layout<sizeof...(DimTypes), StrippedIdxLin, StrideOne>;
  using DimArr = std::array<StrippedIdxLin, sizeof...(DimTypes)>;

  // Pull in base constructors
  using Base::Base;

  /*!
   * Computes a linear space index from specified indices.
   * This is formed by the dot product of the indices and the layout strides.
   *
   * @param indices  Indices in the n-dimensional space of this layout
   * @return Linear space index.
   */
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      DimTypes... indices) const
  {
    return IdxLin(Base::operator()(stripIndexType(indices)...));
  }

  /*!
   * Given a linear-space index, compute the n-dimensional indices defined
   * by this layout.
   *
   * Note that this operation requires 2n integer divide instructions
   *
   * @param linear_index  Linear space index to be converted to indices.
   * @param indices  Variadic list of indices to be assigned, number must match
   *                 dimensionality of this layout.
   */
  RAJA_INLINE RAJA_HOST_DEVICE void toIndices(IdxLin linear_index,
                                              DimTypes&... indices) const
  {
    toIndicesHelper(camp::make_idx_seq_t<sizeof...(DimTypes)> {},
                    std::forward<IdxLin>(linear_index),
                    std::forward<DimTypes&>(indices)...);
  }

private:
  /*!
   * @internal
   *
   * Helper that uses the non-typed toIndices() function, and converts the
   * result to typed indices
   *
   */
  template<typename... Indices, camp::idx_t... RangeInts>
  RAJA_INLINE RAJA_HOST_DEVICE void toIndicesHelper(camp::idx_seq<RangeInts...>,
                                                    IdxLin linear_index,
                                                    Indices&... indices) const
  {
    StrippedIdxLin locals[sizeof...(DimTypes)];
    Base::toIndices(stripIndexType(linear_index), locals[RangeInts]...);
    camp::sink(
        (indices = Indices {static_cast<Indices>(locals[RangeInts])})...);
  }
};

/*!
 * Convert a non-stride-one Layout to a stride-1 Layout
 *
 */
template<ptrdiff_t s1_dim, size_t n_dims, typename IdxLin>
RAJA_INLINE RAJA_HOST_DEVICE constexpr Layout<n_dims, IdxLin, s1_dim>
make_stride_one(Layout<n_dims, IdxLin> const& l)
{
  return Layout<n_dims, IdxLin, s1_dim>(l);
}

/*!
 * Convert a non-stride-one TypedLayout to a stride-1 TypedLayout
 *
 */
template<ptrdiff_t s1_dim, typename IdxLin, typename IdxTuple>
RAJA_INLINE RAJA_HOST_DEVICE constexpr TypedLayout<IdxLin, IdxTuple, s1_dim>
make_stride_one(TypedLayout<IdxLin, IdxTuple> const& l)
{
  // strip l to it's base-class type
  using Base    = typename TypedLayout<IdxLin, IdxTuple>::Base;
  Base const& b = (Base const&)l;

  // Use non-typed layout to initialize new typed layout
  return TypedLayout<IdxLin, IdxTuple, s1_dim>(b);
}


}  // namespace RAJA

#endif
