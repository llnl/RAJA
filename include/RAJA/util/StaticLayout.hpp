/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining Layout, a N-dimensional index calculator
 *          with compile-time defined sizes and permutation
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_util_static_layout_HPP
#define RAJA_util_static_layout_HPP

#include "RAJA/config.hpp"

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


template<typename IdxLin,
         typename Range,
         typename Sizes,
         typename Strides,
         typename DimTypeList = void>
struct StaticLayoutBase_impl;

template<typename IdxLin,
         IdxLin... RangeInts,
         IdxLin... Sizes,
         IdxLin... Strides>
struct StaticLayoutBase_impl<IdxLin,
                             camp::int_seq<IdxLin, RangeInts...>,
                             camp::int_seq<IdxLin, Sizes...>,
                             camp::int_seq<IdxLin, Strides...>,
                             void>
{

  using IndexLinear = IdxLin;
  using sizes       = camp::int_seq<IdxLin, Sizes...>;
  using strides     = camp::int_seq<IdxLin, Strides...>;

  static constexpr camp::idx_t stride_one_dim = RAJA::max<camp::idx_t>(
      (camp::seq_at<RangeInts, strides>::value == 1 ? camp::idx_t(RangeInts)
                                                    : -1)...);

  static constexpr size_t n_dims = sizeof...(Sizes);

  /*!
   * Default constructor.
   */
  RAJA_INLINE RAJA_HOST_DEVICE constexpr StaticLayoutBase_impl() {}

  RAJA_INLINE static void print()
  {
    camp::sink(printf("StaticLayout: arg%d: size=%d, stride=%d\n",
                      (int)RangeInts, (int)Sizes, (int)Strides)...);
  }

  /*!
   * Computes a linear space index from specified indices.
   * This is formed by the dot product of the indices and the layout strides.
   *
   * @param indices  Indices in the n-dimensional space of this layout
   * @return Linear space index.
   */
  template<typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      Indices... indices) const
  {
    // dot product of strides and indices
    return RAJA::sum<IdxLin>((IdxLin(indices * Strides))...);
  }

  template<typename... Indices>
  static RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin s_oper(
      Indices... indices)
  {
    // dot product of strides and indices
    return RAJA::sum<IdxLin>((IdxLin(indices * Strides))...);
  }

  // Multiply together all of the sizes,
  // replacing 1 for any zero-sized dimensions
  static constexpr IdxLin s_size =
      RAJA::product<IdxLin>((Sizes == IdxLin(0) ? IdxLin(1) : Sizes)...);

  // Multiply together all of the sizes
  static constexpr IdxLin s_size_noproj = RAJA::product<IdxLin>(Sizes...);

  /*!
   * Computes a size of the layout's space with projections as size 1.
   * This is the produce of each dimensions size or 1 if projected.
   *
   * @return Total size spanned by indices
   */
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr IdxLin size()
  {
    // Multiply together all of the sizes,
    // replacing 1 for any zero-sized dimensions
    return s_size;
  }

  /*!
   * Computes a total size of the layout's space.
   * This is the produce of each dimensions size.
   *
   * @return Total size spanned by indices
   */
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr IdxLin size_noproj()
  {
    // Multiply together all of the sizes
    return s_size_noproj;
  }

  template<camp::idx_t DIM>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexLinear get_dim_stride() const
  {
    return camp::seq_at<DIM, strides>::value;
  }

  template<camp::idx_t DIM>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexLinear get_dim_size() const
  {
    return camp::seq_at<DIM, sizes>::value;
  }

  template<camp::idx_t DIM>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexLinear get_dim_begin() const
  {
    return 0;
  }
};

template<typename IdxLin, IdxLin N, IdxLin Idx, IdxLin... Sizes>
struct StrideCalculatorIdx
{
  static_assert(N == sizeof...(Sizes), "");

  using sizes_seq              = camp::int_seq<IdxLin, Sizes...>;
  static constexpr IdxLin size = camp::seq_at<Idx, sizes_seq>::value;
  static constexpr IdxLin size_last =
      StrideCalculatorIdx<IdxLin, N, Idx + 1, Sizes...>::size;
  static constexpr IdxLin value =
      (size_last > 0 ? size_last : 1) *
      StrideCalculatorIdx<IdxLin, N, Idx + 1, Sizes...>::value;
  static constexpr IdxLin stride = size > 0 ? value : 0;
};

template<typename IdxLin, IdxLin N, IdxLin... Sizes>
struct StrideCalculatorIdx<IdxLin, N, N, Sizes...>
{
  static_assert(N == sizeof...(Sizes), "");

  static constexpr IdxLin size   = 1;
  static constexpr IdxLin value  = 1;
  static constexpr IdxLin stride = size > 0 ? value : 0;
};

template<typename IdxLin, typename Range, typename Perm, typename Sizes>
struct StrideCalculator;

template<typename IdxLin, IdxLin... Range, camp::idx_t... Perm, IdxLin... Sizes>
struct StrideCalculator<IdxLin,
                        camp::int_seq<IdxLin, Range...>,
                        camp::idx_seq<Perm...>,
                        camp::int_seq<IdxLin, Sizes...>>
{
  static_assert(sizeof...(Sizes) == sizeof...(Perm), "");

  using sizes               = camp::int_seq<IdxLin, Sizes...>;
  static constexpr IdxLin N = sizeof...(Sizes);
  using range               = camp::int_seq<IdxLin, Range...>;
  using perm                = camp::idx_seq<Perm...>;
  using inv_perm            = invert_permutation<perm>;

  using strides_unperm = camp::int_seq<
      IdxLin,
      StrideCalculatorIdx<IdxLin,
                          N,
                          Range,
                          camp::seq_at<Perm, sizes>::value...>::stride...>;

  using strides =
      camp::int_seq<IdxLin,
                    camp::seq_at<camp::seq_at<Range, inv_perm>::value,
                                 strides_unperm>::value...>;
};

template<typename IdxLin,
         IdxLin... RangeInts,
         IdxLin... Sizes,
         IdxLin... Strides,
         typename... DimTypes>
struct StaticLayoutBase_impl<IdxLin,
                             camp::int_seq<IdxLin, RangeInts...>,
                             camp::int_seq<IdxLin, Sizes...>,
                             camp::int_seq<IdxLin, Strides...>,
                             camp::list<DimTypes...>>
{


  using IndexLinear = IdxLin;
  using ranges      = camp::int_seq<IdxLin, RangeInts...>;
  using sizes       = camp::int_seq<IdxLin, Sizes...>;
  using strides     = camp::int_seq<IdxLin, Strides...>;

  using InnerLayout =
      StaticLayoutBase_impl<IdxLin, ranges, sizes, strides, void>;

  static constexpr camp::idx_t stride_one_dim = InnerLayout::stride_one_dim;

  static constexpr IndexLinear n_dims = sizeof...(DimTypes);

  /*!
   * Computes a linear space index from specified indices.
   * This is formed by the dot product of the indices and the layout strides.
   *
   * @param indices  Indices in the n-dimensional space of this layout
   * @return Linear space index.
   */
  static RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexLinear s_oper(
      DimTypes... indices)
  {
    return InnerLayout::s_oper(stripIndexType(indices)...);
  }

  static constexpr IndexLinear s_size        = InnerLayout::s_size;
  static constexpr IndexLinear s_size_noproj = InnerLayout::s_size_noproj;

  RAJA_INLINE RAJA_HOST_DEVICE constexpr static IndexLinear size()
  {
    return s_size;
  }

  RAJA_INLINE RAJA_HOST_DEVICE constexpr static IndexLinear size_noproj()
  {
    return s_size_noproj;
  }

  template<camp::idx_t DIM>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexLinear get_dim_stride() const
  {
    return InnerLayout {}.get_dim_stride();
  }

  template<camp::idx_t DIM>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexLinear get_dim_size() const
  {
    return camp::seq_at<DIM, sizes>::value;
  }

  template<camp::idx_t DIM>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexLinear get_dim_begin() const
  {
    return 0;
  }

  RAJA_INLINE
  static void print() { InnerLayout::print(); }
};

template<typename Perm,
         typename IdxLin,
         typename Sizes,
         typename Indexes,
         typename TypeList>
struct StaticLayoutMaker
{
  using strides =
      typename detail::StrideCalculator<IdxLin, Indexes, Perm, Sizes>::strides;
  using type = StaticLayoutBase_impl<IdxLin, Indexes, Sizes, strides, TypeList>;
};


}  // namespace detail

template<typename Perm, typename IdxLin, camp::idx_t... Sizes>
using StaticLayoutT = typename detail::StaticLayoutMaker<
    Perm,
    IdxLin,
    camp::int_seq<IdxLin, Sizes...>,
    camp::make_int_seq_t<IdxLin, sizeof...(Sizes)>,
    void>::type;

template<typename Perm, camp::idx_t... Sizes>
using StaticLayout = StaticLayoutT<Perm, camp::idx_t, Sizes...>;

template<typename Perm,
         typename IdxLin,
         typename TypeList,
         camp::idx_t... Sizes>
using TypedStaticLayout = typename detail::StaticLayoutMaker<
    Perm,
    IdxLin,
    camp::int_seq<IdxLin, Sizes...>,
    camp::make_int_seq_t<IdxLin, sizeof...(Sizes)>,
    TypeList>::type;

}  // namespace RAJA

#endif
