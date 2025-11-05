/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining the IndexLayout class and IndexList
 *classes.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_INDEXLAYOUT_HPP
#define RAJA_INDEXLAYOUT_HPP

#include "RAJA/util/Layout.hpp"

namespace RAJA
{

/*!
 * DirectIndex struct contains call operator that returns the same index that
 * was input
 *
 */
template<typename IdxLin = Index_type>
struct DirectIndex
{

  IdxLin RAJA_INLINE RAJA_HOST_DEVICE constexpr operator()(
      const IdxLin idx) const
  {
    return idx;
  }
};

/*!
 * IndexList struct stores a pointer to an array containing the index list.
 * Its call operator returns the entry at the input location (idx) of its index
 * list.
 *
 */
template<typename IdxLin = Index_type>
struct IndexList
{

  IdxLin* index_list {nullptr};

  IdxLin RAJA_INLINE RAJA_HOST_DEVICE constexpr operator()(
      const IdxLin idx) const
  {
    return index_list[idx];
  }
};

/*!
 * ConditionalIndexList struct stores a pointer to an array containing the index
 * list. Its call operator returns the same index that was input if the index
 * list is a nullptr, or otherwise returns the entry at the input location (idx)
 * of its index list.
 *
 */
template<typename IdxLin = Index_type>
struct ConditionalIndexList
{

  IdxLin* index_list {nullptr};

  IdxLin RAJA_INLINE RAJA_HOST_DEVICE constexpr operator()(
      const IdxLin idx) const
  {
    if (index_list)
    {
      return index_list[idx];
    }
    else
    {
      return idx;
    }
  }
};

namespace internal
{

template<typename Range, typename IdxLin, typename... IndexTypes>
struct IndexLayout_impl;

template<camp::idx_t... RangeInts, typename IdxLin, typename... IndexTypes>
struct IndexLayout_impl<camp::idx_seq<RangeInts...>, IdxLin, IndexTypes...>
{
  using IndexRange  = camp::idx_seq<RangeInts...>;
  using IndexLinear = IdxLin;
  using Base        = RAJA::detail::LayoutBase_impl<IndexRange, IdxLin>;
  Base base_;

  static constexpr size_t n_dims = sizeof...(RangeInts);

  camp::tuple<IndexTypes...> tuple;

  template<typename... Types>
  constexpr RAJA_INLINE IndexLayout_impl(
      camp::tuple<IndexTypes...> index_tuple_in,
      Types... ns)
      : base_ {(ns)...},
        tuple(index_tuple_in)
  {}

  /*!
   * Computes a linear space index from entries of index lists stored in tuple.
   * This is accomplished through the inner product of the strides and the
   * entry in the index list along each dimension.
   * @param indices Indices in the n-dimensional space of this layout
   * @return Linear space index.
   */
  template<typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      Indices... indices) const
  {
    return sum<IdxLin>(
        (base_.strides[RangeInts] * camp::get<RangeInts>(tuple)(indices))...);
  }

  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexLinear size() const
  {
    return base_.size();
  }

  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexLinear size_noproj() const
  {
    return base_.size_noproj();
  }

  template<camp::idx_t DIM>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexLinear get_dim_stride() const
  {
    return base_.template get_dim_stride<DIM>();
  }

  template<camp::idx_t DIM>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexLinear get_dim_size() const
  {
    return base_.template get_dim_size<DIM>();
  }
};

}  // namespace internal

template<size_t n_dims   = 1,
         typename IdxLin = Index_type,
         typename... IndexTypes>
struct IndexLayout
    : public internal::
          IndexLayout_impl<camp::make_idx_seq_t<n_dims>, IdxLin, IndexTypes...>
{
  using Base = internal::
      IndexLayout_impl<camp::make_idx_seq_t<n_dims>, IdxLin, IndexTypes...>;

  using internal::IndexLayout_impl<camp::make_idx_seq_t<n_dims>,
                                   IdxLin,
                                   IndexTypes...>::IndexLayout_impl;

  constexpr RAJA_INLINE RAJA_HOST_DEVICE
  IndexLayout(const internal::IndexLayout_impl<camp::make_idx_seq_t<n_dims>,
                                               IdxLin,
                                               IndexTypes...>& rhs)
      : Base {rhs}
  {}
};

/*!
 * creates of a camp::tuple of index types
 * (such as DirectIndex, IndexList, or ConditionalIndexList)
 *
 */
template<typename... IndexTypes>
RAJA_INLINE RAJA_HOST_DEVICE auto make_index_tuple(IndexTypes... it)
    -> camp::tuple<IndexTypes...>
{
  return camp::tuple<IndexTypes...>(it...);
}

/*!
 * creates an index layout based on the input camp::tuple of index types
 *
 */
template<typename IdxLin = Index_type,
         typename... Types,
         typename... IndexTypes>
RAJA_INLINE RAJA_HOST_DEVICE auto make_index_layout(
    camp::tuple<IndexTypes...> index_tuple_in,
    Types... ns) -> IndexLayout<sizeof...(Types), IdxLin, IndexTypes...>
{
  static_assert(sizeof...(Types) == sizeof...(IndexTypes), "");
  return IndexLayout<sizeof...(Types), IdxLin, IndexTypes...>(index_tuple_in,
                                                              ns...);
}

}  // namespace RAJA

#endif
