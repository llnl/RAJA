/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA CombingAdapter.
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

#ifndef RAJA_CombingAdapter_HPP
#define RAJA_CombingAdapter_HPP

#include <type_traits>

#include "RAJA/index/RangeSegment.hpp"
#include "RAJA/util/camp_aliases.hpp"
#include "RAJA/util/concepts.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/Layout.hpp"
#include "RAJA/util/OffsetLayout.hpp"

namespace RAJA
{

/*!
 * @brief A holder for adapting lambdas meant for multidimensional index spaces
 *        to allow their use in 1-dimensional index spaces.
 *
 * Creates callable object that when called with a 1-dimensional index, converts
 * that index to a multi-dimensional index and calls the given lambda.
 * This allows lambdas meant for use in multi-dimensional loop abstractions to
 * be used in 1-dimensional loop abstractions.
 * The 1-dimensional index is of the type of the IndexType template parameter.
 *
 * For example:
 *
 *     // lambda for use in 3-d loop
 *     auto lambda = [=](int i, int j, int k) {...};
 *
 *     // 3D loop bounds
 *     int isize;
 *     int jsize;
 *     int ksize;

 *     // example lambda usage
 *     for (int i = 0; i < isize; ++i) {
 *       for (int j = 0; j < jsize; ++j) {
 *         for (int k = 0; k < ksize; ++k) {
 *           lambda(i, j, k);
 *         }
 *       }
 *     }
 *
 *     using Layout = Layout<3, int, // 3 dimensions, index type is int
 *                           2>;    // optimization, stride 1 dim is 2
 *
 *     // Create a CombiningAdapter object for lambda and the ranges
 *     // NOTE Use make_CombiningAdapter
 *     RAJA::CombiningAdapter<decltype(lambda), Layout>
 *         adapter(lambda, Layout(isize, jsize, ksize));
 *
 *     // Use with RAJA forall
 *     RAJA::forall<policy>(adapter.getRange(), adapter);
 *
 *     // Use with c++-style loop
 *     auto range = adapter.getRange();
 *     for (auto it = begin(range); it < end(range); ++it) {
 *       adapter(*it)
 *     }
 *
 */
template<typename Lambda, typename Layout_>
struct CombiningAdapter
{
  using Layout = Layout_;

  using IndexRange     = typename Layout::IndexRange;
  using StrippedIdxLin = typename Layout::StrippedIdxLin;
  using IndexLinear    = typename Layout::IndexLinear;
  using DimTuple       = typename Layout::DimTuple;
  using DimArr         = typename Layout::DimArr;

  using RangeLinear = RAJA::TypedRangeSegment<IndexLinear>;

private:
  Lambda m_lambda;
  Layout m_layout;

  RAJA_SUPPRESS_HD_WARN
  template<camp::idx_t... RangeInts>
  RAJA_HOST_DEVICE inline auto call_helper(IndexLinear linear_index,
                                           camp::idx_seq<RangeInts...>)
      -> decltype(m_lambda(
          camp::val<camp::tuple_element_t<RangeInts, DimTuple>>()...))
  {
    DimTuple indices;
    m_layout.toIndices(linear_index, camp::get<RangeInts>(indices)...);
    return m_lambda(camp::get<RangeInts>(indices)...);
  }

  ///
  RAJA_SUPPRESS_HD_WARN
  template<camp::idx_t... RangeInts>
  RAJA_HOST_DEVICE inline auto call_helper(IndexLinear linear_index,
                                           camp::idx_seq<RangeInts...>) const
      -> decltype(m_lambda(
          camp::val<camp::tuple_element_t<RangeInts, DimTuple>>()...))
  {
    DimTuple indices;
    m_layout.toIndices(linear_index, camp::get<RangeInts>(indices)...);
    return m_lambda(camp::get<RangeInts>(indices)...);
  }

public:
  /*!
   * Constructor from lambda and layout.
   */
  template<typename C_Lambda, typename C_Layout>
  RAJA_HOST_DEVICE CombiningAdapter(C_Lambda&& lambda, C_Layout&& layout)
      : m_lambda(std::forward<C_Lambda>(lambda)),
        m_layout(std::forward<C_Layout>(layout))
  {}

  /*!
   * Call the lambda by converting the linear index to multidimensional indices.
   *
   * @return return value of lambda
   */
  RAJA_HOST_DEVICE RAJA_INLINE auto operator()(IndexLinear linear_index)
      -> decltype(call_helper(linear_index, IndexRange()))
  {
    return call_helper(linear_index, IndexRange());
  }

  ///
  RAJA_HOST_DEVICE RAJA_INLINE auto operator()(IndexLinear linear_index) const
      -> decltype(call_helper(linear_index, IndexRange()))
  {
    return call_helper(linear_index, IndexRange());
  }

  /*!
   * Computes the total size of the layout's space.
   *
   * @return Total size of layout
   */
  RAJA_HOST_DEVICE RAJA_INLINE IndexLinear size() const
  {
    return m_layout.size_noproj();
  }

  /*!
   * Convenience method to get a 1-dimensional range representing the
   * total size of the layout.
   *
   * @return Range representing the total size of the layout
   */
  RAJA_HOST_DEVICE RAJA_INLINE RangeLinear getRange() const
  {
    return RangeLinear(static_cast<IndexLinear>(0), size());
  }
};

/*!
 * @brief Creates a CombiningAdapter class from a lambda and segments.
 * @param lambda functional object
 * @param segs range segments defining the multi-dimensional index space
 * @return Returns a CombiningAdapter for the given lambda and segments
 *
 * Creates a CombiningAdapter object given a lambda and range segments.
 *
 * NOTE: the stride 1 index is the right-most index
 *
 * For example:
 *
 *     // lambda to be used in 2D loop
 *     auto lambda = [](int i, int j) {...};
 *
 *     // example lambda usage
 *     for (int i = ibegin; i < iend; ++i) {
 *       for (int j = jbegin; j < jend; ++j) {
 *         lambda(i, j);
 *       }
 *     }
 *
 *     // ranges for 2D loop bounds
 *     RAJA::TypedRangeSegment<int> irange(ibegin, iend);
 *     RAJA::TypedRangeSegment<int> jrange(jbegin, jend);
 *
 *     // adapter class
 *     auto adapter = RAJA::make_CombiningAdapter(lambda, irange, jrange);
 *
 *     // Use with RAJA forall
 *     RAJA::forall<policy>(adapter.getRange(), adapter);
 *
 *     // Use with c++-style loop
 *     auto range = adapter.getRange();
 *     for (auto it = begin(range); it < end(range); ++it) {
 *       adapter(*it)
 *     }
 *
 */
template<typename Lambda, typename Layout>
RAJA_HOST_DEVICE RAJA_INLINE auto make_CombiningAdapter_from_layout(
    Lambda&& lambda,
    Layout&& layout)
// -> CombiningAdapter<camp::decay<Lambda>, camp::decay<Layout>>
{
  return CombiningAdapter<camp::decay<Lambda>, camp::decay<Layout>>(
      std::forward<Lambda>(lambda), std::forward<Layout>(layout));
}
///
RAJA_SUPPRESS_HD_WARN
template<typename Lambda, typename... IdxTs>
RAJA_INLINE auto make_CombiningAdapter(
    Lambda&& lambda,
    ::RAJA::TypedRangeSegment<IdxTs> const&... segs)
// -> decltype(make_CombiningAdapter_from_layout(std::forward<Lambda>(lambda),
//             camp::val<RAJA::TypedOffsetLayout<
//                 typename std::common_type< strip_index_type_t<IdxTs>...
//                 >::type, IdxTs...>>()))
{
  using std::begin;
  using std::distance;
  using std::end;
  using IdxLin = typename std::common_type<strip_index_type_t<IdxTs>...>::type;
  using Layout = RAJA::Layout<sizeof...(IdxTs), IdxLin>;
  using OffsetLayout = RAJA::TypedOffsetLayout<IdxLin, camp::tuple<IdxTs...>>;

  Layout layout(static_cast<IdxLin>(distance(begin(segs), end(segs)))...);
  OffsetLayout offset_layout = OffsetLayout::from_layout_and_offsets(
      {{(distance(begin(segs), end(segs)) ? static_cast<IdxLin>(*begin(segs))
                                          : static_cast<IdxLin>(0))...}},
      std::move(layout));
  return make_CombiningAdapter_from_layout(std::forward<Lambda>(lambda),
                                           std::move(offset_layout));
}
///
RAJA_SUPPRESS_HD_WARN
template<typename Perm, typename Lambda, typename... IdxTs>
RAJA_INLINE auto make_PermutedCombiningAdapter(
    Lambda&& lambda,
    ::RAJA::TypedRangeSegment<IdxTs> const&... segs)
// -> decltype(make_CombiningAdapter_from_layout(std::forward<Lambda>(lambda),
//             camp::val<RAJA::TypedOffsetLayout<
//                 typename std::common_type< strip_index_type_t<IdxTs>...
//                 >::type, IdxTs...>>()))
{
  using std::begin;
  using std::distance;
  using std::end;
  using IdxLin = typename std::common_type<strip_index_type_t<IdxTs>...>::type;
  using OffsetLayout = RAJA::TypedOffsetLayout<IdxLin, camp::tuple<IdxTs...>>;

  auto layout = make_permuted_layout<sizeof...(IdxTs), IdxLin>(
      {{static_cast<IdxLin>(distance(begin(segs), end(segs)))...}},
      RAJA::as_array<Perm>::get());
  OffsetLayout offset_layout = OffsetLayout::from_layout_and_offsets(
      {{(distance(begin(segs), end(segs)) ? static_cast<IdxLin>(*begin(segs))
                                          : static_cast<IdxLin>(0))...}},

      std::move(layout));
  return make_CombiningAdapter_from_layout(std::forward<Lambda>(lambda),
                                           std::move(offset_layout));
}

}  // end namespace RAJA

#endif /* RAJA_CombingAdapter_HPP */
