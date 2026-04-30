/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for flattened N-D forall helpers built on
 *dynamic_forall.
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

#ifndef RAJA_pattern_forall_nd_HPP
#define RAJA_pattern_forall_nd_HPP

#include <utility>

#include "RAJA/pattern/forall.hpp"
#include "RAJA/util/CombiningAdapter.hpp"
#include "RAJA/util/View.hpp"

namespace RAJA
{

template<typename... IdxTs>
struct TypedRangeSegmentPack
{
  camp::tuple<RAJA::TypedRangeSegment<IdxTs>...> data;
};

template<typename... IdxTs>
RAJA_INLINE auto segments(RAJA::TypedRangeSegment<IdxTs> const&... segs)
{
  return TypedRangeSegmentPack<IdxTs...> {camp::make_tuple(segs...)};
}

namespace detail
{

template<typename Seq>
struct reverse_idx_seq;

template<camp::idx_t... Idx>
struct reverse_idx_seq<camp::idx_seq<Idx...>>
{
  using type = camp::idx_seq<sizeof...(Idx) - 1U - Idx...>;
};

template<typename LayoutTag, camp::idx_t Rank>
struct layout_to_permutation;

template<camp::idx_t Rank>
struct layout_to_permutation<RAJA::layout_right, Rank>
{
  using type = camp::make_idx_seq_t<Rank>;
};

template<camp::idx_t Rank>
struct layout_to_permutation<RAJA::layout_left, Rank>
{
  using type = typename reverse_idx_seq<camp::make_idx_seq_t<Rank>>::type;
};

template<typename LayoutTag, typename Lambda, typename... IdxTs>
RAJA_INLINE auto make_forall_nd_adapter(
    Lambda&& body,
    TypedRangeSegmentPack<IdxTs...> const& segs)
{
  using perm =
      typename layout_to_permutation<LayoutTag, sizeof...(IdxTs)>::type;

  return camp::apply(
      [&](auto const&... unpacked_segs) {
        return RAJA::make_PermutedCombiningAdapter<perm>(
            std::forward<Lambda>(body), unpacked_segs...);
      },
      segs.data);
}

}  // namespace detail

template<typename PolicyList,
         typename LayoutTag = RAJA::layout_right,
         typename... IdxTs,
         typename Lambda>
void forall_nd(int pol,
               TypedRangeSegmentPack<IdxTs...> const& segs,
               Lambda&& body)
{
  if (pol < 0)
  {
    RAJA_ABORT_OR_THROW("Policy value out of range");
  }

  auto adapter = detail::make_forall_nd_adapter<LayoutTag>(
      std::forward<Lambda>(body), segs);

  RAJA::dynamic_forall<PolicyList>(pol, adapter.getRange(), adapter);
}

template<typename PolicyList,
         typename LayoutTag = RAJA::layout_right,
         typename... IdxTs,
         typename Lambda>
resources::EventProxy<resources::Resource> forall_nd(
    RAJA::resources::Resource resource,
    int pol,
    TypedRangeSegmentPack<IdxTs...> const& segs,
    Lambda&& body)
{
  if (pol < 0)
  {
    RAJA_ABORT_OR_THROW("Policy value out of range");
  }

  auto adapter = detail::make_forall_nd_adapter<LayoutTag>(
      std::forward<Lambda>(body), segs);

  return RAJA::dynamic_forall<PolicyList>(resource, pol, adapter.getRange(),
                                          adapter);
}

}  // namespace RAJA

#endif /* RAJA_pattern_forall_nd_HPP */
