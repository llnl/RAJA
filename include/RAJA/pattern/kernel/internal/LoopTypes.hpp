/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for loop kernel internals and related helper functions.
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

#ifndef RAJA_pattern_kernel_internal_LoopTypes_HPP
#define RAJA_pattern_kernel_internal_LoopTypes_HPP

#include "RAJA/config.hpp"
#include "RAJA/pattern/kernel/internal/Template.hpp"
#include "camp/camp.hpp"

namespace RAJA
{
namespace internal
{


template<typename SegmentTypes, typename OffsetTypes>
struct LoopTypes;

template<typename... SegmentTypes, typename... OffsetTypes>
struct LoopTypes<camp::list<SegmentTypes...>, camp::list<OffsetTypes...>>
{

  using Self =
      LoopTypes<camp::list<SegmentTypes...>, camp::list<OffsetTypes...>>;

  static constexpr size_t s_num_segments = sizeof...(SegmentTypes);

  // This ensures that you don't double-loop over a segment within the same
  // loop nesting
  static_assert(s_num_segments == sizeof...(OffsetTypes),
                "Number of segments and offsets must match");

  using segment_types_t = camp::list<SegmentTypes...>;
  using offset_types_t  = camp::list<OffsetTypes...>;
};

template<typename Data>
using makeInitialLoopTypes = LoopTypes<
    list_of_n<void, camp::tuple_size<typename Data::segment_tuple_t>::value>,
    list_of_n<void, camp::tuple_size<typename Data::segment_tuple_t>::value>>;


template<typename Types, camp::idx_t Segment, typename T, typename Seq>
struct SetSegmentTypeHelper;

template<typename Types, camp::idx_t Segment, typename T, camp::idx_t... SEQ>
struct SetSegmentTypeHelper<Types, Segment, T, camp::idx_seq<SEQ...>>
{
  using segment_list = typename Types::segment_types_t;
  using offset_list  = typename Types::offset_types_t;

  static_assert(std::is_same<camp::at_v<segment_list, Segment>, void>::value,
                "Segment was already assigned: Probably looping over same "
                "segment in loop nest");

  using type = LoopTypes<
      camp::list<
          typename std::conditional<SEQ == Segment,
                                    T,
                                    camp::at_v<segment_list, SEQ>>::type...>,
      camp::list<
          typename std::conditional<SEQ == Segment,
                                    T,
                                    camp::at_v<segment_list, SEQ>>::type...>>;
};

template<typename Types, camp::idx_t Segment, typename T>
using setSegmentType = typename SetSegmentTypeHelper<
    Types,
    Segment,
    T,
    camp::make_idx_seq_t<Types::s_num_segments>>::type;

template<typename Types, camp::idx_t Segment, typename Data>
using setSegmentTypeFromData = setSegmentType<
    Types,
    Segment,
    camp::at_v<typename camp::decay<Data>::index_types_t, Segment>>;


}  // end namespace internal
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_internal_LoopTypes_HPP */
