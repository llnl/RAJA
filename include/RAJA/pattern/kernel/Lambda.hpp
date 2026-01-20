/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for kernel lambda executor.
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

#ifndef RAJA_pattern_kernel_Lambda_HPP
#define RAJA_pattern_kernel_Lambda_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel/internal.hpp"
#include "RAJA/pattern/params/reducer.hpp"

namespace RAJA
{

namespace internal
{
struct lambda_arg_seg_t
{};

struct lambda_arg_param_t
{};

struct lambda_arg_offset_t
{};

template<typename T>
struct lambda_arg_value_t
{
  using type = T;
};

template<typename T, camp::idx_t V>
struct LambdaArg
{
  static constexpr camp::idx_t value = V;
};

}  // namespace internal

/*!
 * Used in RAJA::statement::Lambda to specify that one or more segment values
 * should be passed into the lambda as an argument
 */
template<camp::idx_t... args>
using Segs =
    camp::list<internal::LambdaArg<internal::lambda_arg_seg_t, args>...>;

/*!
 * Used in RAJA::statement::Lambda to specify that one or more segment offsets
 * should be passed into the lambda as an argument
 *
 * The offset is the distance from the segment begin() that is currently being
 * iterated on.
 *
 * In the case of tiling (with Tile) the offset is w.r.t. the beginning of the
 * current tile.
 */
template<camp::idx_t... args>
using Offsets =
    camp::list<internal::LambdaArg<internal::lambda_arg_offset_t, args>...>;


/*!
 * Used in RAJA::statement::Lambda to specify that one or more parameters that
 * should be passed into the lambda as an argument.
 */
template<camp::idx_t... args>
using Params =
    camp::list<internal::LambdaArg<internal::lambda_arg_param_t, args>...>;

/*!
 * Used in RAJA::statement::Lambda to specify that one or more constant values
 * should be passed into the lambda as an argument.
 *
 * Values specified in the template parameters
 *
 * Example:
 * writing:   Lambda<0, ValuesT<int, 3>>
 * invokes:   lambda0( (int)3 )
 *
 * writing:   Lambda<0, ValuesT<double, 3, 4>>
 * invokes:   lambda0( (double)3, (double) 4 )
 */
template<typename T, camp::idx_t... values>
using ValuesT =
    camp::list<internal::LambdaArg<internal::lambda_arg_value_t<T>, values>...>;

namespace statement
{
/*!
 * A RAJA::kernel statement that invokes a lambda function.
 *
 * The lambda is specified by its index in the sequence of lambda arguments
 * to a RAJA::kernel method.
 *
 * for example:
 * RAJA::kernel<exec_pol>(make_tuple{s0, s1, s2}, lambda0, lambda1);
 *
 */
template<camp::idx_t BodyIdx, typename... Args>
struct Lambda : internal::Statement<camp::nil>
{
  static const camp::idx_t loop_body_index = BodyIdx;
};

}  // end namespace statement

namespace internal
{


/*
 * Helper that extracts a segment value for a lambda argument
 *
 * By default we use the Span and offset in LoopData to construct the
 * value.
 *
 * This class allows specialization on the segment type in LoopTypes so that
 * fancier constructions can happen (ie vector_exec, etc.)
 */
template<typename SegmentType, camp::idx_t id>
struct LambdaSegExtractor
{

  static_assert(
      !std::is_same<SegmentType, void>::value,
      "Segment not assigned, but used in Lambda with Segs<> argument");

  template<typename Data>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr static SegmentType extract(Data&& data)
  {
    return SegmentType(camp::get<id>(data.segment_tuple)
                           .begin()[camp::get<id>(data.offset_tuple)]);
  }
};

/*
 * Helper that extracts a segment value for a lambda argument
 *
 * By default we use the Span and offset in LoopData to construct the
 * value.
 *
 * This class allows specialization on the segment type in LoopTypes so that
 * fancier constructions can happen (ie vector_exec, etc.)
 */
template<typename OffsetType, camp::idx_t id>
struct LambdaOffsetExtractor
{

  static_assert(
      !std::is_same<OffsetType, void>::value,
      "Segment not assigned, but used in Lambda with Offsets<> argument");

  template<typename Data>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr static OffsetType extract(Data&& data)
  {
    return OffsetType(camp::get<id>(data.offset_tuple));
  }
};


/*
 * Helper that provides first level of argument extraction
 * This acts as a switchboard between Segs, Offsets, and Params
 *
 * It calls LambdaArgExtractor to perform the actual argument extraction.
 * This allows LambdaArgExtractor to be specialized
 */
template<typename Types, typename T>
struct LambdaArgSwitchboard;

template<typename Types, camp::idx_t id>
struct LambdaArgSwitchboard<Types, LambdaArg<lambda_arg_offset_t, id>>
{

  using OffsetType = camp::at_v<typename Types::offset_types_t, id>;

  static_assert(
      !std::is_same<OffsetType, void>::value,
      "Offset not assigned, but used in Lambda with Offsets<> argument");

  template<typename Data>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr static OffsetType extract(Data&& data)
  {
    return LambdaOffsetExtractor<OffsetType, id>::extract(
        std::forward<Data>(data));
  }
};

template<typename Types, camp::idx_t id>
struct LambdaArgSwitchboard<Types, LambdaArg<lambda_arg_seg_t, id>>
{

  using SegmentType = camp::at_v<typename Types::segment_types_t, id>;

  static_assert(
      !std::is_same<SegmentType, void>::value,
      "Segment not assigned, but used in Lambda with Segs<> argument");

  template<typename Data>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr static SegmentType extract(Data&& data)
  {
    return LambdaSegExtractor<SegmentType, id>::extract(
        std::forward<Data>(data));
  }
};

template<typename Types, camp::idx_t id>
struct LambdaArgSwitchboard<Types, LambdaArg<lambda_arg_param_t, id>>
{
  template<typename Data>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr static auto extract(Data&& data)
      -> std::add_lvalue_reference_t<
          camp::tuple_element_t<id, typename camp::decay<Data>::arg_tuple_t>>
  {
    return RAJA::expt::detail::get_lambda_arg(camp::get<id>(data.param_tuple));
  }
};

template<typename Types, typename T, camp::idx_t value>
struct LambdaArgSwitchboard<Types, LambdaArg<lambda_arg_value_t<T>, value>>
{
  template<typename Data>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr static T extract(Data&&)
  {
    return T(value);
  }
};

RAJA_SUPPRESS_HD_WARN
template<camp::idx_t LoopIndex,
         typename Types,
         typename Data,
         typename... targLists>
RAJA_INLINE RAJA_HOST_DEVICE void invoke_lambda_with_args(
    Data&& data,
    camp::list<targLists...> const&)
{
  camp::get<LoopIndex>(data.bodies)(
      LambdaArgSwitchboard<Types, targLists>::extract(data)...);
}

/*!
 * A RAJA::kernel statement that invokes a lambda function
 * with user specified arguments.
 */
template<camp::idx_t LambdaIndex, typename... Args, typename Types>
struct StatementExecutor<statement::Lambda<LambdaIndex, Args...>, Types>
{

  template<typename Data>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(Data&& data)
  {

    // Convert SegList, ParamList into Seg, Param types, and store in a list
    using targList = typename camp::flatten<camp::list<Args...>>::type;

    invoke_lambda_with_args<LambdaIndex, Types>(std::forward<Data>(data),
                                                targList {});
  }
};

template<camp::idx_t LambdaIndex,
         typename Types,
         typename Data,
         camp::idx_t... SEGS,
         camp::idx_t... PARAMS>
RAJA_INLINE RAJA_HOST_DEVICE void invoke_lambda(Data&& data,
                                                camp::idx_seq<SEGS...> const&,
                                                camp::idx_seq<PARAMS...> const&)
{

  using AllSegs   = Segs<SEGS...>;
  using AllParams = Params<PARAMS...>;

  // invoke the expanded Lambda executor, passing in all segments and params
  StatementExecutor<statement::Lambda<LambdaIndex, AllSegs, AllParams>,
                    Types>::exec(std::forward<Data>(data));
}

template<camp::idx_t LambdaIndex, typename Types>
struct StatementExecutor<statement::Lambda<LambdaIndex>, Types>
{

  template<typename Data>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(Data&& data)
  {

    using Data_t         = camp::decay<Data>;
    using offset_tuple_t = typename Data_t::offset_tuple_t;
    using param_tuple_t  = typename Data_t::param_tuple_t;

    invoke_lambda<LambdaIndex, Types>(
        std::forward<Data>(data),
        camp::make_idx_seq_t<camp::tuple_size<offset_tuple_t>::value> {},
        camp::make_idx_seq_t<camp::tuple_size<param_tuple_t>::value> {});
  }
};


}  // namespace internal

}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
