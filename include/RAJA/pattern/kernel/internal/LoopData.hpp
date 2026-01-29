/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for loop kernel internals: LoopData structure and
 *          related helper functions.
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

#ifndef RAJA_pattern_kernel_internal_LoopData_HPP
#define RAJA_pattern_kernel_internal_LoopData_HPP

#include "RAJA/config.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "camp/camp.hpp"

#include "RAJA/pattern/detail/privatizer.hpp"
#include "RAJA/pattern/kernel/internal/StatementList.hpp"
#include "RAJA/pattern/kernel/internal/Template.hpp"
#include "RAJA/pattern/params/params_base.hpp"

#include <iterator>
#include <type_traits>

namespace RAJA
{
namespace internal
{


// Universal base of all For wrappers for type traits
struct ForList
{};

struct ForBase
{};

struct CollapseBase
{};

template<camp::idx_t ArgumentId, typename Policy>
struct ForTraitBase : public ForBase
{
  constexpr static camp::idx_t index_val = ArgumentId;
  using index                            = camp::num<ArgumentId>;
  using index_type  = camp::nil;  // default to invalid type
  using policy_type = Policy;
  using type        = ForTraitBase;  // make camp::value compatible
};

template<typename Iterator>
struct iterable_difftype_getter
{
  using type = typename std::iterator_traits<
      typename Iterator::iterator>::difference_type;
};

template<typename Segments>
using difftype_list_from_segments =
    typename camp::transform<iterable_difftype_getter, Segments>::type;


template<typename Segments>
using difftype_tuple_from_segments =
    typename camp::apply_l<camp::lambda<camp::tuple>,
                           difftype_list_from_segments<Segments>>::type;

template<typename Iterator>
struct iterable_value_type_getter
{
  using type =
      typename std::iterator_traits<typename Iterator::iterator>::value_type;
};

template<typename Segments>
using value_type_list_from_segments =
    typename camp::transform<iterable_value_type_getter, Segments>::type;


template<typename Segments>
using index_tuple_from_segments =
    typename camp::apply_l<camp::lambda<camp::tuple>,
                           value_type_list_from_segments<Segments>>::type;

template<typename Segments>
using index_types_from_segments =
    typename camp::apply_l<camp::lambda<camp::list>,
                           value_type_list_from_segments<Segments>>::type;

template<typename SegmentTuple,
         typename ParamTuple,
         typename Resource,
         typename... Bodies>
struct LoopData
{

  using Self = LoopData<SegmentTuple, ParamTuple, Resource, Bodies...>;

  // Offset tuple holds offset from begin() for each of the segments
  using offset_tuple_t =
      difftype_tuple_from_segments<typename SegmentTuple::TList>;

  // Used by LoopTypes and various execution policies to determine the
  // index value type of each segment
  using index_types_t = index_types_from_segments<typename SegmentTuple::TList>;

  // Tuple of segments that can be iterated over
  using segment_tuple_t = SegmentTuple;
  SegmentTuple segment_tuple;

  // Tuple of parameters that are thread privatized
  using param_tuple_t = ParamTuple;
  using arg_tuple_t =
      typename RAJA::expt::detail::ParamToArgHelper<ParamTuple>::type;
  ParamTuple param_tuple;

  Resource res;

  // Lambdas that were passed into the kernel
  using BodiesTuple = camp::tuple<Bodies...>;
  const BodiesTuple bodies;
  offset_tuple_t offset_tuple;

  // Vector sizes of each segment.  This is only used by the vector_exec
  // policies
  using vector_sizes_t = tuple_of_n<int, camp::tuple_size<SegmentTuple>::value>;
  vector_sizes_t vector_sizes;

  RAJA_INLINE RAJA_HOST_DEVICE constexpr LoopData(SegmentTuple const& s,
                                                  ParamTuple const& p,
                                                  Resource r,
                                                  Bodies const&... b)
      : segment_tuple(s),
        param_tuple(p),
        res(r),
        bodies(b...)
  {}

  constexpr LoopData(LoopData const&) = default;
  constexpr LoopData(LoopData&&)      = default;

  template<camp::idx_t Idx, typename IndexT>
  RAJA_HOST_DEVICE RAJA_INLINE void assign_offset(IndexT const& i)
  {
    camp::get<Idx>(offset_tuple) = i;
  }

  template<typename ParamId, typename IndexT>
  RAJA_HOST_DEVICE RAJA_INLINE void assign_param(IndexT const& i)
  {
    using param_t =
        camp::at_v<typename param_tuple_t::TList, ParamId::param_idx>;
    camp::get<ParamId::param_idx>(param_tuple) = param_t(i);
  }

  template<typename ParamId>
  RAJA_HOST_DEVICE RAJA_INLINE auto get_param()
      -> camp::at_v<typename param_tuple_t::TList, ParamId::param_idx>
  {
    return camp::get<ParamId::param_idx>(param_tuple);
  }

  RAJA_HOST_DEVICE RAJA_INLINE Resource get_resource() { return res; }
};

template<camp::idx_t ArgumentId, typename Data>
using segment_diff_type = typename std::iterator_traits<
    typename camp::at_v<typename Data::segment_tuple_t::TList,
                        ArgumentId>::iterator>::difference_type;

template<camp::idx_t ArgumentId, typename Data>
RAJA_INLINE RAJA_HOST_DEVICE auto segment_length(Data const& data)
    -> segment_diff_type<ArgumentId, Data>
{
  return camp::get<ArgumentId>(data.segment_tuple).end() -
         camp::get<ArgumentId>(data.segment_tuple).begin();
}

template<typename Data, typename Types, typename... EnclosedStmts>
struct GenericWrapper : GenericWrapperBase
{
  using data_t = camp::decay<Data>;

  data_t& data;

  RAJA_INLINE
  constexpr explicit GenericWrapper(data_t& d) : data {d} {}

  RAJA_INLINE
  void exec()
  {
    execute_statement_list<camp::list<EnclosedStmts...>, Types>(data);
  }
};

/*!
 * Convenience object used to create a thread-private LoopData object.
 */
template<typename T>
struct NestedPrivatizer
{
  using data_t         = typename T::data_t;
  using value_type     = camp::decay<T>;
  using reference_type = value_type&;

  data_t privatized_data;
  value_type privatized_wrapper;

  RAJA_INLINE
  constexpr NestedPrivatizer(const T& o)
      : privatized_data {o.data},
        privatized_wrapper(privatized_data)
  {}

  RAJA_INLINE
  reference_type get_priv() { return privatized_wrapper; }
};


}  // end namespace internal
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_internal_LoopData_HPP */
