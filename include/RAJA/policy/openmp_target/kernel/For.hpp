//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_openmp_target_kernel_For_HPP
#define RAJA_policy_openmp_target_kernel_For_HPP

#include "RAJA/pattern/kernel/internal.hpp"
#include <type_traits>

namespace RAJA
{
namespace internal
{

// OpenMP target needs by-value LoopData copy; the generic wrapper uses references.
// Handles both For and ForICount.
template<camp::idx_t ArgumentId,
         typename ParamId,
         typename Data,
         typename Types,
         typename... EnclosedStmts>
struct OpenMPTargetForXWrapper : public GenericWrapperBase
{
  using data_t = camp::decay<Data>;

  data_t data;

  /*!
   * \brief Deferences data so that it can be mapped to the device
   */
  RAJA_INLINE
  constexpr explicit OpenMPTargetForXWrapper(data_t& d) : data {d} {}

  RAJA_INLINE
  void exec()
  {
    execute_statement_list<camp::list<EnclosedStmts...>, Types>(data);
  }

  template<typename InIndexType>
  RAJA_INLINE void operator()(InIndexType i)
  {
    data.template assign_offset<ArgumentId>(i);
    if constexpr (!std::is_same<ParamId, camp::nil>::value)
    {
      data.template assign_param<ParamId>(i);
    }
    exec();
  }
};

template<camp::idx_t ArgumentId,
         typename ParamId,
         typename ExecPolicy,
         typename Types,
         typename... EnclosedStmts>
struct OpenMPTargetForXExec
{

  template<typename Data>
  static RAJA_INLINE void exec(Data&& data)
  {
    // Set the argument type for this loop
    using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

    OpenMPTargetForXWrapper<ArgumentId, ParamId, Data, NewTypes, EnclosedStmts...>
        for_wrapper(data);

    auto len    = segment_length<ArgumentId>(data);
    using len_t = decltype(len);

    auto r = resources::Omp::get_default();
    forall_impl(r, ExecPolicy {},
                TypedRangeSegment<len_t>(0, len), for_wrapper,
                RAJA::expt::get_empty_forall_param_pack());
  }
};


// Partial specializations on For, ForICount, omp_target_parallel_for_exec<N>, and _nt
template<camp::idx_t ArgumentId,
         int N,
         typename... EnclosedStmts,
         typename Types>
struct StatementExecutor<statement::For<ArgumentId,
                                        omp_target_parallel_for_exec<N>,
                                        EnclosedStmts...>,
                         Types>
  : OpenMPTargetForXExec<ArgumentId,
                         camp::nil,
                         omp_target_parallel_for_exec<N>,
                         Types,
                         EnclosedStmts...>
{};

template<camp::idx_t ArgumentId,
         typename... EnclosedStmts,
         typename Types>
struct StatementExecutor<statement::For<ArgumentId,
                                        omp_target_parallel_for_exec_nt,
                                        EnclosedStmts...>,
                         Types>
  : OpenMPTargetForXExec<ArgumentId,
                         camp::nil,
                         omp_target_parallel_for_exec_nt,
                         Types,
                         EnclosedStmts...>
{};

template<camp::idx_t ArgumentId,
         typename ParamId,
         int N,
         typename... EnclosedStmts,
         typename Types>
struct StatementExecutor<statement::ForICount<ArgumentId,
                                              ParamId,
                                              omp_target_parallel_for_exec<N>,
                                              EnclosedStmts...>,
                         Types>
  : OpenMPTargetForXExec<ArgumentId,
                         ParamId,
                         omp_target_parallel_for_exec<N>,
                         Types,
                         EnclosedStmts...>
{};

template<camp::idx_t ArgumentId,
         typename ParamId,
         typename... EnclosedStmts,
         typename Types>
struct StatementExecutor<statement::ForICount<ArgumentId,
                                              ParamId,
                                              omp_target_parallel_for_exec_nt,
                                              EnclosedStmts...>,
                         Types>
  : OpenMPTargetForXExec<ArgumentId,
                         ParamId,
                         omp_target_parallel_for_exec_nt,
                         Types,
                         EnclosedStmts...>
{};


}  // namespace internal
}  // namespace RAJA

#endif  // RAJA_policy_openmp_kernel_For_HPP
