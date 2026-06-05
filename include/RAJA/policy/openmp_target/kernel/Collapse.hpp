//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_openmp_target_kernel_Collapse_HPP
#define RAJA_policy_openmp_target_kernel_Collapse_HPP

#include "RAJA/pattern/kernel/internal.hpp"

namespace RAJA
{
namespace internal
{

template<camp::idx_t Arg0,
         camp::idx_t Arg1,
         typename... EnclosedStmts,
         typename Types>
struct StatementExecutor<statement::Collapse<omp_target_parallel_collapse_exec,
                                             ArgList<Arg0, Arg1>,
                                             EnclosedStmts...>,
                         Types>
{
  template<typename Data>
  static RAJA_INLINE void exec(Data&& data)
  {
    auto l0 = segment_length<Arg0>(data);
    auto l1 = segment_length<Arg1>(data);

    // Set the argument types for this loop
    using NewTypes0 = setSegmentTypeFromData<Types, Arg0, Data>;
    using NewTypes1 = setSegmentTypeFromData<NewTypes0, Arg1, Data>;

    using RAJA::internal::thread_privatize;
    auto privatizer = thread_privatize(data);
#pragma omp target teams distribute parallel for schedule(static, 1)           \
    firstprivate(privatizer) collapse(2)
    for (auto i0 = (decltype(l0))0; i0 < l0; ++i0)
    {
      for (auto i1 = (decltype(l1))0; i1 < l1; ++i1)
      {
        auto& private_data = privatizer.get_priv();
        private_data.template assign_offset<Arg0>(i0);
        private_data.template assign_offset<Arg1>(i1);
        execute_statement_list<camp::list<EnclosedStmts...>, NewTypes1>(
            private_data);
      }
    }
  }
};

template<camp::idx_t Arg0,
         camp::idx_t Arg1,
         camp::idx_t Arg2,
         typename... EnclosedStmts,
         typename Types>
struct StatementExecutor<statement::Collapse<omp_target_parallel_collapse_exec,
                                             ArgList<Arg0, Arg1, Arg2>,
                                             EnclosedStmts...>,
                         Types>
{
  template<typename Data>
  static RAJA_INLINE void exec(Data&& data)
  {
    auto l0 = segment_length<Arg0>(data);
    auto l1 = segment_length<Arg1>(data);
    auto l2 = segment_length<Arg2>(data);

    // Set the argument types for this loop
    using NewTypes0 = setSegmentTypeFromData<Types, Arg0, Data>;
    using NewTypes1 = setSegmentTypeFromData<NewTypes0, Arg1, Data>;
    using NewTypes2 = setSegmentTypeFromData<NewTypes1, Arg2, Data>;

    using RAJA::internal::thread_privatize;
    auto privatizer = thread_privatize(data);
#pragma omp target teams distribute parallel for schedule(static, 1)           \
    firstprivate(privatizer) collapse(3)
    for (auto i0 = (decltype(l0))0; i0 < l0; ++i0)
    {
      for (auto i1 = (decltype(l1))0; i1 < l1; ++i1)
      {
        for (auto i2 = (decltype(l2))0; i2 < l2; ++i2)
        {
          auto& private_data = privatizer.get_priv();
          private_data.template assign_offset<Arg0>(i0);
          private_data.template assign_offset<Arg1>(i1);
          private_data.template assign_offset<Arg2>(i2);
          execute_statement_list<camp::list<EnclosedStmts...>, NewTypes2>(
              private_data);
        }
      }
    }
  }
};

template<camp::idx_t Arg0,
         camp::idx_t Arg1,
         camp::idx_t Arg2,
         camp::idx_t Arg3,
         typename... EnclosedStmts,
         typename Types>
struct StatementExecutor<statement::Collapse<omp_target_parallel_collapse_exec,
                                             ArgList<Arg0, Arg1, Arg2, Arg3>,
                                             EnclosedStmts...>,
                         Types>
{
  template<typename Data>
  static RAJA_INLINE void exec(Data&& data)
  {
    auto l0 = segment_length<Arg0>(data);
    auto l1 = segment_length<Arg1>(data);
    auto l2 = segment_length<Arg2>(data);
    auto l3 = segment_length<Arg3>(data);

    // Set the argument types for this loop
    using NewTypes0 = setSegmentTypeFromData<Types, Arg0, Data>;
    using NewTypes1 = setSegmentTypeFromData<NewTypes0, Arg1, Data>;
    using NewTypes2 = setSegmentTypeFromData<NewTypes1, Arg2, Data>;
    using NewTypes3 = setSegmentTypeFromData<NewTypes2, Arg3, Data>;

    using RAJA::internal::thread_privatize;
    auto privatizer = thread_privatize(data);
#pragma omp target teams distribute parallel for schedule(static, 1)           \
    firstprivate(privatizer) collapse(4)
    for (auto i0 = (decltype(l0))0; i0 < l0; ++i0)
    {
      for (auto i1 = (decltype(l1))0; i1 < l1; ++i1)
      {
        for (auto i2 = (decltype(l2))0; i2 < l2; ++i2)
        {
          for (auto i3 = (decltype(l3))0; i3 < l3; ++i3)
          {
            auto& private_data = privatizer.get_priv();
            private_data.template assign_offset<Arg0>(i0);
            private_data.template assign_offset<Arg1>(i1);
            private_data.template assign_offset<Arg2>(i2);
            private_data.template assign_offset<Arg3>(i2);
            execute_statement_list<camp::list<EnclosedStmts...>, NewTypes3>(
                private_data);
          }
        }
      }
    }
  }
};

}  // namespace internal
}  // namespace RAJA

#endif  // RAJA_policy_openmp_target_kernel_Collapse_HPP
