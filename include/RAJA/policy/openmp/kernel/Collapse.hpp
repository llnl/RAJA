/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_openmp_kernel_collapse_HPP
#define RAJA_policy_openmp_kernel_collapse_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_OPENMP)

#include "RAJA/pattern/detail/privatizer.hpp"

#include "RAJA/pattern/kernel/Collapse.hpp"
#include "RAJA/pattern/kernel/internal.hpp"
#include "RAJA/pattern/params/forall.hpp"
#include "RAJA/pattern/kernel/Lambda.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/openmp/policy.hpp"

namespace RAJA
{

struct omp_parallel_collapse_exec
    : make_policy_pattern_t<RAJA::Policy::openmp,
                            RAJA::Pattern::forall,
                            RAJA::policy::omp::For>
{};

namespace internal
{

/////////
// Collapsing two loops
/////////

template<camp::idx_t Arg0,
         camp::idx_t Arg1,
         typename... EnclosedStmts,
         typename Types>
struct StatementExecutor<statement::Collapse<omp_parallel_collapse_exec,
                                             ArgList<Arg0, Arg1>,
                                             EnclosedStmts...>,
                         Types>
{


  template<typename Data>
  static RAJA_INLINE concepts::enable_if<
      RAJA::internal::loop_data_has_reducers<camp::decay<Data>>>
  exec(Data&& data)
  {
    const auto l0 = segment_length<Arg0>(data);
    const auto l1 = segment_length<Arg1>(data);
    // NOTE: these are here to avoid a use-after-scope detected by address
    // sanitizer, probably a false positive, but the result should be
    // essentially identical
    auto i0 = l0;
    auto i1 = l1;

    // Set the argument types for this loop
    using NewTypes0 = setSegmentTypeFromData<Types, Arg0, Data>;
    using NewTypes1 = setSegmentTypeFromData<NewTypes0, Arg1, Data>;

    using RAJA::internal::thread_privatize;
    auto reducers_tuple = data.param_tuple;
    RAJA::expt::detail::init_params<omp_parallel_collapse_exec>(reducers_tuple);
    auto privatizer = thread_privatize(data);
    using EXEC_POL  = omp_parallel_collapse_exec;
    RAJA_UNUSED_VAR(EXEC_POL {});
    RAJA_OMP_DECLARE_TUPLE_REDUCTION_COMBINE;
#pragma omp parallel for private(i0, i1) firstprivate(privatizer)              \
    RAJA_COLLAPSE(2) reduction(combine                                         \
                               : reducers_tuple)
    for (i0 = 0; i0 < l0; ++i0)
    {
      for (i1 = 0; i1 < l1; ++i1)

      {
        auto& private_data = privatizer.get_priv();
        private_data.template assign_offset<Arg0>(i0);
        private_data.template assign_offset<Arg1>(i1);
        execute_statement_list<camp::list<EnclosedStmts...>, NewTypes1>(
            private_data);
        // Note: we don't want to do this copy, but it is necessary for now
        // due to limitations of the LoopData interface in OpenMP combine
        // calls. See note in policy/openmp/forall.hpp for more detail.
        reducers_tuple = private_data.param_tuple;
      }
    }
    RAJA::expt::detail::resolve_params<EXEC_POL>(reducers_tuple);
  }

  template<typename Data>
  static RAJA_INLINE concepts::enable_if<concepts::negate<
      RAJA::internal::loop_data_has_reducers<camp::decay<Data>>>>
  exec(Data&& data)
  {
    const auto l0 = segment_length<Arg0>(data);
    const auto l1 = segment_length<Arg1>(data);
    // NOTE: these are here to avoid a use-after-scope detected by address
    // sanitizer, probably a false positive, but the result should be
    // essentially identical
    auto i0 = l0;
    auto i1 = l1;

    // Set the argument types for this loop
    using NewTypes0 = setSegmentTypeFromData<Types, Arg0, Data>;
    using NewTypes1 = setSegmentTypeFromData<NewTypes0, Arg1, Data>;

    using RAJA::internal::thread_privatize;
    auto privatizer = thread_privatize(data);
#pragma omp parallel for private(i0, i1) firstprivate(privatizer)              \
    RAJA_COLLAPSE(2)
    for (i0 = 0; i0 < l0; ++i0)
    {
      for (i1 = 0; i1 < l1; ++i1)
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
struct StatementExecutor<statement::Collapse<omp_parallel_collapse_exec,
                                             ArgList<Arg0, Arg1, Arg2>,
                                             EnclosedStmts...>,
                         Types>
{


  template<typename Data>
  static RAJA_INLINE concepts::enable_if<
      RAJA::internal::loop_data_has_reducers<camp::decay<Data>>>
  exec(Data&& data)
  {
    const auto l0 = segment_length<Arg0>(data);
    const auto l1 = segment_length<Arg1>(data);
    const auto l2 = segment_length<Arg2>(data);
    auto i0       = l0;
    auto i1       = l1;
    auto i2       = l2;

    // Set the argument types for this loop
    using NewTypes0 = setSegmentTypeFromData<Types, Arg0, Data>;
    using NewTypes1 = setSegmentTypeFromData<NewTypes0, Arg1, Data>;
    using NewTypes2 = setSegmentTypeFromData<NewTypes1, Arg2, Data>;

    auto reducers_tuple = data.param_tuple;
    RAJA::expt::detail::init_params<omp_parallel_collapse_exec>(reducers_tuple);
    using RAJA::internal::thread_privatize;
    auto privatizer = thread_privatize(data);
    using EXEC_POL  = omp_parallel_collapse_exec;
    RAJA_OMP_DECLARE_TUPLE_REDUCTION_COMBINE;
#pragma omp parallel for private(i0, i1, i2) firstprivate(privatizer)          \
    RAJA_COLLAPSE(3) reduction(combine                                         \
                               : reducers_tuple)
    for (i0 = 0; i0 < l0; ++i0)
    {
      for (i1 = 0; i1 < l1; ++i1)
      {
        for (i2 = 0; i2 < l2; ++i2)
        {
          auto& private_data = privatizer.get_priv();
          private_data.template assign_offset<Arg0>(i0);
          private_data.template assign_offset<Arg1>(i1);
          private_data.template assign_offset<Arg2>(i2);
          execute_statement_list<camp::list<EnclosedStmts...>, NewTypes2>(
              private_data);
          reducers_tuple = private_data.param_tuple;
        }
      }
    }

    RAJA::expt::detail::resolve_params<EXEC_POL>(reducers_tuple);
  }

  template<typename Data>
  static RAJA_INLINE concepts::enable_if<concepts::negate<
      RAJA::internal::loop_data_has_reducers<camp::decay<Data>>>>
  exec(Data&& data)
  {
    const auto l0 = segment_length<Arg0>(data);
    const auto l1 = segment_length<Arg1>(data);
    const auto l2 = segment_length<Arg2>(data);
    auto i0       = l0;
    auto i1       = l1;
    auto i2       = l2;

    // Set the argument types for this loop
    using NewTypes0 = setSegmentTypeFromData<Types, Arg0, Data>;
    using NewTypes1 = setSegmentTypeFromData<NewTypes0, Arg1, Data>;
    using NewTypes2 = setSegmentTypeFromData<NewTypes1, Arg2, Data>;

    using RAJA::internal::thread_privatize;
    auto privatizer = thread_privatize(data);
#pragma omp parallel for private(i0, i1, i2) firstprivate(privatizer)          \
    RAJA_COLLAPSE(3)
    for (i0 = 0; i0 < l0; ++i0)
    {
      for (i1 = 0; i1 < l1; ++i1)
      {
        for (i2 = 0; i2 < l2; ++i2)
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


}  // namespace internal
}  // namespace RAJA

#undef RAJA_COLLAPSE

#endif  // closing endif for RAJA_ENABLE_OPENMP guard

#endif  // closing endif for header file include guard
