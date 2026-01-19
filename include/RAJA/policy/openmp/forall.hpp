/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set and segment iteration
 *          template methods for OpenMP.
 *
 *          These methods should work on any platform that supports OpenMP.
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

#ifndef RAJA_forall_openmp_HPP
#define RAJA_forall_openmp_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_OPENMP)

#include <iostream>
#include <type_traits>

#include <omp.h>

#include "RAJA/util/types.hpp"
#include "RAJA/pattern/kernel/TypeTraits.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/policy/openmp/policy.hpp"

#include "RAJA/pattern/forall.hpp"
#include "RAJA/pattern/region.hpp"

#include "RAJA/pattern/params/forall.hpp"

#include "RAJA/policy/openmp/params/forall.hpp"

namespace RAJA
{

namespace policy
{
namespace omp
{

template<typename Iterable,
         typename Func,
         typename InnerPolicy,
         typename ForallParam>
RAJA_INLINE concepts::enable_if_t<
    resources::EventProxy<resources::Host>,
    RAJA::expt::type_traits::is_ForallParamPack<ForallParam>,
    RAJA::expt::type_traits::is_ForallParamPack_empty<ForallParam>>
forall_impl(resources::Host host_res,
            const omp_parallel_exec<InnerPolicy>&,
            Iterable&& iter,
            Func&& loop_body,
            ForallParam f_params)
{
  if constexpr (!RAJA::internal::is_wrapper_with_reducers<
                    camp::decay<Func>>::value)
  {
    RAJA::region<RAJA::omp_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto body = thread_privatize(loop_body);
      forall_impl(host_res, InnerPolicy {}, iter, body.get_priv(), f_params);
    });
    return resources::EventProxy<resources::Host>(host_res);
  }
  else
  {
    auto reducers_tuple = loop_body.data.param_tuple;

    using EXEC_POL = camp::decay<InnerPolicy>;
    RAJA::expt::detail::init_params<EXEC_POL>(reducers_tuple);

    using RAJA::internal::thread_privatize;
    RAJA_UNUSED_VAR(EXEC_POL {});
    RAJA_EXTRACT_BED_IT(iter);
    RAJA_OMP_DECLARE_TUPLE_REDUCTION_COMBINE;

#pragma omp parallel
    {
      auto body = thread_privatize(loop_body);
#pragma omp for reduction(combine : reducers_tuple)
      for (decltype(distance_it) i = 0; i < distance_it; ++i)
      {
        body.get_priv()(begin_it[i]);
        // Note: this is inefficient. However, the structure of loop data
        // requires us to perform this manual copy. This is because body
        // performs the local reduction on its own copy of the reducers, not the
        // OpenMP managed copy of the reducers_tuple.  Alternatively, we could
        // have OpenMP use LoopData as the combination object, but this would
        // require additional changes to make LoopData trivially constructable
        // (a requirement for OpenMP combinations.)
        reducers_tuple = body.get_priv().data.param_tuple;
      }
    }
    RAJA::expt::detail::resolve_params<EXEC_POL>(reducers_tuple);
  }
  return resources::EventProxy<resources::Host>(host_res);
}

///
/// OpenMP parallel for schedule policy implementation
///

namespace internal
{

/// Tag dispatch for omp forall

//
// omp for (Auto)
//
template<typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const ::RAJA::policy::omp::Auto&,
                             Iterable&& iter,
                             Func&& loop_body)
{
  RAJA_EXTRACT_BED_IT(iter);
#pragma omp for
  for (decltype(distance_it) i = 0; i < distance_it; ++i)
  {
    loop_body(begin_it[i]);
  }
}

//
// omp for schedule(static, ChunkSize)
//
template<typename Iterable, typename Func, int ChunkSize>
RAJA_INLINE void forall_impl(const ::RAJA::policy::omp::Static<ChunkSize>&,
                             Iterable&& iter,
                             Func&& loop_body)
{
  RAJA_EXTRACT_BED_IT(iter);
  if constexpr (ChunkSize > 0)
  {
#pragma omp for schedule(static, ChunkSize)
    for (decltype(distance_it) i = 0; i < distance_it; ++i)
    {
      loop_body(begin_it[i]);
    }
  }
  else
  {
#pragma omp for schedule(static)
    for (decltype(distance_it) i = 0; i < distance_it; ++i)
    {
      loop_body(begin_it[i]);
    }
  }
}

//
// omp for schedule(dynamic) / schedule(dynamic, ChunkSize)
//
template<typename Iterable, typename Func, int ChunkSize>
RAJA_INLINE void forall_impl(const ::RAJA::policy::omp::Dynamic<ChunkSize>&,
                             Iterable&& iter,
                             Func&& loop_body)
{
  RAJA_EXTRACT_BED_IT(iter);
  if constexpr (ChunkSize > 0)
  {
#pragma omp for schedule(dynamic, ChunkSize)
    for (decltype(distance_it) i = 0; i < distance_it; ++i)
    {
      loop_body(begin_it[i]);
    }
  }
  else
  {
#pragma omp for schedule(dynamic)
    for (decltype(distance_it) i = 0; i < distance_it; ++i)
    {
      loop_body(begin_it[i]);
    }
  }
}

//
// omp for schedule(guided)/schedule(guided, ChunkSize)
//
template<typename Iterable, typename Func, int ChunkSize>
RAJA_INLINE void forall_impl(const ::RAJA::policy::omp::Guided<ChunkSize>&,
                             Iterable&& iter,
                             Func&& loop_body)
{
  RAJA_EXTRACT_BED_IT(iter);
  if constexpr (ChunkSize > 0)
  {
#pragma omp for schedule(guided, ChunkSize)
    for (decltype(distance_it) i = 0; i < distance_it; ++i)
    {
      loop_body(begin_it[i]);
    }
  }
  else
  {
#pragma omp for schedule(guided)
    for (decltype(distance_it) i = 0; i < distance_it; ++i)
    {
      loop_body(begin_it[i]);
    }
  }
}

//
// omp for schedule(runtime)
//
template<typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const ::RAJA::policy::omp::Runtime&,
                             Iterable&& iter,
                             Func&& loop_body)
{
  RAJA_EXTRACT_BED_IT(iter);
#pragma omp for schedule(runtime)
  for (decltype(distance_it) i = 0; i < distance_it; ++i)
  {
    loop_body(begin_it[i]);
  }
}

// TODO :: not implemented in forall param interface ...
#if !defined(RAJA_COMPILER_MSVC)
// dynamic & guided
template<typename Policy, typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const Policy&, Iterable&& iter, Func&& loop_body)
{
  omp_sched_t prev_sched;
  int prev_chunk;
  omp_get_schedule(&prev_sched, &prev_chunk);
  omp_set_schedule(Policy::schedule, Policy::chunk_size);
  forall_impl(::RAJA::policy::omp::Runtime {}, std::forward<Iterable>(iter),
              std::forward<Func>(loop_body));
  omp_set_schedule(prev_sched, prev_chunk);
}
#endif


/// Tag dispatch for omp forall with nowait

//
// omp for nowait (Auto)
//
template<typename Iterable, typename Func>
RAJA_INLINE void forall_impl_nowait(const ::RAJA::policy::omp::Auto&,
                                    Iterable&& iter,
                                    Func&& loop_body)
{
  RAJA_EXTRACT_BED_IT(iter);
#pragma omp for nowait
  for (decltype(distance_it) i = 0; i < distance_it; ++i)
  {
    loop_body(begin_it[i]);
  }
}

//
// omp for schedule(static) nowait/omp for schedule(static, ChunkSize) nowait
//
template<typename Iterable, typename Func, int ChunkSize>
RAJA_INLINE void forall_impl_nowait(
    const ::RAJA::policy::omp::Static<ChunkSize>&,
    Iterable&& iter,
    Func&& loop_body)
{
  RAJA_EXTRACT_BED_IT(iter);
  if constexpr (ChunkSize > 0)
  {
#pragma omp for schedule(static, ChunkSize) nowait
    for (decltype(distance_it) i = 0; i < distance_it; ++i)
    {
      loop_body(begin_it[i]);
    }
  }
  else
  {
#pragma omp for schedule(static) nowait
    for (decltype(distance_it) i = 0; i < distance_it; ++i)
    {
      loop_body(begin_it[i]);
    }
  }
}

// TODO :: not implemented in param interface...
#if !defined(RAJA_COMPILER_MSVC)
// dynamic & guided
template<typename Policy, typename Iterable, typename Func>
RAJA_INLINE void forall_impl_nowait(const Policy&,
                                    Iterable&& iter,
                                    Func&& loop_body)
{
  omp_sched_t prev_sched;
  int prev_chunk;
  omp_get_schedule(&prev_sched, &prev_chunk);
  omp_set_schedule(Policy::schedule, Policy::chunk_size);
  forall_impl_nowait(::RAJA::policy::omp::Runtime {},
                     std::forward<Iterable>(iter),
                     std::forward<Func>(loop_body));
  omp_set_schedule(prev_sched, prev_chunk);
}
#endif

}  // end namespace internal

template<typename Schedule,
         typename Iterable,
         typename Func,
         typename ForallParam>
RAJA_INLINE concepts::enable_if_t<
    resources::EventProxy<resources::Host>,
    RAJA::expt::type_traits::is_ForallParamPack<ForallParam>,
    RAJA::expt::type_traits::is_ForallParamPack_empty<ForallParam>>
forall_impl(resources::Host host_res,
            const omp_for_schedule_exec<Schedule>&,
            Iterable&& iter,
            Func&& loop_body,
            ForallParam)
{
  internal::forall_impl(Schedule {}, std::forward<Iterable>(iter),
                        std::forward<Func>(loop_body));
  return resources::EventProxy<resources::Host>(host_res);
}

template<typename Schedule,
         typename Iterable,
         typename Func,
         typename ForallParam>
RAJA_INLINE concepts::enable_if_t<
    resources::EventProxy<resources::Host>,
    RAJA::expt::type_traits::is_ForallParamPack<ForallParam>,
    RAJA::expt::type_traits::is_ForallParamPack_empty<ForallParam>>
forall_impl(resources::Host host_res,
            const omp_for_nowait_schedule_exec<Schedule>&,
            Iterable&& iter,
            Func&& loop_body,
            ForallParam)
{
  internal::forall_impl_nowait(Schedule {}, std::forward<Iterable>(iter),
                               std::forward<Func>(loop_body));
  return resources::EventProxy<resources::Host>(host_res);
}

//
//////////////////////////////////////////////////////////////////////
//
// The following function templates iterate over index set
// segments using omp execution. Segment execution is defined by
// segment execution policy template parameter.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Iterate over index set segments using an omp parallel loop and
 *         segment dependency graph. Individual segment execution will use
 *         execution policy template parameter.
 *
 *         This method assumes that a task dependency graph has been
 *         properly set up for each segment in the index set.
 *
 ******************************************************************************
 */

/*
 * TODO: Fix this!!!
 */

/*
template <typename SEG_EXEC_POLICY_T, typename LOOP_BODY, typename ...
SEG_TYPES>
RAJA_INLINE void forall(
    ExecPolicy<omp_taskgraph_segit, SEG_EXEC_POLICY_T>,
    const IndexSet<SEG_TYPES ...>& iset,
    LOOP_BODY loop_body)
{
  if (!iset.dependencyGraphSet()) {
    std::cerr << "\n RAJA IndexSet dependency graph not set , "
              << "FILE: " << __FILE__ << " line: " << __LINE__ << std::endl;
    RAJA_ABORT_OR_THROW("IndexSet dependency graph");
  }

  IndexSet& ncis = (*const_cast<IndexSet*>(&iset));

  int num_seg = ncis.getNumSegments();

#pragma omp parallel for schedule(static, 1)
  for (int isi = 0; isi < num_seg; ++isi) {
    IndexSetSegInfo* seg_info = ncis.getSegmentInfo(isi);
    DepGraphNode* task = seg_info->getDepGraphNode();

    task->wait();

    executeRangeList_forall<SEG_EXEC_POLICY_T>(seg_info, loop_body);

    task->reset();

    if (task->numDepTasks() != 0) {
      for (int ii = 0; ii < task->numDepTasks(); ++ii) {
        // Alternateively, we could get the return value of this call
        // and actively launch the task if we are the last depedent
        // task. In that case, we would not need the semaphore spin
        // loop above.
        int seg = task->depTaskNum(ii);
        DepGraphNode* dep = ncis.getSegmentInfo(seg)->getDepGraphNode();
        dep->satisfyOne();
      }
    }

  }  // iterate over segments of index set
}
*/

}  // namespace omp

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for if defined(RAJA_ENABLE_OPENMP)

#endif  // closing endif for header file include guard
