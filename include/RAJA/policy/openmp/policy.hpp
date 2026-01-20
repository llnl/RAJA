/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA OpenMP policy definitions.
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

#ifndef policy_openmp_HPP
#define policy_openmp_HPP

#include <type_traits>

#include "RAJA/policy/PolicyBase.hpp"

// Rely on builtin_atomic when OpenMP can't do the job
#include "RAJA/policy/atomic_builtin.hpp"

#if defined(RAJA_COMPILER_MSVC)
typedef enum omp_sched_t
{
  // schedule kinds
  omp_sched_static  = 0x1,
  omp_sched_dynamic = 0x2,
  omp_sched_guided  = 0x3,
  omp_sched_auto    = 0x4,

  // schedule modifier
  omp_sched_monotonic = 0x80000000u
} omp_sched_t;
#else
#include <omp.h>
#endif

namespace RAJA
{
namespace omp
{

enum struct multi_reduce_algorithm : int
{
  combine_on_destruction,
  combine_on_get
};

template<multi_reduce_algorithm t_algorithm>
struct MultiReduceTuning
{
  static constexpr multi_reduce_algorithm algorithm = t_algorithm;
  static constexpr bool consistent =
      (algorithm == multi_reduce_algorithm::combine_on_get);
};

}  // namespace omp

namespace policy
{
namespace omp
{

namespace internal
{
struct ScheduleTag
{};

template<omp_sched_t Sched, int Chunk>
struct Schedule : public ScheduleTag
{
  constexpr static omp_sched_t schedule = Sched;
  constexpr static int chunk_size       = Chunk;
  constexpr static Policy policy        = Policy::openmp;
};
}  // namespace internal

//
//////////////////////////////////////////////////////////////////////
//
// Basic tag types
//
//////////////////////////////////////////////////////////////////////
//

struct Parallel
{};

struct For
{};

struct NoWait
{};

static constexpr int default_chunk_size = -1;

struct Auto : public internal::Schedule<omp_sched_auto, default_chunk_size>
{};

template<int ChunkSize = default_chunk_size>
struct Static : public internal::Schedule<omp_sched_static, ChunkSize>
{};

template<int ChunkSize = default_chunk_size>
using Dynamic = internal::Schedule<omp_sched_dynamic, ChunkSize>;

template<int ChunkSize = default_chunk_size>
using Guided = internal::Schedule<omp_sched_guided, ChunkSize>;

struct Runtime : private internal::Schedule<static_cast<omp_sched_t>(-1),
                                            default_chunk_size>
{};

//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////
//

///
///  Struct supporting OpenMP parallel region.
///
struct omp_parallel_region
    : make_policy_pattern_launch_platform_t<Policy::openmp,
                                            Pattern::region,
                                            Launch::undefined,
                                            Platform::host>
{};

///
///  Struct supporting OpenMP parallel region for Teams
///
struct omp_launch_t : make_policy_pattern_launch_platform_t<Policy::openmp,
                                                            Pattern::region,
                                                            Launch::undefined,
                                                            Platform::host>
{};

///
///  Struct supporting OpenMP 'for nowait schedule( )'
///
template<typename Sched>
struct omp_for_nowait_schedule_exec
    : make_policy_pattern_launch_platform_t<Policy::openmp,
                                            Pattern::forall,
                                            Launch::undefined,
                                            Platform::host,
                                            omp::For,
                                            omp::NoWait,
                                            Sched>
{
  static_assert(
      std::is_base_of<::RAJA::policy::omp::internal::ScheduleTag, Sched>::value,
      "Schedule type must be one of: Auto|Runtime|Static|Dynamic|Guided");
};

///
///  Struct supporting OpenMP 'for schedule( )'
///
template<typename Sched>
struct omp_for_schedule_exec
    : make_policy_pattern_launch_platform_t<Policy::openmp,
                                            Pattern::forall,
                                            Launch::undefined,
                                            Platform::host,
                                            omp::For,
                                            Sched>
{
  static_assert(
      std::is_base_of<::RAJA::policy::omp::internal::ScheduleTag, Sched>::value,
      "Schedule type must be one of: Auto|Runtime|Static|Dynamic|Guided");
};

///
///  Internal type aliases supporting 'omp for schedule( )' for specific
///  schedule types.
///
using omp_for_exec = omp_for_schedule_exec<Auto>;

///
template<int ChunkSize = default_chunk_size>
using omp_for_static_exec = omp_for_schedule_exec<omp::Static<ChunkSize>>;

///
template<int ChunkSize = default_chunk_size>
using omp_for_dynamic_exec = omp_for_schedule_exec<omp::Dynamic<ChunkSize>>;

///
template<int ChunkSize = default_chunk_size>
using omp_for_guided_exec = omp_for_schedule_exec<omp::Guided<ChunkSize>>;

///
using omp_for_runtime_exec = omp_for_schedule_exec<omp::Runtime>;


///
///  Internal type aliases supporting 'omp for schedule( ) nowait' for specific
///  schedule types.
///
///  IMPORTANT: We only provide a nowait policy option for static scheduling
///             since that is the only scheduling case that can be used with
///             nowait and be correct in general. Paraphrasing the OpenMP
///             standard:
///
///             Programs that depend on which thread executes a particular
///             iteration under any circumstance other than static schedule
///             are non-conforming.
///
template<int ChunkSize = default_chunk_size>
using omp_for_nowait_static_exec =
    omp_for_nowait_schedule_exec<omp::Static<ChunkSize>>;

///
///  Struct supporting OpenMP 'parallel' region containing an inner loop
///  execution construct.
///
template<typename InnerPolicy>
using omp_parallel_exec =
    make_policy_pattern_launch_platform_t<Policy::openmp,
                                          Pattern::forall,
                                          Launch::undefined,
                                          Platform::host,
                                          omp::Parallel,
                                          wrapper<InnerPolicy>>;

///
///  Internal type aliases supporting 'omp parallel for schedule( )' for
///  specific schedule types.
///
using omp_parallel_for_exec = omp_parallel_exec<omp_for_exec>;

///
template<int ChunkSize = default_chunk_size>
using omp_parallel_for_static_exec =
    omp_parallel_exec<omp_for_schedule_exec<omp::Static<ChunkSize>>>;

///
template<int ChunkSize = default_chunk_size>
using omp_parallel_for_dynamic_exec =
    omp_parallel_exec<omp_for_schedule_exec<omp::Dynamic<ChunkSize>>>;

///
template<int ChunkSize = default_chunk_size>
using omp_parallel_for_guided_exec =
    omp_parallel_exec<omp_for_schedule_exec<omp::Guided<ChunkSize>>>;

///
using omp_parallel_for_runtime_exec =
    omp_parallel_exec<omp_for_schedule_exec<omp::Runtime>>;


///
///////////////////////////////////////////////////////////////////////
///
/// Basic Indexset segment iteration policies
///
///////////////////////////////////////////////////////////////////////
///
using omp_parallel_for_segit = omp_parallel_for_exec;

///
using omp_parallel_segit = omp_parallel_for_segit;

///
///////////////////////////////////////////////////////////////////////
///
/// Taskgraph Indexset segment iteration policies
///
///////////////////////////////////////////////////////////////////////
///
struct omp_taskgraph_segit
    : make_policy_pattern_t<Policy::openmp, Pattern::taskgraph, omp::Parallel>
{};

///
struct omp_taskgraph_interval_segit
    : make_policy_pattern_t<Policy::openmp, Pattern::taskgraph, omp::Parallel>
{};

///
///////////////////////////////////////////////////////////////////////
///
/// WorkGroup execution policies
///
///////////////////////////////////////////////////////////////////////
///
struct omp_work : make_policy_pattern_launch_platform_t<Policy::openmp,
                                                        Pattern::workgroup_exec,
                                                        Launch::sync,
                                                        Platform::host>
{};

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///
struct omp_reduce : make_policy_pattern_t<Policy::openmp, Pattern::reduce>
{};

///
struct omp_reduce_ordered
    : make_policy_pattern_t<Policy::openmp, Pattern::reduce, reduce::ordered>
{};

///
template<typename tuning>
struct omp_multi_reduce_policy : make_policy_pattern_launch_platform_t<
                                     Policy::openmp,
                                     Pattern::multi_reduce,
                                     Launch::undefined,
                                     Platform::host,
                                     std::conditional_t<tuning::consistent,
                                                        reduce::ordered,
                                                        reduce::unordered>>
{};

///
struct omp_synchronize : make_policy_pattern_launch_t<Policy::openmp,
                                                      Pattern::synchronize,
                                                      Launch::sync>
{};

#if defined(RAJA_COMPILER_MSVC)

// For MS Visual C, just default to builtin_atomic for everything
using omp_atomic = builtin_atomic;

#else  // RAJA_COMPILER_MSVC not defined

struct omp_atomic
{};

#endif

struct omp_thread
{};

template<RAJA::omp::multi_reduce_algorithm algorithm>
using omp_multi_reduce_tuning =
    omp_multi_reduce_policy<RAJA::omp::MultiReduceTuning<algorithm>>;

// Policies for RAJA::MultiReduce* objects with specific behaviors.
// - combine_on_destruction policies combine new values into a single value for
//   each object then each object combines its values into the parent object's
//   values on destruction in a critical region.
using omp_multi_reduce_combine_on_destruction = omp_multi_reduce_tuning<
    RAJA::omp::multi_reduce_algorithm::combine_on_destruction>;
// - combine_on_get policies combine new values into a single value for
//   each thread then when get is called those values are combined.
using omp_multi_reduce_combine_on_get =
    omp_multi_reduce_tuning<RAJA::omp::multi_reduce_algorithm::combine_on_get>;

// Policy for RAJA::MultiReduce* objects that gives the
// same answer every time when used in the same way
using omp_multi_reduce_ordered = omp_multi_reduce_combine_on_get;

// Policy for RAJA::MultiReduce* objects that may not give the
// same answer every time when used in the same way
using omp_multi_reduce_unordered = omp_multi_reduce_combine_on_destruction;

using omp_multi_reduce = omp_multi_reduce_unordered;

}  // namespace omp
}  // namespace policy

///
///////////////////////////////////////////////////////////////////////
///
/// Type aliases exposed to users in the RAJA namespace.
///
///////////////////////////////////////////////////////////////////////
///

///
/// Type alias for atomics
///
using policy::omp::omp_atomic;

///
/// Type alias for thread
///
using policy::omp::omp_thread;

///
/// Type aliases to simplify common omp parallel for loop execution
///
using policy::omp::omp_parallel_for_exec;
///
using policy::omp::omp_parallel_for_static_exec;
///
using policy::omp::omp_parallel_for_dynamic_exec;
///
using policy::omp::omp_parallel_for_guided_exec;
///
using policy::omp::omp_parallel_for_runtime_exec;

///
/// Type aliases for omp parallel for iteration over indexset segments
///
using policy::omp::omp_parallel_for_segit;
///
using policy::omp::omp_parallel_segit;

///
/// Type alias for omp parallel region containing an inner 'omp for' loop
/// execution policy. Inner policy types follow.
///
using policy::omp::omp_parallel_exec;

///
/// Type alias for 'omp for' loop execution within an omp_parallel_exec
/// construct
///
using policy::omp::omp_for_exec;

///
/// Type aliases for 'omp for' and 'omp for nowait' loop execution with a
/// scheduling policy within an omp_parallel_exec construct
/// Scheduling policies are near the top of this file and include:
/// RAJA::policy::omp::{Auto, Static, Dynamic, Guided, Runtime}
///
/// Helper aliases to make usage less verbose for common use cases follow these.
///
/// Important: 'nowait' schedule must be used with care to guarantee code
///             correctness.
///
using policy::omp::omp_for_schedule_exec;
///
using policy::omp::omp_for_nowait_schedule_exec;

///
/// Type aliases for 'omp for' and 'omp for nowait' loop execution with a
/// static scheduling policy within an omp_parallel_exec construct
///
using policy::omp::omp_for_static_exec;
///
using policy::omp::omp_for_nowait_static_exec;
///
using policy::omp::omp_for_dynamic_exec;
///
using policy::omp::omp_for_guided_exec;
///
using policy::omp::omp_for_runtime_exec;

///
/// Type aliases for omp parallel region
///
using policy::omp::omp_launch_t;
using policy::omp::omp_parallel_region;

///
/// Type aliases for omp reductions
///
using policy::omp::omp_reduce;
///
using policy::omp::omp_reduce_ordered;
///
using policy::omp::omp_multi_reduce;
///
using policy::omp::omp_multi_reduce_ordered;

///
/// Type aliases for omp reductions
///
using policy::omp::omp_synchronize;

///
using policy::omp::omp_work;

}  // namespace RAJA

#endif
