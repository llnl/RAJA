//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-2025, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forall_param_openmp_HPP
#define RAJA_forall_param_openmp_HPP

namespace RAJA
{

namespace policy
{
namespace omp
{
namespace expt
{

namespace internal
{
//
// omp for (Auto)
//
template<typename ExecPol,
         typename Iterable,
         typename Func,
         typename ForallParam>
RAJA_INLINE concepts::enable_if<std::is_same<ExecPol, RAJA::policy::omp::Auto>>
forall_impl(const ExecPol& p,
            Iterable&& iter,
            Func&& loop_body,
            ForallParam&& f_params)
{
  using EXEC_POL = camp::decay<decltype(p)>;

  RAJA::expt::ParamMultiplexer::parampack_init(p, f_params);
  RAJA_OMP_DECLARE_REDUCTION_COMBINE;

  RAJA_EXTRACT_BED_IT(iter);
#pragma omp parallel
  {

    using RAJA::internal::thread_privatize;
    auto body = thread_privatize(loop_body);

#pragma omp for reduction(combine : f_params)
    for (decltype(distance_it) i = 0; i < distance_it; ++i)
    {
      RAJA::expt::invoke_body(f_params, body.get_priv(), begin_it[i]);
    }
  }

  RAJA::expt::ParamMultiplexer::parampack_resolve(p, f_params);
}

//
// omp for schedule(static)
//
template<template<int> class ExecPol,
         typename Iterable,
         typename Func,
         int ChunkSize,
         typename ForallParam>
RAJA_INLINE concepts::enable_if<
    std::is_same<ExecPol<ChunkSize>, RAJA::policy::omp::Static<ChunkSize>>>
forall_impl(const ExecPol<ChunkSize>& p,
            Iterable&& iter,
            Func&& loop_body,
            ForallParam&& f_params)
{
  using EXEC_POL = camp::decay<decltype(p)>;

  RAJA::expt::ParamMultiplexer::parampack_init(p, f_params);
  RAJA_OMP_DECLARE_REDUCTION_COMBINE;

  RAJA_EXTRACT_BED_IT(iter);
#pragma omp parallel
  {

    using RAJA::internal::thread_privatize;
    auto body = thread_privatize(loop_body);

    if constexpr (ChunkSize > 0)
    {
#pragma omp for schedule(static, ChunkSize) reduction(combine : f_params)
      for (decltype(distance_it) i = 0; i < distance_it; ++i)
      {
        RAJA::expt::invoke_body(f_params, body.get_priv(), begin_it[i]);
      }
    }
    else
    {
#pragma omp for schedule(static) reduction(combine : f_params)
      for (decltype(distance_it) i = 0; i < distance_it; ++i)
      {
        RAJA::expt::invoke_body(f_params, body.get_priv(), begin_it[i]);
      }
    }
  }

  RAJA::expt::ParamMultiplexer::parampack_resolve(p, f_params);
}

//
// omp for schedule(runtime)
//
template<typename Iterable, typename Func, typename ForallParam>
RAJA_INLINE void forall_impl(const ::RAJA::policy::omp::Runtime& p,
                             Iterable&& iter,
                             Func&& loop_body,
                             ForallParam&& f_params)
{
  using EXEC_POL = camp::decay<decltype(p)>;

  RAJA::expt::ParamMultiplexer::parampack_init(p, f_params);
  RAJA_OMP_DECLARE_REDUCTION_COMBINE;

  RAJA_EXTRACT_BED_IT(iter);
#pragma omp parallel
  {

    using RAJA::internal::thread_privatize;
    auto body = thread_privatize(loop_body);

#pragma omp for schedule(runtime) reduction(combine : f_params)
    for (decltype(distance_it) i = 0; i < distance_it; ++i)
    {
      RAJA::expt::invoke_body(f_params, body.get_priv(), begin_it[i]);
    }
  }

  RAJA::expt::ParamMultiplexer::parampack_resolve(p, f_params);
}

//
// omp for nowait (Auto)
//
template<typename Iterable, typename Func, typename ForallParam>
RAJA_INLINE void forall_impl_nowait(const ::RAJA::policy::omp::Auto& p,
                                    Iterable&& iter,
                                    Func&& loop_body,
                                    ForallParam&& f_params)
{
  using EXEC_POL = camp::decay<decltype(p)>;

  RAJA::expt::ParamMultiplexer::parampack_init(p, f_params);
  RAJA_OMP_DECLARE_REDUCTION_COMBINE;

  RAJA_EXTRACT_BED_IT(iter);
#pragma omp parallel
  {

    using RAJA::internal::thread_privatize;
    auto body = thread_privatize(loop_body);

#pragma omp for nowait reduction(combine : f_params)
    for (decltype(distance_it) i = 0; i < distance_it; ++i)
    {
      RAJA::expt::invoke_body(f_params, body.get_priv(), begin_it[i]);
    }
  }

  RAJA::expt::ParamMultiplexer::parampack_resolve(p, f_params);
}

//
// omp for schedule(dynamic, ChunkSize)
//
template<typename Iterable, typename Func, int ChunkSize, typename ForallParam>
RAJA_INLINE void forall_impl(const ::RAJA::policy::omp::Dynamic<ChunkSize>& p,
                             Iterable&& iter,
                             Func&& loop_body,
                             ForallParam&& f_params)
{
  using EXEC_POL = camp::decay<decltype(p)>;

  RAJA::expt::ParamMultiplexer::parampack_init(p, f_params);
  RAJA_OMP_DECLARE_REDUCTION_COMBINE;

  RAJA_EXTRACT_BED_IT(iter);
#pragma omp parallel
  {

    using RAJA::internal::thread_privatize;
    auto body = thread_privatize(loop_body);
    if constexpr (ChunkSize > 0)
    {
#pragma omp for schedule(dynamic, ChunkSize) reduction(combine : f_params)
      for (decltype(distance_it) i = 0; i < distance_it; ++i)
      {
        RAJA::expt::invoke_body(f_params, body.get_priv(), begin_it[i]);
      }
    }
    else
    {
#pragma omp for schedule(dynamic) reduction(combine : f_params)
      for (decltype(distance_it) i = 0; i < distance_it; ++i)
      {
        RAJA::expt::invoke_body(f_params, body.get_priv(), begin_it[i]);
      }
    }
  }


  RAJA::expt::ParamMultiplexer::parampack_resolve(p, f_params);
}

//
// omp for schedule(guided)
//
template<typename Iterable, typename Func, int ChunkSize, typename ForallParam>
RAJA_INLINE void forall_impl(const ::RAJA::policy::omp::Guided<ChunkSize>& p,
                             Iterable&& iter,
                             Func&& loop_body,
                             ForallParam&& f_params)
{
  using EXEC_POL = camp::decay<decltype(p)>;

  RAJA::expt::ParamMultiplexer::parampack_init(p, f_params);
  RAJA_OMP_DECLARE_REDUCTION_COMBINE;

  RAJA_EXTRACT_BED_IT(iter);
#pragma omp parallel
  {

    using RAJA::internal::thread_privatize;
    auto body = thread_privatize(loop_body);
    if constexpr (ChunkSize <= 0)
    {
#pragma omp for schedule(guided) reduction(combine : f_params)
      for (decltype(distance_it) i = 0; i < distance_it; ++i)
      {
        RAJA::expt::invoke_body(f_params, body.get_priv(), begin_it[i]);
      }
    }
    else
    {
#pragma omp for schedule(guided, ChunkSize) reduction(combine : f_params)
      for (decltype(distance_it) i = 0; i < distance_it; ++i)
      {
        RAJA::expt::invoke_body(f_params, body.get_priv(), begin_it[i]);
      }
    }
  }

  RAJA::expt::ParamMultiplexer::parampack_resolve(p, f_params);
}

//
// omp for schedule(static) nowait
//
template<typename Iterable, typename Func, int ChunkSize, typename ForallParam>
RAJA_INLINE void forall_impl_nowait(
    const ::RAJA::policy::omp::Static<ChunkSize>& p,
    Iterable&& iter,
    Func&& loop_body,
    ForallParam&& f_params)
{
  using EXEC_POL = camp::decay<decltype(p)>;

  RAJA::expt::ParamMultiplexer::parampack_init(p, f_params);
  RAJA_OMP_DECLARE_REDUCTION_COMBINE;

  RAJA_EXTRACT_BED_IT(iter);
#pragma omp parallel
  {

    using RAJA::internal::thread_privatize;
    auto body = thread_privatize(loop_body);
    if constexpr (ChunkSize <= 0)
    {
#pragma omp for schedule(static) nowait reduction(combine : f_params)
      for (decltype(distance_it) i = 0; i < distance_it; ++i)
      {
        RAJA::expt::invoke_body(f_params, body.get_priv(), begin_it[i]);
      }
    }
    else
    {
#pragma omp for schedule(static, ChunkSize) nowait reduction(combine : f_params)
      for (decltype(distance_it) i = 0; i < distance_it; ++i)
      {
        RAJA::expt::invoke_body(f_params, body.get_priv(), begin_it[i]);
      }
    }
  }
  RAJA::expt::ParamMultiplexer::parampack_resolve(p, f_params);
}
}  //  namespace internal

template<typename Schedule,
         typename Iterable,
         typename Func,
         typename ForallParam>
RAJA_INLINE resources::EventProxy<resources::Host> forall_impl(
    resources::Host host_res,
    const omp_for_schedule_exec<Schedule>&,
    Iterable&& iter,
    Func&& loop_body,
    ForallParam f_params)
{
  expt::internal::forall_impl(Schedule {}, std::forward<Iterable>(iter),
                              std::forward<Func>(loop_body),
                              std::forward<ForallParam>(f_params));
  return resources::EventProxy<resources::Host>(host_res);
}
}  //  namespace expt

///
/// OpenMP parallel policy implementation
///
template<typename Iterable,
         typename Func,
         typename InnerPolicy,
         typename ForallParam>
RAJA_INLINE concepts::enable_if_t<
    resources::EventProxy<resources::Host>,
    RAJA::expt::type_traits::is_ForallParamPack<ForallParam>,
    concepts::negate<
        RAJA::expt::type_traits::is_ForallParamPack_empty<ForallParam>>>
forall_impl(resources::Host host_res,
            const omp_parallel_exec<InnerPolicy>&,
            Iterable&& iter,
            Func&& loop_body,
            ForallParam f_params)
{
  expt::forall_impl(host_res, InnerPolicy {}, iter, loop_body, f_params);
  return resources::EventProxy<resources::Host>(host_res);
}

}  // namespace omp

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for header file include guard
