/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA resource definitions.
 *
 *          Definitions in this file will propagate to all RAJA header files.
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

#ifndef RAJA_resource_HPP
#define RAJA_resource_HPP

#include "camp/resource.hpp"
#if defined(RAJA_CUDA_ACTIVE)
#include "RAJA/policy/cuda/policy.hpp"
#endif
#if defined(RAJA_HIP_ACTIVE)
#include "RAJA/policy/hip/policy.hpp"
#endif
#if defined(RAJA_SYCL_ACTIVE)
#include "RAJA/policy/sycl/policy.hpp"
#endif
#include "RAJA/policy/sequential/policy.hpp"
#include "RAJA/policy/openmp_target/policy.hpp"
#include "RAJA/internal/get_platform.hpp"

namespace RAJA
{

namespace resources
{
using namespace camp::resources;

template<typename e>
struct get_resource
{
  using type = camp::resources::Host;
};

template<Platform>
struct get_resource_from_platform
{
  using type = camp::resources::Host;
};

template<typename ExecPol>
using resource_from_pol_t = typename get_resource_from_platform<
    detail::get_platform<ExecPol>::value>::type;

template<typename ExecPol>
constexpr resource_from_pol_t<ExecPol> get_default_resource()
{
  return resource_from_pol_t<ExecPol>::get_default();
}

#if defined(RAJA_CUDA_ACTIVE)
template<>
struct get_resource_from_platform<Platform::cuda>
{
  using type = camp::resources::Cuda;
};

template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         size_t BLOCKS_PER_SM,
         bool Async>
struct get_resource<::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                                             IterationGetter,
                                                             Concretizer,
                                                             BLOCKS_PER_SM,
                                                             Async>>
{
  using type = camp::resources::Cuda;
};

template<bool Async, int num_threads, size_t BLOCKS_PER_SM>
struct get_resource<::RAJA::policy::cuda::cuda_launch_explicit_t<Async,
                                                                 num_threads,
                                                                 BLOCKS_PER_SM>>
{
  using type = camp::resources::Cuda;
};

template<typename ISetIter,
         typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         size_t BLOCKS_PER_SM,
         bool Async>
struct get_resource<
    ExecPolicy<ISetIter,
               ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                                        IterationGetter,
                                                        Concretizer,
                                                        BLOCKS_PER_SM,
                                                        Async>>>
{
  using type = camp::resources::Cuda;
};
#endif

#if defined(RAJA_HIP_ACTIVE)
template<>
struct get_resource_from_platform<Platform::hip>
{
  using type = camp::resources::Hip;
};

template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         bool Async>
struct get_resource<
    ::RAJA::policy::hip::
        hip_exec<IterationMapping, IterationGetter, Concretizer, Async>>
{
  using type = camp::resources::Hip;
};

template<bool Async, int num_threads>
struct get_resource<::RAJA::policy::hip::hip_launch_t<Async, num_threads>>
{
  using type = camp::resources::Hip;
};

template<typename ISetIter,
         typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         bool Async>
struct get_resource<ExecPolicy<
    ISetIter,
    ::RAJA::policy::hip::
        hip_exec<IterationMapping, IterationGetter, Concretizer, Async>>>
{
  using type = camp::resources::Hip;
};
#endif

#if defined(RAJA_SYCL_ACTIVE)
template<>
struct get_resource_from_platform<Platform::sycl>
{
  using type = camp::resources::Sycl;
};

template<size_t BlockSize, bool Async>
struct get_resource<::RAJA::policy::sycl::sycl_exec<BlockSize, Async>>
{
  using type = camp::resources::Sycl;
};

template<bool Async, int num_threads>
struct get_resource<::RAJA::policy::sycl::sycl_launch_t<Async, num_threads>>
{
  using type = camp::resources::Sycl;
};

template<typename ISetIter, size_t BlockSize, bool Async>
struct get_resource<
    ExecPolicy<ISetIter, ::RAJA::policy::sycl::sycl_exec<BlockSize, Async>>>
{
  using type = camp::resources::Sycl;
};
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
template<>
struct get_resource_from_platform<Platform::omp_target>
{
  using type = camp::resources::Omp;
};

template<>
struct get_resource<::RAJA::policy::omp::omp_target_parallel_for_exec_nt>
{
  using type = camp::resources::Omp;
};

template<size_t ThreadsPerTeam>
struct get_resource<
    ::RAJA::policy::omp::omp_target_parallel_for_exec<ThreadsPerTeam>>
{
  using type = camp::resources::Omp;
};

template<typename ISetIter>
struct get_resource<
    ExecPolicy<ISetIter, ::RAJA::policy::omp::omp_target_parallel_for_exec_nt>>
{
  using type = camp::resources::Omp;
};

template<typename ISetIter, size_t ThreadsPerTeam>
struct get_resource<ExecPolicy<
    ISetIter,
    ::RAJA::policy::omp::omp_target_parallel_for_exec<ThreadsPerTeam>>>
{
  using type = camp::resources::Omp;
};
#endif

}  // end namespace resources

namespace type_traits
{
template<typename T>
struct is_resource : std::false_type
{};

template<>
struct is_resource<resources::Host> : std::true_type
{};
#if defined(RAJA_CUDA_ACTIVE)
template<>
struct is_resource<resources::Cuda> : std::true_type
{};
#endif
#if defined(RAJA_HIP_ACTIVE)
template<>
struct is_resource<resources::Hip> : std::true_type
{};
#endif
#if defined(RAJA_SYCL_ACTIVE)
template<>
struct is_resource<resources::Sycl> : std::true_type
{};
#endif
#if defined(RAJA_ENABLE_TARGET_OPENMP)
template<>
struct is_resource<resources::Omp> : std::true_type
{};
#endif
}  // end namespace type_traits

}  // end namespace RAJA

#endif  // RAJA_resources_HPP#
