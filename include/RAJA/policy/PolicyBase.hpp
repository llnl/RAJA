/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for basic RAJA policy mechanics.
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

#ifndef RAJA_POLICYBASE_HPP
#define RAJA_POLICYBASE_HPP

#include "RAJA/util/camp_aliases.hpp"
#include "RAJA/util/concepts.hpp"

#include <cstddef>

namespace RAJA
{

enum class Policy
{
  undefined,
  sequential,
  simd,
  openmp,
  target_openmp,
  cuda,
  hip,
  sycl
};

enum class Pattern
{
  undefined,
  forall,
  region,
  reduce,
  multi_reduce,
  taskgraph,
  synchronize,
  workgroup,
  workgroup_exec,
  workgroup_order,
  workgroup_storage,
  workgroup_dispatch
};

enum class Launch
{
  undefined,
  sync,
  async
};

struct PolicyBase
{};

template<Policy Policy_,
         Pattern Pattern_,
         Launch Launch_,
         Platform Platform_,
         typename... Traits>
struct PolicyBaseT : PolicyBase
{
  static constexpr Policy policy     = Policy_;
  static constexpr Pattern pattern   = Pattern_;
  static constexpr Launch launch     = Launch_;
  static constexpr Platform platform = Platform_;
};

template<typename PolicyType>
struct policy_of
{
  static constexpr Policy value = PolicyType::policy;
};

template<typename PolicyType>
struct pattern_of
{
  static constexpr Pattern value = PolicyType::pattern;
};

template<typename PolicyType>
struct launch_of
{
  static constexpr Launch value = PolicyType::launch;
};

template<typename PolicyType>
struct platform_of
{
  static constexpr Platform value = PolicyType::platform;
};

template<typename PolicyType, RAJA::Policy P_>
struct policy_is : camp::num<policy_of<camp::decay<PolicyType>>::value == P_>
{};

template<typename PolicyType, RAJA::Policy... Ps_>
struct policy_any_of
    : camp::num<camp::concepts::any_of<policy_is<PolicyType, Ps_>...>::value>
{};

template<typename PolicyType, RAJA::Pattern P_>
struct pattern_is : camp::num<pattern_of<camp::decay<PolicyType>>::value == P_>
{};

template<typename PolicyType, RAJA::Launch L_>
struct launch_is : camp::num<launch_of<camp::decay<PolicyType>>::value == L_>
{};

template<typename PolicyType, RAJA::Platform P_>
struct platform_is
    : camp::num<platform_of<camp::decay<PolicyType>>::value == P_>
{};

template<typename PolicyType, typename Trait>
struct policy_has_trait_impl : camp::num<false>
{};

///
template<typename Trait,
         Policy Policy_,
         Pattern Pattern_,
         Launch Launch_,
         Platform Platform_,
         typename... Traits>
struct policy_has_trait_impl<
    PolicyBaseT<Policy_, Pattern_, Launch_, Platform_, Traits...>,
    Trait>
    : camp::num<camp::concepts::any_of<std::is_same<Trait, Traits>...>::value>
{};

///
template<typename PolicyType, typename Trait>
using policy_has_trait = policy_has_trait_impl<camp::decay<PolicyType>, Trait>;

template<typename Inner>
struct wrapper
{
  using inner = Inner;
};

namespace reduce
{

struct ordered
{};

struct unordered
{};

}  // namespace reduce

template<Policy Pol, Pattern Pat, typename... Args>
using make_policy_pattern_t =
    PolicyBaseT<Pol, Pat, Launch::undefined, Platform::undefined, Args...>;

template<Policy Policy_,
         Pattern Pattern_,
         Launch Launch_,
         Platform Platform_,
         typename... Args>
using make_policy_pattern_launch_platform_t =
    PolicyBaseT<Policy_, Pattern_, Launch_, Platform_, Args...>;

template<Policy Policy_, Pattern Pattern_, Launch Launch_, typename... Args>
using make_policy_pattern_launch_t =
    PolicyBaseT<Policy_, Pattern_, Launch_, Platform::undefined, Args...>;

template<Policy Policy_, Pattern Pattern_, Platform Platform_, typename... Args>
using make_policy_pattern_platform_t =
    PolicyBaseT<Policy_, Pattern_, Launch::undefined, Platform_, Args...>;

namespace concepts
{

template<typename Pol>
struct ExecutionPolicy
    : DefineConcept(::RAJA::concepts::has_type<::RAJA::Policy>(
                        camp::decay<decltype(Pol::policy)>()),
                    ::RAJA::concepts::has_type<::RAJA::Pattern>(
                        camp::decay<decltype(Pol::pattern)>()),
                    ::RAJA::concepts::has_type<::RAJA::Launch>(
                        camp::decay<decltype(Pol::launch)>()),
                    ::RAJA::concepts::has_type<::RAJA::Platform>(
                        camp::decay<decltype(Pol::platform)>()))
{};

}  // end namespace concepts

namespace type_traits
{

template<typename Pol>
struct is_sequential_policy : RAJA::policy_is<Pol, RAJA::Policy::sequential>
{};

template<typename Pol>
struct is_simd_policy : RAJA::policy_is<Pol, RAJA::Policy::simd>
{};

template<typename Pol>
struct is_openmp_policy : RAJA::policy_is<Pol, RAJA::Policy::openmp>
{};

template<typename Pol>
struct is_target_openmp_policy
    : RAJA::policy_is<Pol, RAJA::Policy::target_openmp>
{};

template<typename Pol>
struct is_cuda_policy : RAJA::policy_is<Pol, RAJA::Policy::cuda>
{};

template<typename Pol>
struct is_hip_policy : RAJA::policy_is<Pol, RAJA::Policy::hip>
{};

template<typename Pol>
struct is_sycl_policy : RAJA::policy_is<Pol, RAJA::Policy::sycl>
{};

template<typename Pol>
struct is_device_exec_policy
    : RAJA::policy_any_of<Pol, RAJA::Policy::cuda, RAJA::Policy::hip>
{};

DefineTypeTraitFromConcept(is_execution_policy,
                           RAJA::concepts::ExecutionPolicy);

template<typename Pol>
struct is_reduce_policy : RAJA::pattern_is<Pol, RAJA::Pattern::reduce>
{};

template<typename Pol>
struct is_multi_reduce_policy
    : RAJA::pattern_is<Pol, RAJA::Pattern::multi_reduce>
{};

}  // end namespace type_traits

}  // end namespace RAJA

#endif /* RAJA_POLICYBASE_HPP */
