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

#include "RAJA/pattern/detail/TypeTraits.hpp"
#include "RAJA/util/camp_aliases.hpp"
#include "RAJA/util/concepts.hpp"

#include <cstddef>
#include <stdexcept>

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

constexpr const char* get_policy_name(Policy p)
{
  switch (p)
  {
    case Policy::undefined:
      return "undefined";
    case Policy::sequential:
      return "sequential";
    case Policy::simd:
      return "simd";
    case Policy::openmp:
      return "openmp";
    case Policy::target_openmp:
      return "target_openmp";
    case Policy::cuda:
      return "cuda";
    case Policy::hip:
      return "hip";
    case Policy::sycl:
      return "sycl";
    default:
      return "unknown";
  }
}

template<RAJA::Policy... policies>
struct PolicyList
{};

template<RAJA::Policy BackendPolicy>
struct reduction_supported_policies
{
  using type = RAJA::PolicyList<>;
};

template<RAJA::Policy BackendPolicy>
using reduction_supported_policies_t =
    typename reduction_supported_policies<BackendPolicy>::type;

template<Policy p>
inline constexpr bool policy_active = false;

template<>
inline constexpr bool policy_active<Policy::undefined> = false;

template<>
inline constexpr bool policy_active<Policy::sequential> = true;

template<>
inline constexpr bool policy_active<Policy::simd> = true;

template<>
inline constexpr bool policy_active<Policy::openmp> =
#ifdef RAJA_OPENMP_ACTIVE
    true;
#else
    false;
#endif

template<>
inline constexpr bool policy_active<Policy::target_openmp> =
#if defined(RAJA_OPENMP_ACTIVE) && defined(RAJA_ENABLE_TARGET_OPENMP)
    true;
#else
    false;
#endif

template<>
inline constexpr bool policy_active<Policy::cuda> =
#ifdef RAJA_CUDA_ACTIVE
    true;
#else
    false;
#endif

template<>
inline constexpr bool policy_active<Policy::hip> =
#ifdef RAJA_HIP_ACTIVE
    true;
#else
    false;
#endif

template<>
inline constexpr bool policy_active<Policy::sycl> =
#ifdef RAJA_SYCL_ACTIVE
    true;
#else
    false;
#endif

// check that policy is supported, undefined or an active policy in the list
template<Policy... supported_policies>
constexpr bool policy_supported(PolicyList<supported_policies...>, Policy p)
{
  return ((p == Policy::undefined) || ... ||
          (p == supported_policies && policy_active<supported_policies>));
}

// check that policy is supported, otherwise throw an exception
template<Policy... supported_policies>
inline bool policy_supported_or_throw(const char* context_name,
                                      PolicyList<supported_policies...> list,
                                      Policy p)
{
  if (policy_supported(list, p))
  {
    return true;
  }
  std::string msg;
  msg += context_name;
  msg += ": unsupported policy ";
  msg += get_policy_name(p);
  throw std::runtime_error(msg);
  return false;
}

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

constexpr const char* get_pattern_name(Pattern p)
{
  switch (p)
  {
    case Pattern::undefined:
      return "undefined";
    case Pattern::forall:
      return "forall";
    case Pattern::region:
      return "region";
    case Pattern::reduce:
      return "reduce";
    case Pattern::multi_reduce:
      return "multi_reduce";
    case Pattern::taskgraph:
      return "taskgraph";
    case Pattern::synchronize:
      return "synchronize";
    case Pattern::workgroup:
      return "workgroup";
    case Pattern::workgroup_exec:
      return "workgroup_exec";
    case Pattern::workgroup_order:
      return "workgroup_order";
    case Pattern::workgroup_storage:
      return "workgroup_storage";
    case Pattern::workgroup_dispatch:
      return "workgroup_dispatch";
    default:
      return "unknown";
  }
}

enum class Launch
{
  undefined,
  sync,
  async
};

constexpr const char* get_launch_name(Launch l)
{
  switch (l)
  {
    case Launch::undefined:
      return "undefined";
    case Launch::sync:
      return "sync";
    case Launch::async:
      return "async";
    default:
      return "unknown";
  }
}

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
