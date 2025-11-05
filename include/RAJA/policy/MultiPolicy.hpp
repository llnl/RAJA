/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA wrapper for "multi-policy" and dynamic policy selection
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_MultiPolicy_HPP
#define RAJA_MultiPolicy_HPP

#include "RAJA/config.hpp"

#include <tuple>

#include "RAJA/policy/PolicyBase.hpp"

#include "RAJA/internal/get_platform.hpp"
#include "RAJA/util/plugins.hpp"

#include "RAJA/util/concepts.hpp"
#include "RAJA/util/resource.hpp"

namespace RAJA
{

namespace detail
{
template<size_t index, size_t size, typename Policy, typename... rest>
struct policy_invoker;
}

namespace policy
{
namespace multi
{

/// MultiPolicy - Meta-policy for choosing between a compile-time list of
/// policies at runtime
///
/// \tparam Selector Functor/Lambda/function type used to select policies
/// \tparam Policies Variadic pack of policies, numbered from 0
template<typename Selector, typename... Policies>
class MultiPolicy
{
  Selector _selector;

public:
  MultiPolicy() = delete;  // No default construction

  MultiPolicy(Selector s) : _selector(s), _policies({Policies {}...}) {}

  MultiPolicy(Selector s, Policies... policies)
      : _selector(s),
        _policies({policies...})
  {}

  MultiPolicy(const MultiPolicy& p)
      : _selector(p._selector),
        _policies(p._policies)
  {}

  template<typename Iterable, typename Body>
  int invoke(Iterable&& i, Body&& b)
  {
    size_t index = _selector(i);
    _policies.invoke(index, i, b);
    return _selector(i);
  }

  detail::
      policy_invoker<sizeof...(Policies) - 1, sizeof...(Policies), Policies...>
          _policies;
};

/// forall_impl - MultiPolicy specialization, select at runtime from a
/// compile-time list of policies, build with make_multi_policy()
/// \param p MultiPolicy to use for selection
/// \param iter iterable of items to supply to body
/// \param body functor, will receive each value produced by iterable iter
template<typename Iterable,
         typename Body,
         typename Selector,
         typename... Policies>
RAJA_INLINE void forall_impl(MultiPolicy<Selector, Policies...> p,
                             Iterable&& iter,
                             Body&& body)
{
  p.invoke(iter, body);
}

template<typename Res,
         typename Iterable,
         typename Body,
         typename Selector,
         typename... Policies>
RAJA_INLINE resources::EventProxy<Res> forall_impl(
    Res r,
    MultiPolicy<Selector, Policies...> p,
    Iterable&& iter,
    Body&& body)
{
  p.invoke(iter, body);
  return resources::EventProxy<Res>(r);
}

}  // end namespace multi
}  // end namespace policy

using policy::multi::MultiPolicy;

namespace detail
{

template<camp::idx_t... Indices, typename... Policies, typename Selector>
auto make_multi_policy(camp::idx_seq<Indices...>,
                       Selector s,
                       std::tuple<Policies...> policies)
    -> MultiPolicy<Selector, Policies...>
{
  return MultiPolicy<Selector, Policies...>(s, std::get<Indices>(policies)...);
}
}  // namespace detail

/// make_multi_policy - Construct a MultiPolicy from the given selector and
/// Policies
///
/// \tparam Policies list of policies, 0 to N-1
/// \tparam Selector type of s, should almost always be inferred
/// \param s functor called with the segment object passed to
/// forall, must return an int in the set 0 to N-1 selecting the policy to use
/// \return A MultiPolicy containing the given selector s
template<typename... Policies, typename Selector>
RAJA_DEPRECATE("In the next RAJA Release, MultiPolicy will be deprecated.")
auto make_multi_policy(Selector s) -> MultiPolicy<Selector, Policies...>
{
  return MultiPolicy<Selector, Policies...>(s, Policies {}...);
}

/// make_multi_policy - Construct a MultiPolicy from the given selector and
/// Policies
///
/// \tparam Policies list of policies, inferred from policies
/// \tparam Selector type of s, should almost always be inferred
/// \param policies tuple of policies, allows value-carrying policies
/// \param s functor called with the segment object passed to
/// forall, must return an int in the set 0 to N-1 selecting the policy to use
/// \return A MultiPolicy containing the given selector s
template<typename... Policies, typename Selector>
RAJA_DEPRECATE("In the next RAJA Release, MultiPolicy will be deprecated.")
auto make_multi_policy(std::tuple<Policies...> policies, Selector s)
    -> MultiPolicy<Selector, Policies...>
{
  return detail::make_multi_policy(camp::make_idx_seq_t<sizeof...(Policies)> {},
                                   s, policies);
}

namespace detail
{

template<size_t index, size_t size, typename Policy, typename... rest>
struct policy_invoker : public policy_invoker<index - 1, size, rest...>
{
  static_assert(index < size, "index must be in the range of possibilities");
  Policy _p;
  using NextInvoker = policy_invoker<index - 1, size, rest...>;

  policy_invoker(Policy p, rest... args) : NextInvoker(args...), _p(p) {}

  template<typename Iterable, typename LoopBody>
  void invoke(int offset, Iterable&& iter, LoopBody&& loop_body)
  {
    if (offset == size - index - 1)
    {

      util::PluginContext context {util::make_context<Policy>(std::string())};
      util::callPreCapturePlugins(context);

      using RAJA::util::trigger_updates_before;
      auto body = trigger_updates_before(loop_body);

      util::callPostCapturePlugins(context);

      util::callPreLaunchPlugins(context);

      using policy::multi::forall_impl;
      RAJA_FORCEINLINE_RECURSIVE
      auto r = resources::get_resource<Policy>::type::get_default();
      forall_impl(r, _p, std::forward<Iterable>(iter), body);

      util::callPostLaunchPlugins(context);
    }
    else
    {
      NextInvoker::invoke(offset, std::forward<Iterable>(iter),
                          std::forward<LoopBody>(loop_body));
    }
  }
};

template<size_t size, typename Policy, typename... rest>
struct policy_invoker<0, size, Policy, rest...>
{
  Policy _p;

  policy_invoker(Policy p, rest...) : _p(p) {}

  template<typename Iterable, typename LoopBody>
  void invoke(int offset, Iterable&& iter, LoopBody&& loop_body)
  {
    if (offset == size - 1)
    {

      util::PluginContext context {util::make_context<Policy>(std::string())};
      util::callPreCapturePlugins(context);

      using RAJA::util::trigger_updates_before;
      auto body = trigger_updates_before(loop_body);

      util::callPostCapturePlugins(context);

      util::callPreLaunchPlugins(context);

      // std::cout <<"policy_invoker: No index\n";
      using policy::multi::forall_impl;
      RAJA_FORCEINLINE_RECURSIVE
      auto r = resources::get_resource<Policy>::type::get_default();
      forall_impl(r, _p, std::forward<Iterable>(iter), body);

      util::callPostLaunchPlugins(context);
    }
    else
    {
      throw std::runtime_error("unknown offset invoked");
    }
  }
};

}  // end namespace detail

namespace type_traits
{

template<typename T>
struct is_multi_policy
    : ::RAJA::type_traits::SpecializationOf<RAJA::MultiPolicy,
                                            typename std::decay<T>::type>
{};
}  // namespace type_traits

}  // end namespace RAJA

#endif
