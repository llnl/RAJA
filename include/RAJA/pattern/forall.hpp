/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set and segment iteration
 *          template methods that take an execution policy as a template
 *          parameter.
 *
 *          The templates for segments support the following usage pattern:
 *
 *             forall<exec_policy>( index set, loop body );
 *
 *          which is equivalent to:
 *
 *             forall( exec_policy(), index set, loop body );
 *
 *          The former is slightly more concise. Here, the execution policy
 *          type is associated with a tag struct defined in the exec_poilicy
 *          hearder file. Usage of the forall_Icount() is similar.
 *
 *          The forall() and forall_Icount() methods that take an index set
 *          take an execution policy of the form:
 *
 *          TypedIndexSet::ExecPolicy< seg_it_policy, seg_exec_policy >
 *
 *          Here, the first template parameter determines the scheme for
 *          iteratiing over the index set segments and the second determines
 *          how each segment is executed.
 *
 *          The forall() templates accept a loop body argument that takes
 *          a single Index_type argument identifying the index of a loop
 *          iteration. The forall_Icount() templates accept a loop body that
 *          takes two Index_type arguments. The first is the number of the
 *          iteration in the indes set or segment, the second if the actual
 *          index of the loop iteration.
 *
 *
 *          IMPORTANT: Use of any of these methods requires a specialization
 *                     for the given index set type and execution policy.
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

#ifndef RAJA_forall_generic_HPP
#define RAJA_forall_generic_HPP

#include "RAJA/config.hpp"

#include <functional>
#include <iterator>
#include <type_traits>

#include "RAJA/internal/Iterators.hpp"

#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/policy/MultiPolicy.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/util/concepts.hpp"
#include "RAJA/util/Span.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/sequential/forall.hpp"

#include "RAJA/pattern/detail/forall.hpp"
#include "RAJA/pattern/detail/privatizer.hpp"
#include "RAJA/pattern/params/kernel_name.hpp"

#include "RAJA/internal/get_platform.hpp"
#include "RAJA/util/plugins.hpp"

#include "RAJA/util/resource.hpp"

namespace RAJA
{

//
//////////////////////////////////////////////////////////////////////
//
// Iteration over generic iterators
//
//////////////////////////////////////////////////////////////////////
//

namespace detail
{
/// Adapter to replace specific implementations for the icount variants
template<typename Range, typename Body, typename IndexT>
struct icount_adapter
{
  using index_type = typename std::decay<IndexT>::type;
  typename std::decay<Body>::type body;
  using container_type = typename std::decay<Range>::type;
  typename container_type::iterator begin_it;
  Index_type icount;

  icount_adapter(Range const& r, Body const& b, IndexT icount_)
      : body {b},
        icount {icount_}
  {
    using std::begin;
    begin_it = begin(r);
  }

  RAJA_SUPPRESS_HD_WARN
  template<typename T>
  RAJA_HOST_DEVICE void operator()(T const& i) const
  {
    body(static_cast<index_type>(i + icount), begin_it[i]);
  }
};

struct CallForall
{
  template<typename T,
           typename ExecPol,
           typename Body,
           typename Res,
           typename ForallParams>
  RAJA_INLINE camp::resources::EventProxy<Res> operator()(T const&,
                                                          ExecPol,
                                                          Body,
                                                          Res,
                                                          ForallParams) const;
};

struct CallForallIcount
{
  constexpr CallForallIcount(int s);

  template<typename T,
           typename ExecPol,
           typename Body,
           typename Res,
           typename ForallParams>
  RAJA_INLINE camp::resources::EventProxy<Res> operator()(T const&,
                                                          ExecPol,
                                                          Body,
                                                          Res,
                                                          ForallParams) const;

  const int start;
};
}  // namespace detail

/*!
 ******************************************************************************
 *
 * \brief The RAJA::wrap layer unwraps dynamic policies before dispatch
 *
 ******************************************************************************
 */
namespace wrap
{


/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers with a value-based policy
 *
 ******************************************************************************
 */
template<typename Res,
         typename ExecutionPolicy,
         typename Container,
         typename LoopBody,
         typename ForallParams>
RAJA_INLINE concepts::enable_if_t<
    RAJA::resources::EventProxy<Res>,
    concepts::negate<type_traits::is_indexset_policy<ExecutionPolicy>>,
    type_traits::is_range<Container>>
forall(Res r,
       ExecutionPolicy&& p,
       Container&& c,
       LoopBody&& loop_body,
       ForallParams&& f_params)
{
  RAJA_FORCEINLINE_RECURSIVE
  return forall_impl(
      r, std::forward<ExecutionPolicy>(p), std::forward<Container>(c),
      std::forward<LoopBody>(loop_body), std::forward<ForallParams>(f_params));
}

template<typename Res,
         typename ExecutionPolicy,
         typename Container,
         typename LoopBody>
RAJA_INLINE concepts::enable_if_t<
    RAJA::resources::EventProxy<Res>,
    concepts::negate<type_traits::is_indexset_policy<ExecutionPolicy>>,
    type_traits::is_range<Container>>
forall(Res r, ExecutionPolicy&& p, Container&& c, LoopBody&& loop_body)
{
  RAJA_FORCEINLINE_RECURSIVE
  return forall_impl(
      r, std::forward<ExecutionPolicy>(p), std::forward<Container>(c),
      std::forward<LoopBody>(loop_body), expt::get_empty_forall_param_pack());
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers with a value-based policy with icount
 *
 ******************************************************************************
 */
template<typename Res,
         typename ExecutionPolicy,
         typename Container,
         typename IndexType,
         typename LoopBody,
         typename ForallParams>
RAJA_INLINE resources::EventProxy<Res> forall_Icount(Res r,
                                                     ExecutionPolicy&& p,
                                                     Container&& c,
                                                     IndexType&& icount,
                                                     LoopBody&& loop_body,
                                                     ForallParams&& f_params)
{
  using std::begin;
  using std::distance;
  using std::end;
  auto range = RangeSegment(0, distance(begin(c), end(c)));
  detail::icount_adapter<Container, LoopBody, IndexType> adapted(c, loop_body,
                                                                 icount);
  using policy::sequential::forall_impl;
  RAJA_FORCEINLINE_RECURSIVE
  return forall_impl(r, std::forward<ExecutionPolicy>(p), range, adapted,
                     std::forward<ForallParams>(f_params));
}

/*!
******************************************************************************
*
* \brief Execute segments from forall_Icount traversal method.
*
*         For usage example, see reducers.hxx.
*
******************************************************************************
*/
template<typename Res,
         typename SegmentIterPolicy,
         typename SegmentExecPolicy,
         typename... SegmentTypes,
         typename LoopBody,
         typename ForallParams>
RAJA_INLINE resources::EventProxy<Res> forall_Icount(
    Res r,
    ExecPolicy<SegmentIterPolicy, SegmentExecPolicy>,
    const TypedIndexSet<SegmentTypes...>& iset,
    LoopBody loop_body,
    ForallParams f_params)
{
  // no need for icount variant here
  auto segIterRes =
      resources::get_resource<SegmentIterPolicy>::type::get_default();
  wrap::forall(segIterRes, SegmentIterPolicy(), iset, [=, &r](int segID) {
    iset.segmentCall(segID,
                     detail::CallForallIcount(iset.getStartingIcount(segID)),
                     SegmentExecPolicy(), loop_body, r, f_params);
  });
  return RAJA::resources::EventProxy<Res>(r);
}

template<typename Res,
         typename SegmentIterPolicy,
         typename SegmentExecPolicy,
         typename LoopBody,
         typename... SegmentTypes,
         typename ForallParams>
RAJA_INLINE resources::EventProxy<Res> forall(
    Res r,
    ExecPolicy<SegmentIterPolicy, SegmentExecPolicy>,
    const TypedIndexSet<SegmentTypes...>& iset,
    LoopBody loop_body,
    ForallParams f_params)
{
  auto segIterRes =
      resources::get_resource<SegmentIterPolicy>::type::get_default();
  wrap::forall(segIterRes, SegmentIterPolicy(), iset, [=, &r](int segID) {
    iset.segmentCall(segID, detail::CallForall {}, SegmentExecPolicy(),
                     loop_body, r, f_params);
  });
  return RAJA::resources::EventProxy<Res>(r);
}

}  // end namespace wrap

/*!
 ******************************************************************************
 *
 * \brief The RAJA::policy_by_value_interface forall functions provide an
 *interface with value-based policies. It also enforces the interface and
 *performs static checks as well as triggering plugins and loop body updates.
 *
 ******************************************************************************
 */
inline namespace policy_by_value_interface
{


/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over index set with icount with a value-based policy
 *
 ******************************************************************************
 */
template<typename ExecutionPolicy,
         typename Res,
         typename IdxSet,
         typename... Params>
RAJA_INLINE resources::EventProxy<Res> forall_Icount(ExecutionPolicy&& p,
                                                     Res r,
                                                     IdxSet&& c,
                                                     Params&&... params)
{
  static_assert(type_traits::is_index_set<IdxSet>::value,
                "Expected a TypedIndexSet but did not get one. Are you using "
                "a TypedIndexSet policy by mistake?");

  std::string kernel_name =
      expt::get_kernel_name(std::forward<Params>(params)...);
  auto f_params = expt::make_forall_param_pack(std::forward<Params>(params)...);
  auto&& loop_body = expt::get_lambda(std::forward<Params>(params)...);
  // expt::check_forall_optional_args(loop_body, f_params);

  util::PluginContext context {
      util::make_context<camp::decay<ExecutionPolicy>>(std::move(kernel_name))};
  util::callPreCapturePlugins(context);

  using RAJA::util::trigger_updates_before;
  auto body = trigger_updates_before(loop_body);

  util::callPostCapturePlugins(context);

  util::callPreLaunchPlugins(context);

  RAJA::resources::EventProxy<Res> e =
      wrap::forall_Icount(r, std::forward<ExecutionPolicy>(p),
                          std::forward<IdxSet>(c), std::move(body), f_params);

  util::callPostLaunchPlugins(context);
  return e;
}

template<typename ExecutionPolicy,
         typename IdxSet,
         typename LoopBody,
         typename Res = typename resources::get_resource<ExecutionPolicy>::type>
RAJA_INLINE resources::EventProxy<Res> forall_Icount(ExecutionPolicy&& p,
                                                     IdxSet&& c,
                                                     LoopBody&& loop_body)
{
  auto r = Res::get_default();
  return ::RAJA::policy_by_value_interface::forall_Icount(
      std::forward<ExecutionPolicy>(p), r, std::forward<IdxSet>(c),
      std::forward<LoopBody>(loop_body));
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over index set with a value-based policy
 *
 ******************************************************************************
 */
template<typename ExecutionPolicy,
         typename Res,
         typename IdxSet,
         typename... Params>
RAJA_INLINE concepts::enable_if_t<
    resources::EventProxy<Res>,
    type_traits::is_indexset_policy<ExecutionPolicy>>
forall(ExecutionPolicy&& p, Res r, IdxSet&& c, Params&&... params)
{
  static_assert(type_traits::is_index_set<IdxSet>::value,
                "Expected a TypedIndexSet but did not get one. Are you using "
                "a TypedIndexSet policy by mistake?");

  auto f_params = expt::make_forall_param_pack(std::forward<Params>(params)...);

  std::string kernel_name =
      expt::get_kernel_name(std::forward<Params>(params)...);
  auto&& loop_body = expt::get_lambda(std::forward<Params>(params)...);
  expt::check_forall_optional_args(loop_body, f_params);

  util::PluginContext context {
      util::make_context<camp::decay<ExecutionPolicy>>(std::move(kernel_name))};
  util::callPreCapturePlugins(context);

  using RAJA::util::trigger_updates_before;
  auto body = trigger_updates_before(loop_body);

  util::callPostCapturePlugins(context);

  util::callPreLaunchPlugins(context);

  resources::EventProxy<Res> e =
      wrap::forall(r, std::forward<ExecutionPolicy>(p), std::forward<IdxSet>(c),
                   std::move(body), f_params);

  util::callPostLaunchPlugins(context);
  return e;
}

template<typename ExecutionPolicy,
         typename IdxSet,
         typename LoopBody,
         typename Res = typename resources::get_resource<ExecutionPolicy>::type>
RAJA_INLINE concepts::enable_if_t<
    resources::EventProxy<Res>,
    type_traits::is_indexset_policy<ExecutionPolicy>>
forall(ExecutionPolicy&& p, IdxSet&& c, LoopBody&& loop_body)
{
  auto r = Res::get_default();
  return ::RAJA::policy_by_value_interface::forall(
      std::forward<ExecutionPolicy>(p), r, std::forward<IdxSet>(c),
      std::forward<LoopBody>(loop_body));
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers with a multi policy
 *
 ******************************************************************************
 */
template<typename ExecutionPolicy,
         typename Container,
         typename LoopBody,
         typename Res = typename resources::get_resource<ExecutionPolicy>::type>
RAJA_INLINE concepts::enable_if_t<resources::EventProxy<Res>,
                                  type_traits::is_multi_policy<ExecutionPolicy>,
                                  type_traits::is_range<Container>>
forall(ExecutionPolicy&& p, Container&& c, LoopBody&& loop_body)
{
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container does not model RandomAccessIterator");

  auto r = Res::get_default();

  // plugins handled in multipolicy policy_invoker
  return forall_impl(r, std::forward<ExecutionPolicy>(p),
                     std::forward<Container>(c),
                     std::forward<LoopBody>(loop_body));
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers with icount with a value-based policy
 *
 ******************************************************************************
 */
template<typename ExecutionPolicy,
         typename Res,
         typename Container,
         typename IndexType,
         typename FirstParam,
         typename... Params>
RAJA_INLINE concepts::enable_if_t<resources::EventProxy<Res>,
                                  type_traits::is_range<Container>,
                                  type_traits::is_integral<IndexType>>
forall_Icount(ExecutionPolicy&& p,
              Res r,
              Container&& c,
              IndexType icount,
              FirstParam&& first,
              Params&&... params)
{
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container does not model RandomAccessIterator");

  auto f_params = expt::make_forall_param_pack(std::forward<FirstParam>(first),
                                               std::forward<Params>(params)...);
  std::string kernel_name =
      expt::get_kernel_name(std::forward<Params>(params)...);
  auto&& loop_body = expt::get_lambda(std::forward<FirstParam>(first),
                                      std::forward<Params>(params)...);
  // expt::check_forall_optional_args(loop_body, f_params);

  util::PluginContext context {
      util::make_context<camp::decay<ExecutionPolicy>>(std::move(kernel_name))};
  util::callPreCapturePlugins(context);

  using RAJA::util::trigger_updates_before;
  auto body = trigger_updates_before(loop_body);

  util::callPostCapturePlugins(context);

  util::callPreLaunchPlugins(context);

  resources::EventProxy<Res> e = wrap::forall_Icount(
      r, std::forward<ExecutionPolicy>(p), std::forward<Container>(c), icount,
      std::move(body), f_params);

  util::callPostLaunchPlugins(context);
  return e;
}

template<typename ExecutionPolicy,
         typename Container,
         typename IndexType,
         typename LoopBody,
         typename Res = typename resources::get_resource<ExecutionPolicy>::type>
RAJA_INLINE concepts::enable_if_t<
    resources::EventProxy<Res>,
    type_traits::is_range<Container>,
    concepts::negate<type_traits::is_indexset_policy<ExecutionPolicy>>,
    type_traits::is_integral<IndexType>>
forall_Icount(ExecutionPolicy&& p,
              Container&& c,
              IndexType icount,
              LoopBody&& loop_body)
{
  auto r = Res::get_default();
  return ::RAJA::policy_by_value_interface::forall_Icount(
      std::forward<ExecutionPolicy>(p), r, std::forward<Container>(c), icount,
      std::forward<LoopBody>(loop_body));
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers with a value-based policy
 *
 ******************************************************************************
 */

template<typename ExecutionPolicy,
         typename Res,
         typename Container,
         typename... Params>
RAJA_INLINE concepts::enable_if_t<
    resources::EventProxy<Res>,
    concepts::negate<type_traits::is_indexset_policy<ExecutionPolicy>>,
    concepts::negate<type_traits::is_multi_policy<ExecutionPolicy>>,
    type_traits::is_range<Container>>
forall(ExecutionPolicy&& p, Res r, Container&& c, Params&&... params)
{
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container does not model RandomAccessIterator");

  auto f_params = expt::make_forall_param_pack(std::forward<Params>(params)...);

  std::string kernel_name =
      expt::get_kernel_name(std::forward<Params>(params)...);
  auto&& loop_body = expt::get_lambda(std::forward<Params>(params)...);

  expt::check_forall_optional_args(loop_body, f_params);

  util::PluginContext context {
      util::make_context<camp::decay<ExecutionPolicy>>(std::move(kernel_name))};
  util::callPreCapturePlugins(context);

  using RAJA::util::trigger_updates_before;
  auto body = trigger_updates_before(loop_body);

  util::callPostCapturePlugins(context);

  util::callPreLaunchPlugins(context);

  resources::EventProxy<Res> e =
      wrap::forall(r, std::forward<ExecutionPolicy>(p),
                   std::forward<Container>(c), std::move(body), f_params);

  util::callPostLaunchPlugins(context);
  return e;
}

template<typename ExecutionPolicy,
         typename Container,
         typename LoopBody,
         typename Res = typename resources::get_resource<ExecutionPolicy>::type>
RAJA_INLINE concepts::enable_if_t<
    resources::EventProxy<Res>,
    concepts::negate<type_traits::is_indexset_policy<ExecutionPolicy>>,
    concepts::negate<type_traits::is_multi_policy<ExecutionPolicy>>,
    type_traits::is_range<Container>>
forall(ExecutionPolicy&& p, Container&& c, LoopBody&& loop_body)
{
  auto r = Res::get_default();
  return ::RAJA::policy_by_value_interface::forall(
      std::forward<ExecutionPolicy>(p), r, std::forward<Container>(c),
      std::forward<LoopBody>(loop_body));
}

}  // namespace policy_by_value_interface

/*!
 * \brief Conversion from template-based policy to value-based policy for forall
 *
 * this reduces implementation overhead and perfectly forwards all arguments
 */
template<typename ExecutionPolicy,
         typename... Args,
         typename Res = typename resources::get_resource<ExecutionPolicy>::type>
RAJA_INLINE resources::EventProxy<Res> forall(Args&&... args)
{
  Res r = Res::get_default();
  return ::RAJA::policy_by_value_interface::forall(ExecutionPolicy(), r,
                                                   std::forward<Args>(args)...);
}

template<typename ExecutionPolicy, typename Res, typename... Args>
RAJA_INLINE concepts::enable_if_t<resources::EventProxy<Res>,
                                  type_traits::is_resource<Res>>
forall(Res r, Args&&... args)
{
  return ::RAJA::policy_by_value_interface::forall(ExecutionPolicy(), r,
                                                   std::forward<Args>(args)...);
}

/*!
 * \brief Conversion from template-based policy to value-based policy for
 * forall_Icount
 *
 * this reduces implementation overhead and perfectly forwards all arguments
 */
template<typename ExecutionPolicy,
         typename... Args,
         typename Res = typename resources::get_resource<ExecutionPolicy>::type>
RAJA_INLINE resources::EventProxy<Res> forall_Icount(Args&&... args)
{
  Res r = Res::get_default();
  return ::RAJA::policy_by_value_interface::forall_Icount(
      ExecutionPolicy(), r, std::forward<Args>(args)...);
}

template<typename ExecutionPolicy, typename Res, typename... Args>
RAJA_INLINE concepts::enable_if_t<resources::EventProxy<Res>,
                                  type_traits::is_resource<Res>>
forall_Icount(Res r, Args&&... args)
{
  return ::RAJA::policy_by_value_interface::forall_Icount(
      ExecutionPolicy(), r, std::forward<Args>(args)...);
}

namespace detail
{

template<typename T,
         typename ExecutionPolicy,
         typename LoopBody,
         typename Res,
         typename ForallParams>
RAJA_INLINE camp::resources::EventProxy<Res> CallForall::operator()(
    T const& segment,
    ExecutionPolicy,
    LoopBody body,
    Res r,
    ForallParams f_params) const
{
  // this is only called inside a region, use impl
  using policy::sequential::forall_impl;
  RAJA_FORCEINLINE_RECURSIVE
  return forall_impl(r, ExecutionPolicy(), segment, body, f_params);
}

constexpr CallForallIcount::CallForallIcount(int s) : start(s) {}

template<typename T,
         typename ExecutionPolicy,
         typename LoopBody,
         typename Res,
         typename ForallParams>
RAJA_INLINE camp::resources::EventProxy<Res> CallForallIcount::operator()(
    T const& segment,
    ExecutionPolicy,
    LoopBody body,
    Res r,
    ForallParams f_params) const
{
  // go through wrap to unwrap icount
  return wrap::forall_Icount(r, ExecutionPolicy(), segment, start, body,
                             f_params);
}

}  // namespace detail

//
// Experimental support for dynamic policy selection
//
// Future directions:
// - Tuple of resources one for each platform
// - Returns a generic event proxy only if a resource is provided
//   avoids overhead of constructing a typed erased resource
//
template<camp::idx_t IDX, typename POLICY_LIST>
struct dynamic_helper
{
  template<typename SEGMENT, typename... PARAMS>
  static void invoke_forall(const int pol,
                            SEGMENT const& seg,
                            PARAMS&&... params)
  {
    if (IDX == pol)
    {
      using t_pol = typename camp::at<POLICY_LIST, camp::num<IDX>>::type;
      RAJA::forall<t_pol>(seg, params...);
      return;
    }
    dynamic_helper<IDX - 1, POLICY_LIST>::invoke_forall(pol, seg, params...);
  }

  template<typename SEGMENT, typename... PARAMS>
  static resources::EventProxy<resources::Resource> invoke_forall(
      RAJA::resources::Resource r,
      const int pol,
      SEGMENT const& seg,
      PARAMS&&... params)
  {

    using t_pol         = typename camp::at<POLICY_LIST, camp::num<IDX>>::type;
    using resource_type = typename resources::get_resource<t_pol>::type;

    if (IDX == pol)
    {
      RAJA::forall<t_pol>(r.get<resource_type>(), seg, params...);

      // Return a generic event proxy from r,
      // because forall returns a typed event proxy
      return {r};
    }

    return dynamic_helper<IDX - 1, POLICY_LIST>::invoke_forall(r, pol, seg,
                                                               params...);
  }
};

template<typename POLICY_LIST>
struct dynamic_helper<0, POLICY_LIST>
{
  template<typename SEGMENT, typename... PARAMS>
  static void invoke_forall(const int pol,
                            SEGMENT const& seg,
                            PARAMS&&... params)
  {
    if (0 == pol)
    {
      using t_pol = typename camp::at<POLICY_LIST, camp::num<0>>::type;
      RAJA::forall<t_pol>(seg, params...);
      return;
    }
    RAJA_ABORT_OR_THROW("Policy enum not supported ");
  }

  template<typename SEGMENT, typename... PARAMS>
  static resources::EventProxy<resources::Resource> invoke_forall(
      RAJA::resources::Resource r,
      const int pol,
      SEGMENT const& seg,
      PARAMS&&... params)
  {
    if (pol != 0) RAJA_ABORT_OR_THROW("Policy value out of range ");

    using t_pol         = typename camp::at<POLICY_LIST, camp::num<0>>::type;
    using resource_type = typename resources::get_resource<t_pol>::type;

    RAJA::forall<t_pol>(r.get<resource_type>(), seg, params...);

    // Return a generic event proxy from r,
    // because forall returns a typed event proxy
    return {r};
  }
};

template<typename POLICY_LIST, typename SEGMENT, typename... PARAMS>
void dynamic_forall(const int pol, SEGMENT const& seg, PARAMS&&... params)
{
  constexpr int N = camp::size<POLICY_LIST>::value;
  static_assert(N > 0, "RAJA policy list must not be empty");

  if (pol > N - 1)
  {
    RAJA_ABORT_OR_THROW("Policy enum not supported");
  }
  dynamic_helper<N - 1, POLICY_LIST>::invoke_forall(pol, seg, params...);
}

template<typename POLICY_LIST, typename SEGMENT, typename... PARAMS>
resources::EventProxy<resources::Resource> dynamic_forall(
    RAJA::resources::Resource r,
    const int pol,
    SEGMENT const& seg,
    PARAMS&&... params)
{
  constexpr int N = camp::size<POLICY_LIST>::value;
  static_assert(N > 0, "RAJA policy list must not be empty");

  if (pol > N - 1)
  {
    RAJA_ABORT_OR_THROW("Policy value out of range");
  }

  return dynamic_helper<N - 1, POLICY_LIST>::invoke_forall(r, pol, seg,
                                                           params...);
}


}  // namespace RAJA


#endif  // closing endif for header file include guard
