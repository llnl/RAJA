/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for N-D launch helpers.
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

#ifndef RAJA_pattern_launch_nd_HPP
#define RAJA_pattern_launch_nd_HPP

#include <cstddef>
#include <utility>

#include "RAJA/pattern/launch.hpp"
#include "RAJA/policy/sequential.hpp"
#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda.hpp"
#endif
#if defined(RAJA_ENABLE_HIP)
#include "RAJA/policy/hip.hpp"
#endif
#if defined(RAJA_ENABLE_SYCL)
#include "RAJA/policy/sycl.hpp"
#endif
#include "RAJA/util/CombiningAdapter.hpp"
#include "RAJA/util/View.hpp"

namespace RAJA
{

template<typename... IdxTs>
struct TypedRangeSegmentPack
{
  camp::tuple<RAJA::TypedRangeSegment<IdxTs>...> data;
};

template<typename... IdxTs>
RAJA_INLINE auto segments(RAJA::TypedRangeSegment<IdxTs> const&... segs)
{
  return TypedRangeSegmentPack<IdxTs...> {camp::make_tuple(segs...)};
}

template<typename ExecPolicy, typename LayoutTag = RAJA::layout_right>
struct launch_nd_flattened_policy
{
  using exec_policy = ExecPolicy;
  using layout_tag  = LayoutTag;
};

/*!
 * A flattened launch_nd policy that supports selecting between a host and
 * device exec policy at runtime via RAJA::ExecPlace.
 *
 * This is useful when the caller wants a single launch site but needs to choose
 * between e.g. seq_exec and cuda_exec/hip_exec based on runtime conditions.
 */
template<typename HostExecPolicy,
         typename DeviceExecPolicy,
         typename LayoutTag = RAJA::layout_right>
struct launch_nd_flattened_place_policy
{
  using host_exec_policy   = HostExecPolicy;
  using device_exec_policy = DeviceExecPolicy;
  using layout_tag         = LayoutTag;
};

/*!
 * General runtime host/device launch_nd policy container.
 *
 * This allows callers to select different launch_nd policy *kinds* for host and
 * device (e.g., a grid policy on host for nested loops and a flattened policy
 * on device for a 1D mapping).
 */
template<typename HostPolicy, typename DevicePolicy>
struct launch_nd_place_policy
{
  HostPolicy host;
  DevicePolicy device;
};

template<typename HostPolicy, typename DevicePolicy>
RAJA_INLINE launch_nd_place_policy<HostPolicy, DevicePolicy>
make_launch_nd_place_policy(HostPolicy host, DevicePolicy device)
{
  return {std::move(host), std::move(device)};
}

template<typename LaunchPolicy, typename... LoopPolicies>
struct launch_nd_grid_policy
{
  using launch_policy = LaunchPolicy;
  using loop_policies = camp::list<LoopPolicies...>;

  LaunchParams launch_params;

  explicit launch_nd_grid_policy(LaunchParams params) : launch_params(params)
  {
    static_assert(sizeof...(LoopPolicies) == 2 || sizeof...(LoopPolicies) == 3,
                  "RAJA::launch_nd launch backend supports 2D or 3D loops");
  }
};

namespace detail
{

template<typename Policy>
RAJA_INLINE auto make_launch_nd_context(std::string&& kernel_name)
{
  return util::make_context<Policy>(std::move(kernel_name));
}

#if defined(RAJA_GPU_ACTIVE)
template<typename PolicyList>
RAJA_INLINE auto make_launch_nd_context(RAJA::resources::Resource resource,
                                        std::string&& kernel_name)
{
  return resource.get_platform() == RAJA::Platform::host
             ? util::make_context<typename PolicyList::host_policy_t>(
                   std::move(kernel_name))
             : util::make_context<typename PolicyList::device_policy_t>(
                   std::move(kernel_name));
}
#endif

template<typename Seq>
struct reverse_idx_seq;

template<camp::idx_t... Idx>
struct reverse_idx_seq<camp::idx_seq<Idx...>>
{
  using type = camp::idx_seq<sizeof...(Idx) - 1U - Idx...>;
};

template<typename LayoutTag, camp::idx_t Rank>
struct layout_to_permutation;

template<camp::idx_t Rank>
struct layout_to_permutation<RAJA::layout_right, Rank>
{
  using type = camp::make_idx_seq_t<Rank>;
};

template<camp::idx_t Rank>
struct layout_to_permutation<RAJA::layout_left, Rank>
{
  using type = typename reverse_idx_seq<camp::make_idx_seq_t<Rank>>::type;
};

template<typename LayoutTag, typename Lambda, typename... IdxTs>
RAJA_INLINE auto make_launch_nd_adapter(
    Lambda&& body,
    TypedRangeSegmentPack<IdxTs...> const& segs)
{
  using perm =
      typename layout_to_permutation<LayoutTag, sizeof...(IdxTs)>::type;

  return camp::apply(
      [&](auto const&... unpacked_segs) {
        return RAJA::make_PermutedCombiningAdapter<perm>(
            std::forward<Lambda>(body), unpacked_segs...);
      },
      segs.data);
}

template<typename ExecPolicy>
struct launch_nd_flattened_launch_traits;

template<>
struct launch_nd_flattened_launch_traits<RAJA::seq_exec>
{
  using launch_policy = RAJA::LaunchPolicy<RAJA::seq_launch_t>;
  using loop_policy   = RAJA::LoopPolicy<RAJA::seq_exec>;

  template<typename SizeT>
  static LaunchParams make_launch_params(SizeT)
  {
    return LaunchParams {};
  }
};

#if defined(RAJA_CUDA_ACTIVE)
template<typename IterationMapping,
         typename IterationGetter,
         typename LaunchConcretizer,
         size_t BlocksPerSM,
         bool Async>
struct launch_nd_flattened_launch_traits<
    RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                           IterationGetter,
                                           LaunchConcretizer,
                                           BlocksPerSM,
                                           Async>>
{
  using launch_policy = RAJA::LaunchPolicy<RAJA::cuda_launch_t<Async>>;
  using loop_policy   = RAJA::LoopPolicy<RAJA::cuda_global_x_loop>;

  template<typename SizeT>
  static LaunchParams make_launch_params(SizeT size)
  {
    constexpr int block_size = IterationGetter::block_size;
    static_assert(block_size > 0,
                  "RAJA::launch_nd flattened launch requires an execution "
                  "policy with a fixed block size");

    constexpr int grid_size = IterationGetter::grid_size;
    const int teams =
        (grid_size == RAJA::named_usage::unspecified)
            ? RAJA_DIVIDE_CEILING_INT(static_cast<int>(size), block_size)
            : grid_size;

    return LaunchParams(RAJA::Teams(teams), RAJA::Threads(block_size));
  }
};
#endif

#if defined(RAJA_HIP_ACTIVE)
template<typename IterationMapping,
         typename IterationGetter,
         typename LaunchConcretizer,
         bool Async>
struct launch_nd_flattened_launch_traits<
    RAJA::policy::hip::
        hip_exec<IterationMapping, IterationGetter, LaunchConcretizer, Async>>
{
  using launch_policy = RAJA::LaunchPolicy<RAJA::hip_launch_t<Async>>;
  using loop_policy   = RAJA::LoopPolicy<RAJA::hip_global_x_loop>;

  template<typename SizeT>
  static LaunchParams make_launch_params(SizeT size)
  {
    constexpr int block_size = IterationGetter::block_size;
    static_assert(block_size > 0,
                  "RAJA::launch_nd flattened launch requires an execution "
                  "policy with a fixed block size");

    constexpr int grid_size = IterationGetter::grid_size;
    const int teams =
        (grid_size == RAJA::named_usage::unspecified)
            ? RAJA_DIVIDE_CEILING_INT(static_cast<int>(size), block_size)
            : grid_size;

    return LaunchParams(RAJA::Teams(teams), RAJA::Threads(block_size));
  }
};
#endif

#if defined(RAJA_SYCL_ACTIVE)
template<size_t BlockSize, bool Async>
struct launch_nd_flattened_launch_traits<
    RAJA::policy::sycl::sycl_exec<BlockSize, Async>>
{
  using launch_policy = RAJA::LaunchPolicy<RAJA::sycl_launch_t<Async>>;
  using loop_policy   = RAJA::LoopPolicy<RAJA::sycl_global_2<BlockSize>>;

  template<typename SizeT>
  static LaunchParams make_launch_params(SizeT size)
  {
    return LaunchParams(
        RAJA::Teams(RAJA_DIVIDE_CEILING_INT(static_cast<int>(size),
                                            static_cast<int>(BlockSize))),
        RAJA::Threads(static_cast<int>(BlockSize)));
  }
};
#endif

template<typename ExecPolicy,
         typename LayoutTag,
         typename Lambda,
         typename... IdxTs>
void launch_nd_flattened_execute(TypedRangeSegmentPack<IdxTs...> const& segs,
                                 Lambda&& body)
{
  auto adapter =
      make_launch_nd_adapter<LayoutTag>(std::forward<Lambda>(body), segs);

  using traits        = launch_nd_flattened_launch_traits<ExecPolicy>;
  using launch_policy = typename traits::launch_policy;
  using loop_policy   = typename traits::loop_policy;

  RAJA::launch<launch_policy>(traits::make_launch_params(adapter.size()),
                              [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
                                RAJA::loop<loop_policy>(ctx, adapter.getRange(),
                                                        adapter);
                              });
}

template<typename ExecPolicy,
         typename LayoutTag,
         typename Lambda,
         typename... IdxTs>
resources::EventProxy<resources::Resource> launch_nd_flattened_execute(
    RAJA::resources::Resource resource,
    TypedRangeSegmentPack<IdxTs...> const& segs,
    Lambda&& body)
{
  auto adapter =
      make_launch_nd_adapter<LayoutTag>(std::forward<Lambda>(body), segs);

  using traits        = launch_nd_flattened_launch_traits<ExecPolicy>;
  using launch_policy = typename traits::launch_policy;
  using loop_policy   = typename traits::loop_policy;

  return RAJA::launch<launch_policy>(
      resource, traits::make_launch_params(adapter.size()),
      [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
        RAJA::loop<loop_policy>(ctx, adapter.getRange(), adapter);
      });
}

template<typename LaunchPolicy,
         typename LoopPolicyList,
         typename Lambda,
         typename Idx0,
         typename Idx1>
void launch_nd_grid_execute(LaunchParams const& launch_params,
                            TypedRangeSegmentPack<Idx0, Idx1> const& segs,
                            Lambda&& body,
                            camp::num<2>)
{
  using loop0 = typename camp::at<LoopPolicyList, camp::num<0>>::type;
  using loop1 = typename camp::at<LoopPolicyList, camp::num<1>>::type;

  auto const seg0 = camp::get<0>(segs.data);
  auto const seg1 = camp::get<1>(segs.data);
  auto user_body  = std::forward<Lambda>(body);

  RAJA::launch<LaunchPolicy>(launch_params,
                             [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
                               RAJA::loop<loop0>(ctx, seg0, [&](Idx0 i0) {
                                 RAJA::loop<loop1>(ctx, seg1, [&](Idx1 i1) {
                                   user_body(i0, i1);
                                 });
                               });
                             });
}

template<typename LaunchPolicy,
         typename LoopPolicyList,
         typename Lambda,
         typename Idx0,
         typename Idx1,
         typename Idx2>
void launch_nd_grid_execute(LaunchParams const& launch_params,
                            TypedRangeSegmentPack<Idx0, Idx1, Idx2> const& segs,
                            Lambda&& body,
                            camp::num<3>)
{
  using loop0 = typename camp::at<LoopPolicyList, camp::num<0>>::type;
  using loop1 = typename camp::at<LoopPolicyList, camp::num<1>>::type;
  using loop2 = typename camp::at<LoopPolicyList, camp::num<2>>::type;

  auto const seg0 = camp::get<0>(segs.data);
  auto const seg1 = camp::get<1>(segs.data);
  auto const seg2 = camp::get<2>(segs.data);
  auto user_body  = std::forward<Lambda>(body);

  RAJA::launch<LaunchPolicy>(launch_params,
                             [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
                               RAJA::loop<loop0>(ctx, seg0, [&](Idx0 i0) {
                                 RAJA::loop<loop1>(ctx, seg1, [&](Idx1 i1) {
                                   RAJA::loop<loop2>(ctx, seg2, [&](Idx2 i2) {
                                     user_body(i0, i1, i2);
                                   });
                                 });
                               });
                             });
}

template<typename LaunchPolicy,
         typename LoopPolicyList,
         typename Lambda,
         typename Idx0,
         typename Idx1>
resources::EventProxy<resources::Resource> launch_nd_grid_execute(
    RAJA::resources::Resource resource,
    LaunchParams const& launch_params,
    TypedRangeSegmentPack<Idx0, Idx1> const& segs,
    Lambda&& body,
    camp::num<2>)
{
  using loop0 = typename camp::at<LoopPolicyList, camp::num<0>>::type;
  using loop1 = typename camp::at<LoopPolicyList, camp::num<1>>::type;

  auto const seg0 = camp::get<0>(segs.data);
  auto const seg1 = camp::get<1>(segs.data);
  auto user_body  = std::forward<Lambda>(body);

  return RAJA::launch<LaunchPolicy>(
      resource, launch_params, [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
        RAJA::loop<loop0>(ctx, seg0, [&](Idx0 i0) {
          RAJA::loop<loop1>(ctx, seg1, [&](Idx1 i1) {
            user_body(i0, i1);
          });
        });
      });
}

template<typename LaunchPolicy,
         typename LoopPolicyList,
         typename Lambda,
         typename Idx0,
         typename Idx1,
         typename Idx2>
resources::EventProxy<resources::Resource> launch_nd_grid_execute(
    RAJA::resources::Resource resource,
    LaunchParams const& launch_params,
    TypedRangeSegmentPack<Idx0, Idx1, Idx2> const& segs,
    Lambda&& body,
    camp::num<3>)
{
  using loop0 = typename camp::at<LoopPolicyList, camp::num<0>>::type;
  using loop1 = typename camp::at<LoopPolicyList, camp::num<1>>::type;
  using loop2 = typename camp::at<LoopPolicyList, camp::num<2>>::type;

  auto const seg0 = camp::get<0>(segs.data);
  auto const seg1 = camp::get<1>(segs.data);
  auto const seg2 = camp::get<2>(segs.data);
  auto user_body  = std::forward<Lambda>(body);

  return RAJA::launch<LaunchPolicy>(
      resource, launch_params, [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
        RAJA::loop<loop0>(ctx, seg0, [&](Idx0 i0) {
          RAJA::loop<loop1>(ctx, seg1, [&](Idx1 i1) {
            RAJA::loop<loop2>(ctx, seg2, [&](Idx2 i2) {
              user_body(i0, i1, i2);
            });
          });
        });
      });
}

}  // namespace detail

template<typename ExecPolicy,
         typename LayoutTag,
         typename... IdxTs,
         typename... Params>
void launch_nd(launch_nd_flattened_policy<ExecPolicy, LayoutTag>,
               TypedRangeSegmentPack<IdxTs...> const& segs,
               Params&&... params)
{
  auto f_params = expt::make_forall_param_pack(std::forward<Params>(params)...);
  std::string kernel_name =
      expt::get_kernel_name(std::forward<Params>(params)...);
  auto&& loop_body = expt::get_lambda(std::forward<Params>(params)...);
  expt::check_forall_optional_args(loop_body, f_params);

  util::PluginContext context {
      detail::make_launch_nd_context<ExecPolicy>(std::move(kernel_name))};
  util::callPreCapturePlugins(context);

  using RAJA::util::trigger_updates_before;
  auto body = trigger_updates_before(loop_body);

  util::callPostCapturePlugins(context);
  util::callPreLaunchPlugins(context);

  detail::launch_nd_flattened_execute<ExecPolicy, LayoutTag>(segs,
                                                             std::move(body));

  util::callPostLaunchPlugins(context);
}

template<typename ExecPolicy,
         typename LayoutTag,
         typename... IdxTs,
         typename... Params>
resources::EventProxy<resources::Resource> launch_nd(
    RAJA::resources::Resource resource,
    launch_nd_flattened_policy<ExecPolicy, LayoutTag>,
    TypedRangeSegmentPack<IdxTs...> const& segs,
    Params&&... params)
{
  auto f_params = expt::make_forall_param_pack(std::forward<Params>(params)...);
  std::string kernel_name =
      expt::get_kernel_name(std::forward<Params>(params)...);
  auto&& loop_body = expt::get_lambda(std::forward<Params>(params)...);
  expt::check_forall_optional_args(loop_body, f_params);

  util::PluginContext context {
      detail::make_launch_nd_context<ExecPolicy>(std::move(kernel_name))};
  util::callPreCapturePlugins(context);

  using RAJA::util::trigger_updates_before;
  auto body = trigger_updates_before(loop_body);

  util::callPostCapturePlugins(context);
  util::callPreLaunchPlugins(context);

  auto event = detail::launch_nd_flattened_execute<ExecPolicy, LayoutTag>(
      resource, segs, std::move(body));

  util::callPostLaunchPlugins(context);
  return event;
}

template<typename LaunchPolicy,
         typename... LoopPolicies,
         typename... IdxTs,
         typename... Params>
void launch_nd(launch_nd_grid_policy<LaunchPolicy, LoopPolicies...> policy,
               TypedRangeSegmentPack<IdxTs...> const& segs,
               Params&&... params)
{
  static_assert(sizeof...(IdxTs) == sizeof...(LoopPolicies),
                "RAJA::launch_nd launch backend requires one loop policy per "
                "segment");

  auto f_params = expt::make_forall_param_pack(std::forward<Params>(params)...);
  std::string kernel_name =
      expt::get_kernel_name(std::forward<Params>(params)...);
  auto&& loop_body = expt::get_lambda(std::forward<Params>(params)...);
  expt::check_forall_optional_args(loop_body, f_params);

  util::PluginContext context {
      detail::make_launch_nd_context<typename LaunchPolicy::host_policy_t>(
          std::move(kernel_name))};
  util::callPreCapturePlugins(context);

  using RAJA::util::trigger_updates_before;
  auto body = trigger_updates_before(loop_body);

  util::callPostCapturePlugins(context);
  util::callPreLaunchPlugins(context);

  detail::launch_nd_grid_execute<LaunchPolicy, camp::list<LoopPolicies...>>(
      policy.launch_params, segs, std::move(body),
      camp::num<sizeof...(IdxTs)> {});

  util::callPostLaunchPlugins(context);
}

template<typename LaunchPolicy,
         typename... LoopPolicies,
         typename... IdxTs,
         typename... Params>
resources::EventProxy<resources::Resource> launch_nd(
    RAJA::resources::Resource resource,
    launch_nd_grid_policy<LaunchPolicy, LoopPolicies...> policy,
    TypedRangeSegmentPack<IdxTs...> const& segs,
    Params&&... params)
{
  static_assert(sizeof...(IdxTs) == sizeof...(LoopPolicies),
                "RAJA::launch_nd launch backend requires one loop policy per "
                "segment");

  auto f_params = expt::make_forall_param_pack(std::forward<Params>(params)...);
  std::string kernel_name =
      expt::get_kernel_name(std::forward<Params>(params)...);
  auto&& loop_body = expt::get_lambda(std::forward<Params>(params)...);
  expt::check_forall_optional_args(loop_body, f_params);

#if defined(RAJA_GPU_ACTIVE)
  util::PluginContext context {detail::make_launch_nd_context<LaunchPolicy>(
      resource, std::move(kernel_name))};
#else
  util::PluginContext context {
      detail::make_launch_nd_context<typename LaunchPolicy::host_policy_t>(
          std::move(kernel_name))};
#endif
  util::callPreCapturePlugins(context);

  using RAJA::util::trigger_updates_before;
  auto body = trigger_updates_before(loop_body);

  util::callPostCapturePlugins(context);
  util::callPreLaunchPlugins(context);

  auto event =
      detail::launch_nd_grid_execute<LaunchPolicy, camp::list<LoopPolicies...>>(
          resource, policy.launch_params, segs, std::move(body),
          camp::num<sizeof...(IdxTs)> {});

  util::callPostLaunchPlugins(context);
  return event;
}

template<typename HostExecPolicy,
         typename DeviceExecPolicy,
         typename LayoutTag,
         typename... IdxTs,
         typename... Params>
void launch_nd(ExecPlace place,
               launch_nd_flattened_place_policy<HostExecPolicy,
                                                DeviceExecPolicy,
                                                LayoutTag>,
               TypedRangeSegmentPack<IdxTs...> const& segs,
               Params&&... params)
{
  switch (place)
  {
    case ExecPlace::HOST:
      RAJA::launch_nd(launch_nd_flattened_policy<HostExecPolicy, LayoutTag> {},
                      segs, std::forward<Params>(params)...);
      break;
#if defined(RAJA_GPU_ACTIVE)
    case ExecPlace::DEVICE:
      RAJA::launch_nd(
          launch_nd_flattened_policy<DeviceExecPolicy, LayoutTag> {}, segs,
          std::forward<Params>(params)...);
      break;
#endif
    default:
      RAJA_ABORT_OR_THROW("Unknown launch place or device is not enabled");
  }
}

template<typename HostExecPolicy,
         typename DeviceExecPolicy,
         typename LayoutTag,
         typename... IdxTs,
         typename... Params>
resources::EventProxy<resources::Resource> launch_nd(
    RAJA::resources::Resource resource,
    ExecPlace place,
    launch_nd_flattened_place_policy<HostExecPolicy,
                                     DeviceExecPolicy,
                                     LayoutTag>,
    TypedRangeSegmentPack<IdxTs...> const& segs,
    Params&&... params)
{
  switch (place)
  {
    case ExecPlace::HOST:
      return RAJA::launch_nd(
          resource, launch_nd_flattened_policy<HostExecPolicy, LayoutTag> {},
          segs, std::forward<Params>(params)...);
#if defined(RAJA_GPU_ACTIVE)
    case ExecPlace::DEVICE:
      return RAJA::launch_nd(
          resource, launch_nd_flattened_policy<DeviceExecPolicy, LayoutTag> {},
          segs, std::forward<Params>(params)...);
#endif
    default:
      RAJA_ABORT_OR_THROW("Unknown launch place or device is not enabled");
  }
  return resources::EventProxy<resources::Resource>(resource);
}

template<typename HostPolicy,
         typename DevicePolicy,
         typename... IdxTs,
         typename... Params>
void launch_nd(ExecPlace place,
               launch_nd_place_policy<HostPolicy, DevicePolicy> const& policy,
               TypedRangeSegmentPack<IdxTs...> const& segs,
               Params&&... params)
{
  switch (place)
  {
    case ExecPlace::HOST:
      RAJA::launch_nd(policy.host, segs, std::forward<Params>(params)...);
      break;
#if defined(RAJA_GPU_ACTIVE)
    case ExecPlace::DEVICE:
      RAJA::launch_nd(policy.device, segs, std::forward<Params>(params)...);
      break;
#endif
    default:
      RAJA_ABORT_OR_THROW("Unknown launch place or device is not enabled");
  }
}

template<typename HostPolicy,
         typename DevicePolicy,
         typename... IdxTs,
         typename... Params>
resources::EventProxy<resources::Resource> launch_nd(
    RAJA::resources::Resource resource,
    ExecPlace place,
    launch_nd_place_policy<HostPolicy, DevicePolicy> const& policy,
    TypedRangeSegmentPack<IdxTs...> const& segs,
    Params&&... params)
{
  switch (place)
  {
    case ExecPlace::HOST:
      return RAJA::launch_nd(resource, policy.host, segs,
                             std::forward<Params>(params)...);
#if defined(RAJA_GPU_ACTIVE)
    case ExecPlace::DEVICE:
      return RAJA::launch_nd(resource, policy.device, segs,
                             std::forward<Params>(params)...);
#endif
    default:
      RAJA_ABORT_OR_THROW("Unknown launch place or device is not enabled");
  }
  return resources::EventProxy<resources::Resource>(resource);
}

}  // namespace RAJA

#endif /* RAJA_pattern_launch_nd_HPP */
