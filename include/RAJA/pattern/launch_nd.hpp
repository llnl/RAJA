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
#include <type_traits>
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

namespace detail
{

template<typename... IdxTs>
struct TypedRangeSegmentPack
{
  camp::tuple<RAJA::TypedRangeSegment<IdxTs>...> data;
};

template<typename T>
struct is_typed_range_segment_pack : std::false_type
{};

template<typename... IdxTs>
struct is_typed_range_segment_pack<TypedRangeSegmentPack<IdxTs...>>
    : std::true_type
{};

template<typename T>
inline constexpr bool is_typed_range_segment_pack_v =
    is_typed_range_segment_pack<T>::value;

template<typename T>
struct typed_range_segment_pack_rank;

template<typename... IdxTs>
struct typed_range_segment_pack_rank<TypedRangeSegmentPack<IdxTs...>>
    : std::integral_constant<camp::idx_t, sizeof...(IdxTs)>
{};

template<typename T>
inline constexpr camp::idx_t typed_range_segment_pack_rank_v =
    typed_range_segment_pack_rank<T>::value;

template<typename T>
concept typed_range_segment_pack =
    is_typed_range_segment_pack_v<std::decay_t<T>>;

}  // namespace detail

/*!
 * Create a segment pack for use with ``RAJA::launch_nd``.
 *
 * Currently supports packs of ``RAJA::TypedRangeSegment`` only.
 */
template<typename... IdxTs>
RAJA_INLINE auto nd_segments(RAJA::TypedRangeSegment<IdxTs> const&... segs)
{
  return detail::TypedRangeSegmentPack<IdxTs...> {camp::make_tuple(segs...)};
}

template<typename ExecPolicy, typename LayoutTag = RAJA::layout_right>
struct launch_nd_flattened_policy
{
  using exec_policy = ExecPolicy;
  using layout_tag  = LayoutTag;
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

/*!
 * Convenience alias for selecting between host/device exec policies for the
 * flattened launch_nd mapping at runtime via RAJA::ExecPlace.
 */
template<typename HostExecPolicy,
         typename DeviceExecPolicy,
         typename LayoutTag = RAJA::layout_right>
using launch_nd_flattened_place_policy = launch_nd_place_policy<
    launch_nd_flattened_policy<HostExecPolicy, LayoutTag>,
    launch_nd_flattened_policy<DeviceExecPolicy, LayoutTag>>;

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
RAJA_INLINE void validate_launch_nd_place_matches_resource(
    RAJA::resources::Resource const& resource,
    ExecPlace place)
{
  const bool resource_is_host =
      (resource.get_platform() == RAJA::Platform::host);

  if (place == ExecPlace::HOST && !resource_is_host)
  {
    RAJA_ABORT_OR_THROW(
        "RAJA::launch_nd: ExecPlace::HOST requires a host resource");
  }
  else if (place == ExecPlace::DEVICE && resource_is_host)
  {
    RAJA_ABORT_OR_THROW(
        "RAJA::launch_nd: ExecPlace::DEVICE requires a device resource");
  }
}
#else
RAJA_INLINE void validate_launch_nd_place_matches_resource(
    RAJA::resources::Resource const& resource,
    ExecPlace place)
{
  RAJA_UNUSED_VAR(resource);
  if (place == ExecPlace::DEVICE)
  {
    RAJA_ABORT_OR_THROW("RAJA::launch_nd: ExecPlace::DEVICE requested but "
                        "device is not enabled");
  }
}
#endif

#if defined(RAJA_GPU_ACTIVE)
template<typename PolicyList>
RAJA_INLINE auto make_launch_nd_context(RAJA::resources::Resource resource,
                                        std::string&& kernel_name)
{
  if (resource.get_platform() == RAJA::Platform::host)
  {
    return util::make_context<typename PolicyList::host_policy_t>(
        std::move(kernel_name));
  }
  return util::make_context<typename PolicyList::device_policy_t>(
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

template<typename LayoutTag,
         typename Lambda,
         typed_range_segment_pack SegmentPack>
RAJA_INLINE auto make_launch_nd_adapter(Lambda&& body, SegmentPack const& segs)
{
  constexpr camp::idx_t Rank =
      typed_range_segment_pack_rank_v<std::decay_t<SegmentPack>>;

  using perm = typename layout_to_permutation<LayoutTag, Rank>::type;

  return camp::apply(
      [&](auto const&... unpacked_segs) {
        return RAJA::make_PermutedCombiningAdapter<perm>(
            std::forward<Lambda>(body), unpacked_segs...);
      },
      segs.data);
}

template<typename Execute, typename... Params>
RAJA_INLINE auto launch_nd_with_plugins(util::PluginContext context,
                                        Execute&& execute,
                                        Params&&... params)
{
  auto f_params = expt::make_forall_param_pack(std::forward<Params>(params)...);
  std::string kernel_name =
      expt::get_kernel_name(std::forward<Params>(params)...);
  auto&& loop_body = expt::get_lambda(std::forward<Params>(params)...);
  expt::check_forall_optional_args(loop_body, f_params);

  context.kernel_name = std::move(kernel_name);
  util::callPreCapturePlugins(context);

  using RAJA::util::trigger_updates_before;
  auto body = trigger_updates_before(loop_body);

  util::callPostCapturePlugins(context);
  util::callPreLaunchPlugins(context);

  using result_type = decltype(std::forward<Execute>(execute)(std::move(body)));
  if constexpr (std::is_void<result_type>::value)
  {
    std::forward<Execute>(execute)(std::move(body));
    util::callPostLaunchPlugins(context);
  }
  else
  {
    auto event = std::forward<Execute>(execute)(std::move(body));
    util::callPostLaunchPlugins(context);
    return event;
  }
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

template<typename LoopPolicy, typename Adapter>
struct launch_nd_flattened_body
{
  Adapter adapter;

  RAJA_HOST_DEVICE RAJA_INLINE void operator()(RAJA::LaunchContext ctx) const
  {
    RAJA::loop<LoopPolicy>(ctx, adapter.getRange(), adapter);
  }
};

template<typename ExecPolicy,
         typename LayoutTag,
         typename Lambda,
         typed_range_segment_pack SegmentPack>
void launch_nd_flattened_execute(SegmentPack const& segs, Lambda&& body)
{
  auto adapter =
      make_launch_nd_adapter<LayoutTag>(std::forward<Lambda>(body), segs);

  using traits        = launch_nd_flattened_launch_traits<ExecPolicy>;
  using launch_policy = typename traits::launch_policy;
  using loop_policy   = typename traits::loop_policy;

  RAJA::launch<launch_policy>(
      traits::make_launch_params(adapter.size()),
      launch_nd_flattened_body<loop_policy, camp::decay<decltype(adapter)>> {
          std::move(adapter)});
}

template<typename ExecPolicy,
         typename LayoutTag,
         typename Lambda,
         typed_range_segment_pack SegmentPack>
resources::EventProxy<resources::Resource> launch_nd_flattened_execute(
    RAJA::resources::Resource resource,
    SegmentPack const& segs,
    Lambda&& body)
{
  auto adapter =
      make_launch_nd_adapter<LayoutTag>(std::forward<Lambda>(body), segs);

  using traits        = launch_nd_flattened_launch_traits<ExecPolicy>;
  using launch_policy = typename traits::launch_policy;
  using loop_policy   = typename traits::loop_policy;

  return RAJA::launch<launch_policy>(
      resource, traits::make_launch_params(adapter.size()),
      launch_nd_flattened_body<loop_policy, camp::decay<decltype(adapter)>> {
          std::move(adapter)});
}

template<typename LoopPolicyList,
         typename Body,
         typed_range_segment_pack SegmentPack>
struct launch_nd_grid_body
{
  SegmentPack segs;
  Body body;

  RAJA_HOST_DEVICE RAJA_INLINE void operator()(RAJA::LaunchContext ctx) const
  {
    exec_dim<0>(ctx);
  }

private:
  static constexpr camp::idx_t rank =
      typed_range_segment_pack_rank_v<std::decay_t<SegmentPack>>;

  template<camp::idx_t Dim, typename... IdxTs>
  RAJA_HOST_DEVICE RAJA_INLINE void exec_dim(RAJA::LaunchContext ctx,
                                             IdxTs... indices) const
  {
    if constexpr (Dim == rank)
    {
      body(indices...);
    }
    else
    {
      using loop_policy = typename camp::at<LoopPolicyList, camp::num<Dim>>::type;
      auto const seg = camp::get<Dim>(segs.data);

      RAJA::loop<loop_policy>(ctx, seg, [&](auto idx) {
        exec_dim<Dim + 1>(ctx, indices..., idx);
      });
    }
  }
};

template<typename LaunchPolicy,
         typename LoopPolicyList,
         typename Lambda,
         typed_range_segment_pack SegmentPack>
void launch_nd_grid_execute(LaunchParams const& launch_params,
                            SegmentPack const& segs,
                            Lambda&& body)
{
  RAJA::launch<LaunchPolicy>(
      launch_params,
      launch_nd_grid_body<LoopPolicyList, camp::decay<Lambda>, SegmentPack> {
          segs, std::forward<Lambda>(body)});
}

template<typename LaunchPolicy,
         typename LoopPolicyList,
         typename Lambda,
         typed_range_segment_pack SegmentPack>
resources::EventProxy<resources::Resource> launch_nd_grid_execute(
    RAJA::resources::Resource resource,
    LaunchParams const& launch_params,
    SegmentPack const& segs,
    Lambda&& body)
{
  return RAJA::launch<LaunchPolicy>(
      resource, launch_params,
      launch_nd_grid_body<LoopPolicyList, camp::decay<Lambda>, SegmentPack> {
          segs, std::forward<Lambda>(body)});
}

}  // namespace detail

template<typename ExecPolicy,
         typename LayoutTag,
         detail::typed_range_segment_pack SegmentPack,
         typename... Params>
void launch_nd(launch_nd_flattened_policy<ExecPolicy, LayoutTag>,
               SegmentPack const& segs,
               Params&&... params)
{
  detail::launch_nd_with_plugins(
      detail::make_launch_nd_context<ExecPolicy>(std::string {}),
      [&](auto&& body) {
        detail::launch_nd_flattened_execute<ExecPolicy, LayoutTag>(
            segs, std::forward<decltype(body)>(body));
      },
      std::forward<Params>(params)...);
}

template<typename ExecPolicy,
         typename LayoutTag,
         detail::typed_range_segment_pack SegmentPack,
         typename... Params>
resources::EventProxy<resources::Resource> launch_nd(
    RAJA::resources::Resource resource,
    launch_nd_flattened_policy<ExecPolicy, LayoutTag>,
    SegmentPack const& segs,
    Params&&... params)
{
  return detail::launch_nd_with_plugins(
      detail::make_launch_nd_context<ExecPolicy>(std::string {}),
      [&](auto&& body) {
        return detail::launch_nd_flattened_execute<ExecPolicy, LayoutTag>(
            resource, segs, std::forward<decltype(body)>(body));
      },
      std::forward<Params>(params)...);
}

template<typename LaunchPolicy,
         typename... LoopPolicies,
         detail::typed_range_segment_pack SegmentPack,
         typename... Params>
void launch_nd(launch_nd_grid_policy<LaunchPolicy, LoopPolicies...> policy,
               SegmentPack const& segs,
               Params&&... params)
{
  static_assert(
      detail::typed_range_segment_pack_rank_v<std::decay_t<SegmentPack>> ==
          static_cast<camp::idx_t>(sizeof...(LoopPolicies)),
      "RAJA::launch_nd launch backend requires one loop policy per "
      "segment");

  detail::launch_nd_with_plugins(
      detail::make_launch_nd_context<typename LaunchPolicy::host_policy_t>(
          std::string {}),
      [&](auto&& body) {
        detail::launch_nd_grid_execute<LaunchPolicy,
                                       camp::list<LoopPolicies...>>(
            policy.launch_params, segs, std::forward<decltype(body)>(body));
      },
      std::forward<Params>(params)...);
}

template<typename LaunchPolicy,
         typename... LoopPolicies,
         detail::typed_range_segment_pack SegmentPack,
         typename... Params>
resources::EventProxy<resources::Resource> launch_nd(
    RAJA::resources::Resource resource,
    launch_nd_grid_policy<LaunchPolicy, LoopPolicies...> policy,
    SegmentPack const& segs,
    Params&&... params)
{
  static_assert(
      detail::typed_range_segment_pack_rank_v<std::decay_t<SegmentPack>> ==
          static_cast<camp::idx_t>(sizeof...(LoopPolicies)),
      "RAJA::launch_nd launch backend requires one loop policy per "
      "segment");

#if defined(RAJA_GPU_ACTIVE)
  util::PluginContext context {
      detail::make_launch_nd_context<LaunchPolicy>(resource, std::string {})};
#else
  util::PluginContext context {
      detail::make_launch_nd_context<typename LaunchPolicy::host_policy_t>(
          std::string {})};
#endif

  return detail::launch_nd_with_plugins(
      std::move(context),
      [&](auto&& body) {
        return detail::launch_nd_grid_execute<LaunchPolicy,
                                              camp::list<LoopPolicies...>>(
            resource, policy.launch_params, segs,
            std::forward<decltype(body)>(body));
      },
      std::forward<Params>(params)...);
}

template<typename HostPolicy,
         typename DevicePolicy,
         detail::typed_range_segment_pack SegmentPack,
         typename... Params>
void launch_nd(ExecPlace place,
               launch_nd_place_policy<HostPolicy, DevicePolicy> const& policy,
               SegmentPack const& segs,
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
         detail::typed_range_segment_pack SegmentPack,
         typename... Params>
resources::EventProxy<resources::Resource> launch_nd(
    RAJA::resources::Resource resource,
    ExecPlace place,
    launch_nd_place_policy<HostPolicy, DevicePolicy> const& policy,
    SegmentPack const& segs,
    Params&&... params)
{
  detail::validate_launch_nd_place_matches_resource(resource, place);
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
