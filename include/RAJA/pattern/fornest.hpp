/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing user interface for RAJA::fornest
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

#ifndef RAJA_pattern_fornest_HPP
#define RAJA_pattern_fornest_HPP

#include "RAJA/config.hpp"

#include <cstddef>
#include <cmath>
#include <iterator>
#include <string>
#include <type_traits>
#include <utility>

#include "camp/camp.hpp"
#include "camp/list.hpp"

#include "RAJA/internal/get_platform.hpp"

#include "RAJA/pattern/forall.hpp"
#include "RAJA/pattern/launch.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/resource.hpp"
#include "RAJA/policy/device.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda.hpp"
#endif
#if defined(RAJA_ENABLE_HIP)
#include "RAJA/policy/hip.hpp"
#endif
#if defined(RAJA_ENABLE_SYCL)
#include "RAJA/policy/sycl.hpp"
#endif

namespace RAJA
{

/*!
 * Policy wrapper that requests a flattened mapping for a rank-2 or rank-3
 * fornest.
 *
 * With a flattened policy, `RAJA::fornest` linearizes the 2D/3D iteration
 * space and executes a single 1D `RAJA::forall` over that linear index.
 * The `LayoutTag` controls how the linear index is mapped back to `(i0,i1)`
 * or `(i0,i1,i2)` (e.g., `RAJA::layout_right` vs `RAJA::layout_left`).
 *
 * Notes:
 * - The provided segments must support random-access iteration since the
 *   implementation computes `*(seg.begin() + offset)` to recover the per-dim
 *   index values.
 * - Optional `forall` parameters (e.g., `RAJA::kernel_name`, reducers) are
 *   supported and forwarded to the underlying `RAJA::forall`.
 */
template<typename ExecPolicy, typename LayoutTag = RAJA::layout_right>
struct fornest_flattened_policy
{
  using exec_policy = ExecPolicy;
  using layout_tag  = LayoutTag;
};

/*!
 * Policy wrapper that requests an explicit per-dimension mapping for a rank-2
 * or rank-3 fornest.
 *
 * With a mapping policy, `RAJA::fornest` builds a nested `RAJA::launch` region
 * and applies one `RAJA::loop` per dimension using the provided loop policies.
 * This is the interface used to request specific CUDA/HIP mappings such as
 * mapping one dimension to `thread_x` and another to `block_y`, etc.
 *
 * `LoopPolicies...` must be `RAJA::LoopPolicy<HostPolicy[, DevicePolicy]>`
 * types (the optional device policy parameter is available when GPU support
 * is enabled). Examples:
 * - `RAJA::LoopPolicy<RAJA::seq_exec>` (portable host-only nested loops)
 * - `RAJA::LoopPolicy<RAJA::seq_exec, RAJA::device_thread_x_direct>` (CUDA/HIP)
 */
template<typename ExecPolicy, typename... LoopPolicies>
struct fornest_mapping_policy
{
  using exec_policy                 = ExecPolicy;
  using loop_policies               = camp::list<LoopPolicies...>;
  static constexpr camp::idx_t rank = sizeof...(LoopPolicies);
};

namespace type_traits
{

template<typename Policy>
struct policy_dimensionality : std::integral_constant<camp::idx_t, 0>
{};

#if defined(RAJA_CUDA_ACTIVE)
template<typename IterationMapping,
         kernel_sync_requirement sync,
         typename... IterationGetters>
struct policy_dimensionality<
    ::RAJA::policy::cuda::
        cuda_indexer<IterationMapping, sync, IterationGetters...>>
    : std::integral_constant<camp::idx_t, sizeof...(IterationGetters)>
{};
#endif

#if defined(RAJA_HIP_ACTIVE)
template<typename IterationMapping,
         kernel_sync_requirement sync,
         typename... IterationGetters>
struct policy_dimensionality<
    ::RAJA::policy::hip::
        hip_indexer<IterationMapping, sync, IterationGetters...>>
    : std::integral_constant<camp::idx_t, sizeof...(IterationGetters)>
{};
#endif

template<typename Policy>
inline constexpr camp::idx_t policy_dimensionality_v =
    policy_dimensionality<camp::decay<Policy>>::value;

}  // namespace type_traits

namespace detail
{

template<typename T>
struct is_camp_list : std::false_type
{};

template<typename... Ts>
struct is_camp_list<camp::list<Ts...>> : std::true_type
{};

template<typename T>
inline constexpr bool is_camp_list_v = is_camp_list<camp::decay<T>>::value;

template<typename Policy>
struct fornest_underlying_exec_policy
{
  using type = camp::decay<Policy>;
};

template<typename ExecPolicy, typename LayoutTag>
struct fornest_underlying_exec_policy<
    ::RAJA::fornest_flattened_policy<ExecPolicy, LayoutTag>>
{
  using type = camp::decay<ExecPolicy>;
};

template<typename ExecPolicy, typename... LoopPolicies>
struct fornest_underlying_exec_policy<
    ::RAJA::fornest_mapping_policy<ExecPolicy, LoopPolicies...>>
{
  using type = camp::decay<ExecPolicy>;
};

template<typename Policy>
using fornest_underlying_exec_policy_t =
    typename fornest_underlying_exec_policy<camp::decay<Policy>>::type;

template<typename Policy>
RAJA_INLINE auto require_typed_resource(RAJA::resources::Resource& r)
{
  using exec_pol      = fornest_underlying_exec_policy_t<Policy>;
  using resource_type = typename resources::get_resource<exec_pol>::type;
  return r.template get<resource_type>();
}

template<typename Seg>
RAJA_INLINE auto segment_length_host(Seg const& seg) -> std::size_t
{
  if constexpr (requires { seg.size(); })
  {
    return static_cast<std::size_t>(seg.size());
  }
  else
  {
    using std::begin;
    using std::end;
    using It = decltype(begin(seg));
    if constexpr (std::is_base_of<std::random_access_iterator_tag,
                                  typename std::iterator_traits<
                                      It>::iterator_category>::value)
    {
      return static_cast<std::size_t>(end(seg) - begin(seg));
    }
    else
    {
      return static_cast<std::size_t>(std::distance(begin(seg), end(seg)));
    }
  }
}

template<typename Seg>
concept segment_like = requires(Seg const& seg) {
  std::begin(seg);
  std::end(seg);
};

template<typename... Ts>
struct first_arg_is_segment_like : std::false_type
{};

template<typename T0, typename... Ts>
struct first_arg_is_segment_like<T0, Ts...>
    : std::bool_constant<segment_like<camp::decay<T0>>>
{};

template<typename... Ts>
inline constexpr bool first_arg_is_segment_like_v =
    first_arg_is_segment_like<Ts...>::value;

template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T ceil_div(T n, T d)
{
  return static_cast<T>(RAJA_DIVIDE_CEILING_INT(n, d));
}

template<typename Tuple, typename F, camp::idx_t... Seq>
RAJA_INLINE decltype(auto) apply_tuple_prefix(F&& f,
                                              Tuple&& tuple,
                                              camp::idx_seq<Seq...>)
{
  return std::forward<F>(f)(camp::get<Seq>(std::forward<Tuple>(tuple))...);
}

template<typename F, typename... Ts>
RAJA_INLINE decltype(auto) apply_without_last(F&& f, camp::tuple<Ts...>&& tuple)
{
  static_assert(sizeof...(Ts) > 0,
                "apply_without_last requires a non-empty tuple");
  return apply_tuple_prefix(std::forward<F>(f), std::move(tuple),
                            camp::make_idx_seq_t<sizeof...(Ts) - 1> {});
}

struct tile3
{
  int x = 1;
  int y = 1;
  int z = 1;
};

RAJA_INLINE tile3 choose_2d_tile(int block_size, std::size_t n0, std::size_t n1)
{
  tile3 best {};
  std::size_t best_padded = static_cast<std::size_t>(-1);
  double best_aspect_cost = 1e300;
  int best_tx_warp_score  = -1;
  int best_tx             = 1;

  for (int tx = 1; tx <= block_size; ++tx)
  {
    if (block_size % tx != 0) continue;
    const int ty = block_size / tx;

    const std::size_t padded0 =
        ceil_div<std::size_t>(n0, static_cast<std::size_t>(tx)) *
        static_cast<std::size_t>(tx);
    const std::size_t padded1 =
        ceil_div<std::size_t>(n1, static_cast<std::size_t>(ty)) *
        static_cast<std::size_t>(ty);
    const std::size_t padded = padded0 * padded1;

    const double aspect = static_cast<double>(tx) / static_cast<double>(ty);
    const double aspect_cost = std::abs(aspect - 4.0);  // prefer 4:1 (32x8)
    const int tx_warp_score  = (tx % 32 == 0) ? 1 : 0;

    const bool better =
        (padded < best_padded) ||
        (padded == best_padded && aspect_cost < best_aspect_cost) ||
        (padded == best_padded && aspect_cost == best_aspect_cost &&
         tx_warp_score > best_tx_warp_score) ||
        (padded == best_padded && aspect_cost == best_aspect_cost &&
         tx_warp_score == best_tx_warp_score && tx > best_tx);

    if (better)
    {
      best_padded        = padded;
      best_aspect_cost   = aspect_cost;
      best_tx_warp_score = tx_warp_score;
      best_tx            = tx;
      best.x             = tx;
      best.y             = ty;
      best.z             = 1;
    }
  }

  return best;
}

RAJA_INLINE tile3 choose_3d_tile(int block_size,
                                 std::size_t n0,
                                 std::size_t n1,
                                 std::size_t n2)
{
  tile3 best {};
  std::size_t best_padded = static_cast<std::size_t>(-1);
  double best_aspect_cost = 1e300;
  int best_tz_bias        = -1;

  for (int tz = 1; tz <= block_size; ++tz)
  {
    if (block_size % tz != 0) continue;
    const int rem  = block_size / tz;
    const tile3 xy = choose_2d_tile(rem, n0, n1);

    const std::size_t padded0 =
        ceil_div<std::size_t>(n0, static_cast<std::size_t>(xy.x)) *
        static_cast<std::size_t>(xy.x);
    const std::size_t padded1 =
        ceil_div<std::size_t>(n1, static_cast<std::size_t>(xy.y)) *
        static_cast<std::size_t>(xy.y);
    const std::size_t padded2 =
        ceil_div<std::size_t>(n2, static_cast<std::size_t>(tz)) *
        static_cast<std::size_t>(tz);
    const std::size_t padded = padded0 * padded1 * padded2;

    const double aspect = static_cast<double>(xy.x) / static_cast<double>(xy.y);
    const double aspect_cost = std::abs(aspect - 4.0);

    const int tz_bias = (n2 <= 1) ? -tz : tz;

    const bool better =
        (padded < best_padded) ||
        (padded == best_padded && aspect_cost < best_aspect_cost) ||
        (padded == best_padded && aspect_cost == best_aspect_cost &&
         tz_bias > best_tz_bias);

    if (better)
    {
      best_padded      = padded;
      best_aspect_cost = aspect_cost;
      best_tz_bias     = tz_bias;
      best.x           = xy.x;
      best.y           = xy.y;
      best.z           = tz;
    }
  }

  return best;
}

template<typename LayoutTag>
RAJA_HOST_DEVICE RAJA_INLINE void linear_to_2d(std::size_t lin,
                                               std::size_t n0,
                                               std::size_t n1,
                                               std::size_t& i0,
                                               std::size_t& i1)
{
  if constexpr (std::is_same<LayoutTag, RAJA::layout_left>::value)
  {
    i0 = lin % n0;
    i1 = lin / n0;
    RAJA_UNUSED_VAR(n1);
  }
  else
  {
    i0 = lin / n1;
    i1 = lin % n1;
    RAJA_UNUSED_VAR(n0);
  }
}

template<typename LayoutTag>
RAJA_HOST_DEVICE RAJA_INLINE void linear_to_3d(std::size_t lin,
                                               std::size_t n0,
                                               std::size_t n1,
                                               std::size_t n2,
                                               std::size_t& i0,
                                               std::size_t& i1,
                                               std::size_t& i2)
{
  if constexpr (std::is_same<LayoutTag, RAJA::layout_left>::value)
  {
    i0 = lin % n0;
    lin /= n0;
    i1 = lin % n1;
    i2 = lin / n1;
    RAJA_UNUSED_VAR(n2);
  }
  else
  {
    i0 = lin / (n1 * n2);
    lin -= i0 * (n1 * n2);
    i1 = lin / n2;
    i2 = lin % n2;
    RAJA_UNUSED_VAR(n0);
  }
}

template<typename LayoutTag, typename Seg0, typename Seg1, typename Body>
struct FornestFlattenedForallBody2D
{
  using seg0_type  = camp::decay<Seg0>;
  using seg1_type  = camp::decay<Seg1>;
  using body_type  = camp::decay<Body>;
  using index_type = std::size_t;

  index_type n0 = 0;
  index_type n1 = 0;
  seg0_type seg0;
  seg1_type seg1;
  body_type body;

  template<typename B>
  RAJA_INLINE FornestFlattenedForallBody2D(index_type n0_,
                                           index_type n1_,
                                           Seg0 const& seg0_,
                                           Seg1 const& seg1_,
                                           B&& body_)
      : n0(n0_),
        n1(n1_),
        seg0(seg0_),
        seg1(seg1_),
        body(std::forward<B>(body_))
  {}

  RAJA_SUPPRESS_HD_WARN
  template<typename... Reducers>
  RAJA_HOST_DEVICE RAJA_INLINE void operator()(index_type lin,
                                               Reducers&... reducers) const
  {
    index_type i0 = 0;
    index_type i1 = 0;
    detail::linear_to_2d<LayoutTag>(lin, n0, n1, i0, i1);

    using It0 = decltype(seg0.begin());
    using It1 = decltype(seg1.begin());
    using D0  = typename std::iterator_traits<It0>::difference_type;
    using D1  = typename std::iterator_traits<It1>::difference_type;

    auto v0 = *(seg0.begin() + static_cast<D0>(i0));
    auto v1 = *(seg1.begin() + static_cast<D1>(i1));
    body(v0, v1, reducers...);
  }
};

template<typename LayoutTag,
         typename Seg0,
         typename Seg1,
         typename Seg2,
         typename Body>
struct FornestFlattenedForallBody3D
{
  using seg0_type  = camp::decay<Seg0>;
  using seg1_type  = camp::decay<Seg1>;
  using seg2_type  = camp::decay<Seg2>;
  using body_type  = camp::decay<Body>;
  using index_type = std::size_t;

  index_type n0 = 0;
  index_type n1 = 0;
  index_type n2 = 0;
  seg0_type seg0;
  seg1_type seg1;
  seg2_type seg2;
  body_type body;

  template<typename B>
  RAJA_INLINE FornestFlattenedForallBody3D(index_type n0_,
                                           index_type n1_,
                                           index_type n2_,
                                           Seg0 const& seg0_,
                                           Seg1 const& seg1_,
                                           Seg2 const& seg2_,
                                           B&& body_)
      : n0(n0_),
        n1(n1_),
        n2(n2_),
        seg0(seg0_),
        seg1(seg1_),
        seg2(seg2_),
        body(std::forward<B>(body_))
  {}

  RAJA_SUPPRESS_HD_WARN
  template<typename... Reducers>
  RAJA_HOST_DEVICE RAJA_INLINE void operator()(index_type lin,
                                               Reducers&... reducers) const
  {
    index_type i0 = 0;
    index_type i1 = 0;
    index_type i2 = 0;
    detail::linear_to_3d<LayoutTag>(lin, n0, n1, n2, i0, i1, i2);

    using It0 = decltype(seg0.begin());
    using It1 = decltype(seg1.begin());
    using It2 = decltype(seg2.begin());
    using D0  = typename std::iterator_traits<It0>::difference_type;
    using D1  = typename std::iterator_traits<It1>::difference_type;
    using D2  = typename std::iterator_traits<It2>::difference_type;

    auto v0 = *(seg0.begin() + static_cast<D0>(i0));
    auto v1 = *(seg1.begin() + static_cast<D1>(i1));
    auto v2 = *(seg2.begin() + static_cast<D2>(i2));
    body(v0, v1, v2, reducers...);
  }
};

template<typename LaunchContext,
         typename Loop0,
         typename Loop1,
         typename Seg0,
         typename Seg1,
         typename Body>
struct FornestLaunchBody2D
{
  using seg0_type    = camp::decay<Seg0>;
  using seg1_type    = camp::decay<Seg1>;
  using body_type    = camp::decay<Body>;
  using context_type = camp::decay<LaunchContext>;

  seg0_type seg0;
  seg1_type seg1;
  body_type body;

  template<typename B>
  RAJA_INLINE FornestLaunchBody2D(Seg0 const& seg0_,
                                  Seg1 const& seg1_,
                                  B&& body_)
      : seg0(seg0_),
        seg1(seg1_),
        body(std::forward<B>(body_))
  {}

  RAJA_SUPPRESS_HD_WARN
  template<typename... Reducers>
  RAJA_HOST_DEVICE RAJA_INLINE void operator()(context_type ctx,
                                               Reducers&... reducers) const
  {
    RAJA::loop<Loop0>(ctx, seg0, [&](auto i0) {
      RAJA::loop<Loop1>(ctx, seg1, [&](auto i1) {
        body(i0, i1, reducers...);
      });
    });
  }
};

// NOTE:
// launch_context_type relies on taking the address of BODY::operator() to
// deduce the launch context type. Since FornestLaunchBody* define operator()
// as a function template (to accept an arbitrary reducer pack), that trait
// cannot deduce the first argument type and falls back to a host context.
// This breaks mixed host/device launch policies (e.g. seq + CUDA/HIP).
// Provide explicit specializations so LaunchExecute selects the correct
// context type.
template<typename LaunchContext,
         typename Loop0,
         typename Loop1,
         typename Seg0,
         typename Seg1,
         typename Body>
struct launch_context_type<
    FornestLaunchBody2D<LaunchContext, Loop0, Loop1, Seg0, Seg1, Body>,
    void>
{
  using type = camp::decay<LaunchContext>;
};

template<typename LaunchContext,
         typename Loop0,
         typename Loop1,
         typename Loop2,
         typename Seg0,
         typename Seg1,
         typename Seg2,
         typename Body>
struct FornestLaunchBody3D
{
  using seg0_type    = camp::decay<Seg0>;
  using seg1_type    = camp::decay<Seg1>;
  using seg2_type    = camp::decay<Seg2>;
  using body_type    = camp::decay<Body>;
  using context_type = camp::decay<LaunchContext>;

  seg0_type seg0;
  seg1_type seg1;
  seg2_type seg2;
  body_type body;

  template<typename B>
  RAJA_INLINE FornestLaunchBody3D(Seg0 const& seg0_,
                                  Seg1 const& seg1_,
                                  Seg2 const& seg2_,
                                  B&& body_)
      : seg0(seg0_),
        seg1(seg1_),
        seg2(seg2_),
        body(std::forward<B>(body_))
  {}

  RAJA_SUPPRESS_HD_WARN
  template<typename... Reducers>
  RAJA_HOST_DEVICE RAJA_INLINE void operator()(context_type ctx,
                                               Reducers&... reducers) const
  {
    RAJA::loop<Loop0>(ctx, seg0, [&](auto i0) {
      RAJA::loop<Loop1>(ctx, seg1, [&](auto i1) {
        RAJA::loop<Loop2>(ctx, seg2, [&](auto i2) {
          body(i0, i1, i2, reducers...);
        });
      });
    });
  }
};

template<typename LaunchContext,
         typename Loop0,
         typename Loop1,
         typename Loop2,
         typename Seg0,
         typename Seg1,
         typename Seg2,
         typename Body>
struct launch_context_type<FornestLaunchBody3D<LaunchContext,
                                               Loop0,
                                               Loop1,
                                               Loop2,
                                               Seg0,
                                               Seg1,
                                               Seg2,
                                               Body>,
                           void>
{
  using type = camp::decay<LaunchContext>;
};

template<typename ExecPolicy>
struct exec_block_size
{
  static constexpr int value = 1;
};

#if defined(RAJA_CUDA_ACTIVE)
template<typename IterationMapping,
         typename IterationGetter,
         typename LaunchConcretizer,
         size_t BlocksPerSM,
         bool Async>
struct exec_block_size<
    ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                             IterationGetter,
                                             LaunchConcretizer,
                                             BlocksPerSM,
                                             Async>>
{
  static constexpr int value = IterationGetter::block_size;
};
#endif

#if defined(RAJA_HIP_ACTIVE)
template<typename IterationMapping,
         typename IterationGetter,
         typename LaunchConcretizer,
         bool Async>
struct exec_block_size<
    ::RAJA::policy::hip::
        hip_exec<IterationMapping, IterationGetter, LaunchConcretizer, Async>>
{
  static constexpr int value = IterationGetter::block_size;
};
#endif

template<typename ExecPolicy>
inline constexpr int exec_block_size_v =
    exec_block_size<camp::decay<ExecPolicy>>::value;

#if defined(RAJA_GPU_ACTIVE)
template<typename ExecPolicy>
struct fornest_launch_policy;

template<>
struct fornest_launch_policy<RAJA::seq_exec>
{
  using type = RAJA::LaunchPolicy<RAJA::seq_launch_t>;
};

#if defined(RAJA_CUDA_ACTIVE)
template<typename IterationMapping,
         typename IterationGetter,
         typename LaunchConcretizer,
         size_t BlocksPerSM,
         bool Async>
struct fornest_launch_policy<
    ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                             IterationGetter,
                                             LaunchConcretizer,
                                             BlocksPerSM,
                                             Async>>
{
  using type =
      RAJA::LaunchPolicy<RAJA::seq_launch_t, RAJA::device_launch_t<Async>>;
};
#endif

#if defined(RAJA_HIP_ACTIVE)
template<typename IterationMapping,
         typename IterationGetter,
         typename LaunchConcretizer,
         bool Async>
struct fornest_launch_policy<
    ::RAJA::policy::hip::
        hip_exec<IterationMapping, IterationGetter, LaunchConcretizer, Async>>
{
  using type =
      RAJA::LaunchPolicy<RAJA::seq_launch_t, RAJA::device_launch_t<Async>>;
};
#endif

#if defined(RAJA_SYCL_ACTIVE)
template<size_t BlockSize, bool Async>
struct fornest_launch_policy<::RAJA::policy::sycl::sycl_exec<BlockSize, Async>>
{
  // sycl_launch_t is specialized only for num_threads=0, and uses LaunchParams
  // for the actual threads/teams configuration.
  using type =
      RAJA::LaunchPolicy<RAJA::seq_launch_t, RAJA::sycl_launch_t<Async, 0>>;
};
#endif

template<typename ExecPolicy>
using fornest_launch_policy_t =
    typename fornest_launch_policy<camp::decay<ExecPolicy>>::type;
#endif

enum class fornest_map_space
{
  seq,
  global,
  block,
  thread
};

enum class fornest_axis
{
  none,
  x,
  y,
  z
};

// Extract IndexGlobal (dimension + sizes) from a policy::cuda/hip *_indexer
// with a single indexer. This supports using sized mapping policies like
// `device_global_size_x_direct<32>` to fully determine `Threads/Teams`.
template<typename Policy>
struct fornest_indexglobal_info
{
  static constexpr bool valid        = false;
  static constexpr fornest_axis axis = fornest_axis::none;
  static constexpr int block_size = -1;  // threads in that axis (if applicable)
  static constexpr int grid_size  = -1;  // blocks in that axis (if applicable)
};

#if defined(RAJA_CUDA_ACTIVE)
template<typename IterationMapping,
         kernel_sync_requirement sync,
         typename Indexer>
struct fornest_indexglobal_info<
    ::RAJA::policy::cuda::cuda_indexer<IterationMapping, sync, Indexer>>
{
  static constexpr bool valid        = false;
  static constexpr fornest_axis axis = fornest_axis::none;
  static constexpr int block_size    = -1;
  static constexpr int grid_size     = -1;
};

template<typename IterationMapping,
         kernel_sync_requirement sync,
         named_dim dim,
         size_t BLOCK_SIZE,
         size_t GRID_SIZE>
struct fornest_indexglobal_info<::RAJA::policy::cuda::cuda_indexer<
    IterationMapping,
    sync,
    ::RAJA::cuda::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>>>
{
  static constexpr bool valid        = true;
  static constexpr fornest_axis axis = (dim == named_dim::x)   ? fornest_axis::x
                                       : (dim == named_dim::y) ? fornest_axis::y
                                       : (dim == named_dim::z)
                                           ? fornest_axis::z
                                           : fornest_axis::none;
  static constexpr std::size_t unspecified_sz =
      static_cast<std::size_t>(named_usage::unspecified);
  static constexpr std::size_t ignored_sz =
      static_cast<std::size_t>(named_usage::ignored);
  static constexpr int block_size =
      (BLOCK_SIZE == unspecified_sz || BLOCK_SIZE == ignored_sz)
          ? -1
          : static_cast<int>(BLOCK_SIZE);
  static constexpr int grid_size =
      (GRID_SIZE == unspecified_sz || GRID_SIZE == ignored_sz)
          ? -1
          : static_cast<int>(GRID_SIZE);
};
#endif

#if defined(RAJA_HIP_ACTIVE)
template<typename IterationMapping,
         kernel_sync_requirement sync,
         typename Indexer>
struct fornest_indexglobal_info<
    ::RAJA::policy::hip::hip_indexer<IterationMapping, sync, Indexer>>
{
  static constexpr bool valid        = false;
  static constexpr fornest_axis axis = fornest_axis::none;
  static constexpr int block_size    = -1;
  static constexpr int grid_size     = -1;
};

template<typename IterationMapping,
         kernel_sync_requirement sync,
         named_dim dim,
         size_t BLOCK_SIZE,
         size_t GRID_SIZE>
struct fornest_indexglobal_info<::RAJA::policy::hip::hip_indexer<
    IterationMapping,
    sync,
    ::RAJA::hip::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>>>
{
  static constexpr bool valid        = true;
  static constexpr fornest_axis axis = (dim == named_dim::x)   ? fornest_axis::x
                                       : (dim == named_dim::y) ? fornest_axis::y
                                       : (dim == named_dim::z)
                                           ? fornest_axis::z
                                           : fornest_axis::none;
  static constexpr std::size_t unspecified_sz =
      static_cast<std::size_t>(named_usage::unspecified);
  static constexpr std::size_t ignored_sz =
      static_cast<std::size_t>(named_usage::ignored);
  static constexpr int block_size =
      (BLOCK_SIZE == unspecified_sz || BLOCK_SIZE == ignored_sz)
          ? -1
          : static_cast<int>(BLOCK_SIZE);
  static constexpr int grid_size =
      (GRID_SIZE == unspecified_sz || GRID_SIZE == ignored_sz)
          ? -1
          : static_cast<int>(GRID_SIZE);
};
#endif

struct fornest_map_spec
{
  fornest_map_space space = fornest_map_space::seq;
  fornest_axis axis       = fornest_axis::none;
  bool loop               = false;  // *_loop vs *_direct
};

template<typename Policy>
struct fornest_map_traits
{
  static constexpr fornest_map_spec spec {};
};

template<>
struct fornest_map_traits<RAJA::seq_exec>
{
  static constexpr fornest_map_spec spec {fornest_map_space::seq,
                                          fornest_axis::none, false};
};

#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE)
template<>
struct fornest_map_traits<RAJA::device_global_x_direct>
{
  static constexpr fornest_map_spec spec {fornest_map_space::global,
                                          fornest_axis::x, false};
};

template<>
struct fornest_map_traits<RAJA::device_global_y_direct>
{
  static constexpr fornest_map_spec spec {fornest_map_space::global,
                                          fornest_axis::y, false};
};

template<>
struct fornest_map_traits<RAJA::device_global_z_direct>
{
  static constexpr fornest_map_spec spec {fornest_map_space::global,
                                          fornest_axis::z, false};
};

template<>
struct fornest_map_traits<RAJA::device_block_x_direct>
{
  static constexpr fornest_map_spec spec {fornest_map_space::block,
                                          fornest_axis::x, false};
};

template<>
struct fornest_map_traits<RAJA::device_block_y_direct>
{
  static constexpr fornest_map_spec spec {fornest_map_space::block,
                                          fornest_axis::y, false};
};

template<>
struct fornest_map_traits<RAJA::device_block_z_direct>
{
  static constexpr fornest_map_spec spec {fornest_map_space::block,
                                          fornest_axis::z, false};
};

template<>
struct fornest_map_traits<RAJA::device_thread_x_direct>
{
  static constexpr fornest_map_spec spec {fornest_map_space::thread,
                                          fornest_axis::x, false};
};

template<>
struct fornest_map_traits<RAJA::device_thread_y_direct>
{
  static constexpr fornest_map_spec spec {fornest_map_space::thread,
                                          fornest_axis::y, false};
};

template<>
struct fornest_map_traits<RAJA::device_thread_z_direct>
{
  static constexpr fornest_map_spec spec {fornest_map_space::thread,
                                          fornest_axis::z, false};
};

template<>
struct fornest_map_traits<RAJA::device_thread_x_loop>
{
  static constexpr fornest_map_spec spec {fornest_map_space::thread,
                                          fornest_axis::x, true};
};

template<>
struct fornest_map_traits<RAJA::device_thread_y_loop>
{
  static constexpr fornest_map_spec spec {fornest_map_space::thread,
                                          fornest_axis::y, true};
};

template<>
struct fornest_map_traits<RAJA::device_thread_z_loop>
{
  static constexpr fornest_map_spec spec {fornest_map_space::thread,
                                          fornest_axis::z, true};
};

// Sized device mapping policies (CUDA/HIP). These allow mapping policies to
// fully determine launch Threads/Teams when used with RAJA::fornest.
template<int nx_threads>
struct fornest_map_traits<RAJA::device_global_size_x_direct<nx_threads>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::global,
                                          fornest_axis::x, false};
};

template<int ny_threads>
struct fornest_map_traits<RAJA::device_global_size_y_direct<ny_threads>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::global,
                                          fornest_axis::y, false};
};

template<int nz_threads>
struct fornest_map_traits<RAJA::device_global_size_z_direct<nz_threads>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::global,
                                          fornest_axis::z, false};
};

template<int nx_threads>
struct fornest_map_traits<
    RAJA::device_global_size_x_direct_unchecked<nx_threads>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::global,
                                          fornest_axis::x, false};
};

template<int ny_threads>
struct fornest_map_traits<
    RAJA::device_global_size_y_direct_unchecked<ny_threads>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::global,
                                          fornest_axis::y, false};
};

template<int nz_threads>
struct fornest_map_traits<
    RAJA::device_global_size_z_direct_unchecked<nz_threads>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::global,
                                          fornest_axis::z, false};
};

template<int nx_threads>
struct fornest_map_traits<RAJA::device_global_size_x_loop<nx_threads>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::global,
                                          fornest_axis::x, true};
};

template<int ny_threads>
struct fornest_map_traits<RAJA::device_global_size_y_loop<ny_threads>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::global,
                                          fornest_axis::y, true};
};

template<int nz_threads>
struct fornest_map_traits<RAJA::device_global_size_z_loop<nz_threads>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::global,
                                          fornest_axis::z, true};
};

template<int X_SIZE>
struct fornest_map_traits<RAJA::device_thread_size_x_direct<X_SIZE>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::thread,
                                          fornest_axis::x, false};
};

template<int Y_SIZE>
struct fornest_map_traits<RAJA::device_thread_size_y_direct<Y_SIZE>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::thread,
                                          fornest_axis::y, false};
};

template<int Z_SIZE>
struct fornest_map_traits<RAJA::device_thread_size_z_direct<Z_SIZE>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::thread,
                                          fornest_axis::z, false};
};

template<int X_SIZE>
struct fornest_map_traits<RAJA::device_thread_size_x_direct_unchecked<X_SIZE>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::thread,
                                          fornest_axis::x, false};
};

template<int Y_SIZE>
struct fornest_map_traits<RAJA::device_thread_size_y_direct_unchecked<Y_SIZE>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::thread,
                                          fornest_axis::y, false};
};

template<int Z_SIZE>
struct fornest_map_traits<RAJA::device_thread_size_z_direct_unchecked<Z_SIZE>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::thread,
                                          fornest_axis::z, false};
};

template<int X_SIZE>
struct fornest_map_traits<RAJA::device_thread_size_x_loop<X_SIZE>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::thread,
                                          fornest_axis::x, true};
};

template<int Y_SIZE>
struct fornest_map_traits<RAJA::device_thread_size_y_loop<Y_SIZE>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::thread,
                                          fornest_axis::y, true};
};

template<int Z_SIZE>
struct fornest_map_traits<RAJA::device_thread_size_z_loop<Z_SIZE>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::thread,
                                          fornest_axis::z, true};
};

template<int X_SIZE>
struct fornest_map_traits<RAJA::device_block_size_x_direct<X_SIZE>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::block,
                                          fornest_axis::x, false};
};

template<int Y_SIZE>
struct fornest_map_traits<RAJA::device_block_size_y_direct<Y_SIZE>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::block,
                                          fornest_axis::y, false};
};

template<int Z_SIZE>
struct fornest_map_traits<RAJA::device_block_size_z_direct<Z_SIZE>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::block,
                                          fornest_axis::z, false};
};

template<int X_SIZE>
struct fornest_map_traits<RAJA::device_block_size_x_direct_unchecked<X_SIZE>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::block,
                                          fornest_axis::x, false};
};

template<int Y_SIZE>
struct fornest_map_traits<RAJA::device_block_size_y_direct_unchecked<Y_SIZE>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::block,
                                          fornest_axis::y, false};
};

template<int Z_SIZE>
struct fornest_map_traits<RAJA::device_block_size_z_direct_unchecked<Z_SIZE>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::block,
                                          fornest_axis::z, false};
};

template<int X_SIZE>
struct fornest_map_traits<RAJA::device_block_size_x_loop<X_SIZE>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::block,
                                          fornest_axis::x, true};
};

template<int Y_SIZE>
struct fornest_map_traits<RAJA::device_block_size_y_loop<Y_SIZE>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::block,
                                          fornest_axis::y, true};
};

template<int Z_SIZE>
struct fornest_map_traits<RAJA::device_block_size_z_loop<Z_SIZE>>
{
  static constexpr fornest_map_spec spec {fornest_map_space::block,
                                          fornest_axis::z, true};
};
#endif

template<typename LoopPolicy>
struct fornest_loop_policy_host_t
{
  using type = typename camp::decay<LoopPolicy>::host_policy_t;
};

#if defined(RAJA_GPU_ACTIVE)
template<typename LoopPolicy>
struct fornest_loop_policy_device_t
{
  using type = typename camp::decay<LoopPolicy>::device_policy_t;
};
#endif

template<typename ExecPolicy, typename LoopPolicy>
struct fornest_loop_policy_active_t
{
  using type = typename fornest_loop_policy_host_t<LoopPolicy>::type;
};

#if defined(RAJA_GPU_ACTIVE)
template<typename ExecPolicy, typename LoopPolicy>
  requires(detail::get_platform<camp::decay<ExecPolicy>>::value !=
           Platform::host)
struct fornest_loop_policy_active_t<ExecPolicy, LoopPolicy>
{
  using type = typename fornest_loop_policy_device_t<LoopPolicy>::type;
};
#endif

template<typename ExecPolicy, typename LoopPolicy>
using fornest_loop_policy_active_t_t =
    typename fornest_loop_policy_active_t<ExecPolicy, LoopPolicy>::type;

template<typename ExecPolicy, typename LoopPolicy>
inline constexpr fornest_map_spec fornest_loop_map_spec_v = fornest_map_traits<
    fornest_loop_policy_active_t_t<ExecPolicy, LoopPolicy>>::spec;

template<typename ExecPolicy, typename LoopPolicy>
inline constexpr bool fornest_is_seq_loop_v =
    std::is_same_v<fornest_loop_policy_active_t_t<ExecPolicy, LoopPolicy>,
                   RAJA::seq_exec>;

template<typename ExecPolicy, typename LoopPolicy>
RAJA_INLINE void validate_loop_policy_supported()
{
  constexpr fornest_map_spec spec =
      fornest_loop_map_spec_v<ExecPolicy, LoopPolicy>;
  if constexpr (!fornest_is_seq_loop_v<ExecPolicy, LoopPolicy> &&
                spec.axis == fornest_axis::none)
  {
    RAJA_ABORT_OR_THROW("RAJA::fornest mapping policy uses an unsupported loop "
                        "policy for the active platform");
  }
}

template<typename ExecPolicy, typename Loop0, typename Loop1>
RAJA_INLINE LaunchParams make_launch_params_for_mapping_2d(int block_size,
                                                           std::size_t e0,
                                                           std::size_t e1)
{
  validate_loop_policy_supported<ExecPolicy, Loop0>();
  validate_loop_policy_supported<ExecPolicy, Loop1>();

  constexpr auto s0 = fornest_loop_map_spec_v<ExecPolicy, Loop0>;
  constexpr auto s1 = fornest_loop_map_spec_v<ExecPolicy, Loop1>;

  int requested_thread_x = -1;
  int requested_thread_y = -1;
  int requested_thread_z = -1;
  int requested_grid_x   = -1;
  int requested_grid_y   = -1;
  int requested_grid_z   = -1;

  auto record_requests = [&](auto loop_tag, fornest_map_spec spec) {
    using LoopPolicy = decltype(loop_tag);
    using active_pol = fornest_loop_policy_active_t_t<ExecPolicy, LoopPolicy>;
    using info       = fornest_indexglobal_info<active_pol>;

    if constexpr (info::valid)
    {
      if (info::axis == fornest_axis::x)
      {
        if ((spec.space == fornest_map_space::global ||
             spec.space == fornest_map_space::thread) &&
            info::block_size > 0)
          requested_thread_x = info::block_size;
        if ((spec.space == fornest_map_space::global ||
             spec.space == fornest_map_space::block) &&
            info::grid_size > 0)
          requested_grid_x = info::grid_size;
      }
      if (info::axis == fornest_axis::y)
      {
        if ((spec.space == fornest_map_space::global ||
             spec.space == fornest_map_space::thread) &&
            info::block_size > 0)
          requested_thread_y = info::block_size;
        if ((spec.space == fornest_map_space::global ||
             spec.space == fornest_map_space::block) &&
            info::grid_size > 0)
          requested_grid_y = info::grid_size;
      }
      if (info::axis == fornest_axis::z)
      {
        if ((spec.space == fornest_map_space::global ||
             spec.space == fornest_map_space::thread) &&
            info::block_size > 0)
          requested_thread_z = info::block_size;
        if ((spec.space == fornest_map_space::global ||
             spec.space == fornest_map_space::block) &&
            info::grid_size > 0)
          requested_grid_z = info::grid_size;
      }
    }
  };

  record_requests(Loop0 {}, s0);
  record_requests(Loop1 {}, s1);

  // Identify which logical dimension (0/1) maps to thread/global axis.
  bool use_x     = false;
  bool use_y     = false;
  std::size_t ex = 1;
  std::size_t ey = 1;

  auto record_axis = [&](fornest_axis ax, std::size_t extent) {
    if (ax == fornest_axis::x)
    {
      if (use_x)
      {
        RAJA_ABORT_OR_THROW(
            "RAJA::fornest: multiple dimensions mapped to thread/global x");
      }
      use_x = true;
      ex    = extent;
    }
    else if (ax == fornest_axis::y)
    {
      if (use_y)
      {
        RAJA_ABORT_OR_THROW(
            "RAJA::fornest: multiple dimensions mapped to thread/global y");
      }
      use_y = true;
      ey    = extent;
    }
  };

  if (s0.space == fornest_map_space::global ||
      s0.space == fornest_map_space::thread)
  {
    record_axis(s0.axis, e0);
  }
  if (s1.space == fornest_map_space::global ||
      s1.space == fornest_map_space::thread)
  {
    record_axis(s1.axis, e1);
  }

  tile3 tile {};
  tile.z = 1;

  const bool any_sized_threads = (requested_thread_x > 0) ||
                                 (requested_thread_y > 0) ||
                                 (requested_thread_z > 0);

  auto require_consistent_block_size = [&](int threads_product) {
    if (threads_product != block_size)
    {
      RAJA_ABORT_OR_THROW(
          "RAJA::fornest: sized mapping requests a Threads configuration that "
          "is inconsistent with the ExecPolicy block size");
    }
  };

  if (use_x && use_y)
  {
    const int sx = (requested_thread_x > 0) ? requested_thread_x : -1;
    const int sy = (requested_thread_y > 0) ? requested_thread_y : -1;

    if (sx > 0 || sy > 0)
    {
      if ((sx > 0) && (sy > 0))
      {
        tile.x = sx;
        tile.y = sy;
        require_consistent_block_size(tile.x * tile.y);
      }
      else
      {
        const int fixed = (sx > 0) ? sx : sy;
        if (fixed <= 0 || (block_size % fixed) != 0)
        {
          RAJA_ABORT_OR_THROW(
              "RAJA::fornest: sized mapping requires ExecPolicy block size to "
              "be divisible by the specified axis size");
        }
        const int other = block_size / fixed;
        tile.x          = (sx > 0) ? fixed : other;
        tile.y          = (sy > 0) ? fixed : other;
      }
    }
    else
    {
      tile = choose_2d_tile(block_size, ex, ey);
    }
  }
  else if (use_x)
  {
    if (requested_thread_x > 0)
    {
      tile.x = requested_thread_x;
      tile.y = 1;
      require_consistent_block_size(tile.x * tile.y);
    }
    else
    {
      tile.x = static_cast<int>(
          std::min<std::size_t>(ex, static_cast<std::size_t>(block_size)));
      if (tile.x < 1) tile.x = 1;
      tile.y = 1;
    }
  }
  else if (use_y)
  {
    if (requested_thread_y > 0)
    {
      tile.y = requested_thread_y;
      tile.x = 1;
      require_consistent_block_size(tile.x * tile.y);
    }
    else
    {
      tile.y = static_cast<int>(
          std::min<std::size_t>(ey, static_cast<std::size_t>(block_size)));
      if (tile.y < 1) tile.y = 1;
      tile.x = 1;
    }
  }
  else
  {
    // No thread/global mapping. Use 1 thread by default unless the user sized
    // thread policies (rare, but keep consistent).
    tile.x = any_sized_threads ? block_size : 1;
    tile.y = 1;
  }

  Teams teams {};
  Threads threads(tile.x, tile.y);

  // Default to 1 team in each dimension.
  teams.value[0] = 1;
  teams.value[1] = 1;
  teams.value[2] = 1;

  auto axis_thread_cap = [&](fornest_axis ax) -> int {
    if (ax == fornest_axis::x) return tile.x;
    if (ax == fornest_axis::y) return tile.y;
    if (ax == fornest_axis::z) return tile.z;
    return 1;
  };

  auto axis_grid_req = [&](fornest_axis ax) -> int {
    if (ax == fornest_axis::x) return requested_grid_x;
    if (ax == fornest_axis::y) return requested_grid_y;
    if (ax == fornest_axis::z) return requested_grid_z;
    return -1;
  };

  auto set_teams_axis = [&](fornest_axis ax, int v) {
    if (ax == fornest_axis::x) teams.value[0] = v;
    if (ax == fornest_axis::y) teams.value[1] = v;
    if (ax == fornest_axis::z) teams.value[2] = v;
  };

  auto apply_dim = [&](fornest_map_spec spec, std::size_t extent) {
    if (spec.space == fornest_map_space::seq)
    {
      return;
    }

    if (spec.space == fornest_map_space::block)
    {
      if (spec.loop)
      {
        RAJA_ABORT_OR_THROW(
            "RAJA::fornest: block_*_loop mapping not supported in v1");
      }

      const int req = axis_grid_req(spec.axis);
      const int t   = (req > 0) ? req : static_cast<int>(extent);
      if (req > 0 && extent > static_cast<std::size_t>(req))
      {
        RAJA_ABORT_OR_THROW("RAJA::fornest: block_*_direct extent exceeds "
                            "requested grid size");
      }
      set_teams_axis(spec.axis, t);
      return;
    }

    if (spec.space == fornest_map_space::global)
    {
      const int cap = axis_thread_cap(spec.axis);
      const int req = axis_grid_req(spec.axis);
      if (req > 0)
      {
        if (extent >
            static_cast<std::size_t>(req) * static_cast<std::size_t>(cap))
        {
          RAJA_ABORT_OR_THROW("RAJA::fornest: global_*_direct extent exceeds "
                              "requested grid size");
        }
        set_teams_axis(spec.axis, req);
      }
      else
      {
        set_teams_axis(spec.axis,
                       static_cast<int>(ceil_div<std::size_t>(extent, cap)));
      }
      return;
    }

    if (spec.space == fornest_map_space::thread)
    {
      if (!spec.loop)
      {
        const int cap = axis_thread_cap(spec.axis);
        if (extent > static_cast<std::size_t>(cap))
        {
          RAJA_ABORT_OR_THROW("RAJA::fornest: thread_*_direct mapping requires "
                              "extent <= Threads in that axis");
        }
      }
      return;
    }
  };

  apply_dim(s0, e0);
  apply_dim(s1, e1);

  // Ensure we launch at least one team.
  if (teams.value[0] < 1) teams.value[0] = 1;
  if (teams.value[1] < 1) teams.value[1] = 1;
  if (teams.value[2] < 1) teams.value[2] = 1;

  return LaunchParams(teams, threads);
}

template<typename ExecPolicy, typename Loop0, typename Loop1, typename Loop2>
RAJA_INLINE LaunchParams make_launch_params_for_mapping_3d(int block_size,
                                                           std::size_t e0,
                                                           std::size_t e1,
                                                           std::size_t e2)
{
  validate_loop_policy_supported<ExecPolicy, Loop0>();
  validate_loop_policy_supported<ExecPolicy, Loop1>();
  validate_loop_policy_supported<ExecPolicy, Loop2>();

  constexpr auto s0 = fornest_loop_map_spec_v<ExecPolicy, Loop0>;
  constexpr auto s1 = fornest_loop_map_spec_v<ExecPolicy, Loop1>;
  constexpr auto s2 = fornest_loop_map_spec_v<ExecPolicy, Loop2>;

  int requested_thread_x = -1;
  int requested_thread_y = -1;
  int requested_thread_z = -1;
  int requested_grid_x   = -1;
  int requested_grid_y   = -1;
  int requested_grid_z   = -1;

  auto record_requests = [&](auto loop_tag, fornest_map_spec spec) {
    using LoopPolicy = decltype(loop_tag);
    using active_pol = fornest_loop_policy_active_t_t<ExecPolicy, LoopPolicy>;
    using info       = fornest_indexglobal_info<active_pol>;

    if constexpr (info::valid)
    {
      if (info::axis == fornest_axis::x)
      {
        if ((spec.space == fornest_map_space::global ||
             spec.space == fornest_map_space::thread) &&
            info::block_size > 0)
          requested_thread_x = info::block_size;
        if ((spec.space == fornest_map_space::global ||
             spec.space == fornest_map_space::block) &&
            info::grid_size > 0)
          requested_grid_x = info::grid_size;
      }
      if (info::axis == fornest_axis::y)
      {
        if ((spec.space == fornest_map_space::global ||
             spec.space == fornest_map_space::thread) &&
            info::block_size > 0)
          requested_thread_y = info::block_size;
        if ((spec.space == fornest_map_space::global ||
             spec.space == fornest_map_space::block) &&
            info::grid_size > 0)
          requested_grid_y = info::grid_size;
      }
      if (info::axis == fornest_axis::z)
      {
        if ((spec.space == fornest_map_space::global ||
             spec.space == fornest_map_space::thread) &&
            info::block_size > 0)
          requested_thread_z = info::block_size;
        if ((spec.space == fornest_map_space::global ||
             spec.space == fornest_map_space::block) &&
            info::grid_size > 0)
          requested_grid_z = info::grid_size;
      }
    }
  };

  record_requests(Loop0 {}, s0);
  record_requests(Loop1 {}, s1);
  record_requests(Loop2 {}, s2);

  bool use_x     = false;
  bool use_y     = false;
  bool use_z     = false;
  std::size_t ex = 1;
  std::size_t ey = 1;
  std::size_t ez = 1;

  auto record_axis = [&](fornest_axis ax, std::size_t extent) {
    if (ax == fornest_axis::x)
    {
      if (use_x)
      {
        RAJA_ABORT_OR_THROW(
            "RAJA::fornest: multiple dimensions mapped to thread/global x");
      }
      use_x = true;
      ex    = extent;
    }
    else if (ax == fornest_axis::y)
    {
      if (use_y)
      {
        RAJA_ABORT_OR_THROW(
            "RAJA::fornest: multiple dimensions mapped to thread/global y");
      }
      use_y = true;
      ey    = extent;
    }
    else if (ax == fornest_axis::z)
    {
      if (use_z)
      {
        RAJA_ABORT_OR_THROW(
            "RAJA::fornest: multiple dimensions mapped to thread/global z");
      }
      use_z = true;
      ez    = extent;
    }
  };

  auto consider = [&](fornest_map_spec spec, std::size_t extent) {
    if (spec.space == fornest_map_space::global ||
        spec.space == fornest_map_space::thread)
    {
      record_axis(spec.axis, extent);
    }
  };

  consider(s0, e0);
  consider(s1, e1);
  consider(s2, e2);

  tile3 tile {};
  const bool any_sized_threads = (requested_thread_x > 0) ||
                                 (requested_thread_y > 0) ||
                                 (requested_thread_z > 0);

  auto require_consistent_block_size = [&](int threads_product) {
    if (threads_product != block_size)
    {
      RAJA_ABORT_OR_THROW(
          "RAJA::fornest: sized mapping requests a Threads configuration that "
          "is inconsistent with the ExecPolicy block size");
    }
  };

  auto min_extent_or = [&](std::size_t n) -> int {
    int v = static_cast<int>(
        std::min<std::size_t>(n, static_cast<std::size_t>(block_size)));
    return (v < 1) ? 1 : v;
  };

  if (use_x && use_y && use_z)
  {
    int sx = (requested_thread_x > 0) ? requested_thread_x : -1;
    int sy = (requested_thread_y > 0) ? requested_thread_y : -1;
    int sz = (requested_thread_z > 0) ? requested_thread_z : -1;

    const bool any = (sx > 0) || (sy > 0) || (sz > 0);
    if (any)
    {
      int product = 1;
      int missing = 0;
      if (sx > 0)
        product *= sx;
      else
        missing++;
      if (sy > 0)
        product *= sy;
      else
        missing++;
      if (sz > 0)
        product *= sz;
      else
        missing++;

      if (product <= 0 || (block_size % product) != 0)
      {
        RAJA_ABORT_OR_THROW(
            "RAJA::fornest: sized mapping requires ExecPolicy block size to be "
            "divisible by the product of specified axis sizes");
      }
      int remaining = block_size / product;

      if (missing == 0)
      {
        tile.x = sx;
        tile.y = sy;
        tile.z = sz;
        require_consistent_block_size(tile.x * tile.y * tile.z);
      }
      else if (missing == 1)
      {
        if (sx < 0) sx = remaining;
        if (sy < 0) sy = remaining;
        if (sz < 0) sz = remaining;
        tile.x = sx;
        tile.y = sy;
        tile.z = sz;
      }
      else if (missing == 2)
      {
        // Choose a reasonable 2D tile for the remaining axes.
        tile3 t2 {};
        if (sx > 0)
        {
          t2     = choose_2d_tile(remaining, ey, ez);
          tile.x = sx;
          tile.y = t2.x;
          tile.z = t2.y;
        }
        else if (sy > 0)
        {
          t2     = choose_2d_tile(remaining, ex, ez);
          tile.y = sy;
          tile.x = t2.x;
          tile.z = t2.y;
        }
        else
        {
          t2     = choose_2d_tile(remaining, ex, ey);
          tile.z = sz;
          tile.x = t2.x;
          tile.y = t2.y;
        }
      }
      else
      {
        tile = choose_3d_tile(block_size, ex, ey, ez);
      }
    }
    else
    {
      tile = choose_3d_tile(block_size, ex, ey, ez);
    }
  }
  else if (use_x && use_y)
  {
    const int sx = (requested_thread_x > 0) ? requested_thread_x : -1;
    const int sy = (requested_thread_y > 0) ? requested_thread_y : -1;

    if (sx > 0 || sy > 0)
    {
      if ((sx > 0) && (sy > 0))
      {
        tile.x = sx;
        tile.y = sy;
        tile.z = 1;
        require_consistent_block_size(tile.x * tile.y * tile.z);
      }
      else
      {
        const int fixed = (sx > 0) ? sx : sy;
        if (fixed <= 0 || (block_size % fixed) != 0)
        {
          RAJA_ABORT_OR_THROW(
              "RAJA::fornest: sized mapping requires ExecPolicy block size to "
              "be divisible by the specified axis size");
        }
        const int other = block_size / fixed;
        tile.x          = (sx > 0) ? fixed : other;
        tile.y          = (sy > 0) ? fixed : other;
        tile.z          = 1;
      }
    }
    else
    {
      tile   = choose_2d_tile(block_size, ex, ey);
      tile.z = 1;
    }
  }
  else if (use_x && use_z)
  {
    // use x/z (rare in 3D mapping policies, but support)
    const int sx = (requested_thread_x > 0) ? requested_thread_x : -1;
    const int sz = (requested_thread_z > 0) ? requested_thread_z : -1;
    if (sx > 0 || sz > 0)
    {
      if ((sx > 0) && (sz > 0))
      {
        tile.x = sx;
        tile.z = sz;
        tile.y = 1;
        require_consistent_block_size(tile.x * tile.y * tile.z);
      }
      else
      {
        const int fixed = (sx > 0) ? sx : sz;
        if (fixed <= 0 || (block_size % fixed) != 0)
        {
          RAJA_ABORT_OR_THROW(
              "RAJA::fornest: sized mapping requires ExecPolicy block size to "
              "be divisible by the specified axis size");
        }
        const int other = block_size / fixed;
        tile.x          = (sx > 0) ? fixed : other;
        tile.z          = (sz > 0) ? fixed : other;
        tile.y          = 1;
      }
    }
    else
    {
      tile3 t2 = choose_2d_tile(block_size, ex, ez);
      tile.x   = t2.x;
      tile.z   = t2.y;
      tile.y   = 1;
    }
  }
  else if (use_y && use_z)
  {
    const int sy = (requested_thread_y > 0) ? requested_thread_y : -1;
    const int sz = (requested_thread_z > 0) ? requested_thread_z : -1;
    if (sy > 0 || sz > 0)
    {
      if ((sy > 0) && (sz > 0))
      {
        tile.y = sy;
        tile.z = sz;
        tile.x = 1;
        require_consistent_block_size(tile.x * tile.y * tile.z);
      }
      else
      {
        const int fixed = (sy > 0) ? sy : sz;
        if (fixed <= 0 || (block_size % fixed) != 0)
        {
          RAJA_ABORT_OR_THROW(
              "RAJA::fornest: sized mapping requires ExecPolicy block size to "
              "be divisible by the specified axis size");
        }
        const int other = block_size / fixed;
        tile.y          = (sy > 0) ? fixed : other;
        tile.z          = (sz > 0) ? fixed : other;
        tile.x          = 1;
      }
    }
    else
    {
      tile3 t2 = choose_2d_tile(block_size, ey, ez);
      tile.y   = t2.x;
      tile.z   = t2.y;
      tile.x   = 1;
    }
  }
  else if (use_x)
  {
    if (requested_thread_x > 0)
    {
      tile.x = requested_thread_x;
      tile.y = 1;
      tile.z = 1;
      require_consistent_block_size(tile.x * tile.y * tile.z);
    }
    else
    {
      tile.x = min_extent_or(ex);
      tile.y = 1;
      tile.z = 1;
    }
  }
  else if (use_y)
  {
    if (requested_thread_y > 0)
    {
      tile.y = requested_thread_y;
      tile.x = 1;
      tile.z = 1;
      require_consistent_block_size(tile.x * tile.y * tile.z);
    }
    else
    {
      tile.y = min_extent_or(ey);
      tile.x = 1;
      tile.z = 1;
    }
  }
  else if (use_z)
  {
    if (requested_thread_z > 0)
    {
      tile.z = requested_thread_z;
      tile.x = 1;
      tile.y = 1;
      require_consistent_block_size(tile.x * tile.y * tile.z);
    }
    else
    {
      tile.z = min_extent_or(ez);
      tile.x = 1;
      tile.y = 1;
    }
  }
  else
  {
    tile.x = any_sized_threads ? block_size : 1;
    tile.y = 1;
    tile.z = 1;
  }

  Teams teams {};
  Threads threads(tile.x, tile.y, tile.z);
  teams.value[0] = 1;
  teams.value[1] = 1;
  teams.value[2] = 1;

  auto axis_thread_cap = [&](fornest_axis ax) -> int {
    if (ax == fornest_axis::x) return tile.x;
    if (ax == fornest_axis::y) return tile.y;
    if (ax == fornest_axis::z) return tile.z;
    return 1;
  };

  auto axis_grid_req = [&](fornest_axis ax) -> int {
    if (ax == fornest_axis::x) return requested_grid_x;
    if (ax == fornest_axis::y) return requested_grid_y;
    if (ax == fornest_axis::z) return requested_grid_z;
    return -1;
  };

  auto set_teams_axis = [&](fornest_axis ax, int v) {
    if (ax == fornest_axis::x) teams.value[0] = v;
    if (ax == fornest_axis::y) teams.value[1] = v;
    if (ax == fornest_axis::z) teams.value[2] = v;
  };

  auto apply_dim = [&](fornest_map_spec spec, std::size_t extent) {
    if (spec.space == fornest_map_space::seq)
    {
      return;
    }

    if (spec.space == fornest_map_space::block)
    {
      if (spec.loop)
      {
        RAJA_ABORT_OR_THROW(
            "RAJA::fornest: block_*_loop mapping not supported in v1");
      }

      const int req = axis_grid_req(spec.axis);
      const int t   = (req > 0) ? req : static_cast<int>(extent);
      if (req > 0 && extent > static_cast<std::size_t>(req))
      {
        RAJA_ABORT_OR_THROW("RAJA::fornest: block_*_direct extent exceeds "
                            "requested grid size");
      }
      set_teams_axis(spec.axis, t);
      return;
    }

    if (spec.space == fornest_map_space::global)
    {
      const int cap = axis_thread_cap(spec.axis);
      const int req = axis_grid_req(spec.axis);
      if (req > 0)
      {
        if (extent >
            static_cast<std::size_t>(req) * static_cast<std::size_t>(cap))
        {
          RAJA_ABORT_OR_THROW("RAJA::fornest: global_*_direct extent exceeds "
                              "requested grid size");
        }
        set_teams_axis(spec.axis, req);
      }
      else
      {
        set_teams_axis(spec.axis,
                       static_cast<int>(ceil_div<std::size_t>(extent, cap)));
      }
      return;
    }

    if (spec.space == fornest_map_space::thread)
    {
      if (!spec.loop)
      {
        const int cap = axis_thread_cap(spec.axis);
        if (extent > static_cast<std::size_t>(cap))
        {
          RAJA_ABORT_OR_THROW("RAJA::fornest: thread_*_direct mapping requires "
                              "extent <= Threads in that axis");
        }
      }
      return;
    }
  };

  apply_dim(s0, e0);
  apply_dim(s1, e1);
  apply_dim(s2, e2);

  if (teams.value[0] < 1) teams.value[0] = 1;
  if (teams.value[1] < 1) teams.value[1] = 1;
  if (teams.value[2] < 1) teams.value[2] = 1;

  return LaunchParams(teams, threads);
}

}  // namespace detail

//------------------------------------------------------------------------------
// Static fornest template wrappers (match forall style)
//------------------------------------------------------------------------------
template<typename Policy, typename... Args>
  requires(!detail::is_camp_list_v<Policy>)
RAJA_INLINE auto fornest(Args&&... args)
{
  // Use unqualified call so argument-dependent lookup can find the
  // ExecPolicy-based overloads declared later in this header.
  return fornest(Policy {}, std::forward<Args>(args)...);
}

//------------------------------------------------------------------------------
// fornest: explicit mapping policy (2D)
//------------------------------------------------------------------------------
template<typename ExecPolicy,
         typename Loop0,
         typename Loop1,
         typename Seg0,
         typename Seg1,
         typename... Params>
  requires(!detail::first_arg_is_segment_like_v<Params...>)
RAJA_INLINE auto fornest(fornest_mapping_policy<ExecPolicy, Loop0, Loop1>,
                         Seg0 const& seg0,
                         Seg1 const& seg1,
                         Params&&... params)
{
  constexpr Platform platform =
      detail::get_platform<camp::decay<ExecPolicy>>::value;

  auto&& user_body = expt::get_lambda(std::forward<Params>(params)...);
  auto body_c      = camp::decay<decltype(user_body)>(
      std::forward<decltype(user_body)>(user_body));

  auto seg0_c = seg0;
  auto seg1_c = seg1;

  auto args_tuple = camp::forward_as_tuple(std::forward<Params>(params)...);

  if constexpr (platform == Platform::host)
  {
    using host_ctx   = RAJA::LaunchContextT<RAJA::LaunchContextHostPolicy>;
    using launch_pol = RAJA::LaunchPolicy<RAJA::seq_launch_t>;
    auto launch_body =
        detail::FornestLaunchBody2D<host_ctx, Loop0, Loop1, decltype(seg0_c),
                                    decltype(seg1_c), decltype(body_c)>(
            seg0_c, seg1_c, std::move(body_c));
    return detail::apply_without_last(
        [&](auto&&... opt_args) {
          return RAJA::launch<launch_pol>(
              LaunchParams {}, std::forward<decltype(opt_args)>(opt_args)...,
              launch_body);
        },
        std::move(args_tuple));
  }
#if defined(RAJA_GPU_ACTIVE)
  else
  {
    const std::size_t e0 = detail::segment_length_host(seg0);
    const std::size_t e1 = detail::segment_length_host(seg1);

    constexpr int block_size = detail::exec_block_size_v<ExecPolicy>;
    static_assert(block_size > 0,
                  "RAJA::fornest mapping requires an execution policy with a "
                  "fixed block size on GPU");

    const LaunchParams launch_params =
        detail::make_launch_params_for_mapping_2d<ExecPolicy, Loop0, Loop1>(
            block_size, e0, e1);

    using launch_pol = detail::fornest_launch_policy_t<ExecPolicy>;
    auto launch_body =
        detail::FornestLaunchBody2D<RAJA::LaunchContext, Loop0, Loop1,
                                    decltype(seg0_c), decltype(seg1_c),
                                    decltype(body_c)>(seg0_c, seg1_c,
                                                      std::move(body_c));
    return detail::apply_without_last(
        [&](auto&&... opt_args) {
          return RAJA::launch<launch_pol>(
              launch_params, std::forward<decltype(opt_args)>(opt_args)...,
              launch_body);
        },
        std::move(args_tuple));
  }
#else
  else
  {
    RAJA_ABORT_OR_THROW("RAJA::fornest: device execution requested but GPU is "
                        "not enabled");
  }
#endif
}

template<typename ExecPolicy,
         typename Loop0,
         typename Loop1,
         typename Seg0,
         typename Seg1,
         typename... Params>
RAJA_INLINE auto fornest(
    typename resources::get_resource<camp::decay<ExecPolicy>>::type r,
    fornest_mapping_policy<ExecPolicy, Loop0, Loop1>,
    Seg0 const& seg0,
    Seg1 const& seg1,
    Params&&... params)
{
  constexpr Platform platform =
      detail::get_platform<camp::decay<ExecPolicy>>::value;

  auto&& user_body = expt::get_lambda(std::forward<Params>(params)...);
  auto body_c      = camp::decay<decltype(user_body)>(
      std::forward<decltype(user_body)>(user_body));

  auto seg0_c = seg0;
  auto seg1_c = seg1;

  auto args_tuple = camp::forward_as_tuple(std::forward<Params>(params)...);

  if constexpr (platform == Platform::host)
  {
    using host_ctx   = RAJA::LaunchContextT<RAJA::LaunchContextHostPolicy>;
    using launch_pol = RAJA::LaunchPolicy<RAJA::seq_launch_t>;
    auto launch_body =
        detail::FornestLaunchBody2D<host_ctx, Loop0, Loop1, decltype(seg0_c),
                                    decltype(seg1_c), decltype(body_c)>(
            seg0_c, seg1_c, std::move(body_c));
    return detail::apply_without_last(
        [&](auto&&... opt_args) {
          return RAJA::launch<launch_pol>(
              r, LaunchParams {}, std::forward<decltype(opt_args)>(opt_args)...,
              launch_body);
        },
        std::move(args_tuple));
  }
#if defined(RAJA_GPU_ACTIVE)
  else
  {
    const std::size_t e0 = detail::segment_length_host(seg0);
    const std::size_t e1 = detail::segment_length_host(seg1);

    constexpr int block_size = detail::exec_block_size_v<ExecPolicy>;
    static_assert(block_size > 0,
                  "RAJA::fornest mapping requires an execution policy with a "
                  "fixed block size on GPU");

    const LaunchParams launch_params =
        detail::make_launch_params_for_mapping_2d<ExecPolicy, Loop0, Loop1>(
            block_size, e0, e1);

    using launch_pol = detail::fornest_launch_policy_t<ExecPolicy>;
    auto launch_body =
        detail::FornestLaunchBody2D<RAJA::LaunchContext, Loop0, Loop1,
                                    decltype(seg0_c), decltype(seg1_c),
                                    decltype(body_c)>(seg0_c, seg1_c,
                                                      std::move(body_c));
    return detail::apply_without_last(
        [&](auto&&... opt_args) {
          return RAJA::launch<launch_pol>(
              r, launch_params, std::forward<decltype(opt_args)>(opt_args)...,
              launch_body);
        },
        std::move(args_tuple));
  }
#else
  else
  {
    RAJA_ABORT_OR_THROW("RAJA::fornest: device execution requested but GPU is "
                        "not enabled");
  }
#endif
}

//------------------------------------------------------------------------------
// fornest: explicit mapping policy (3D)
//------------------------------------------------------------------------------
template<typename ExecPolicy,
         typename Loop0,
         typename Loop1,
         typename Loop2,
         typename Seg0,
         typename Seg1,
         typename Seg2,
         typename... Args,
         typename Body>
  requires detail::segment_like<Seg2>
RAJA_INLINE auto fornest(
    fornest_mapping_policy<ExecPolicy, Loop0, Loop1, Loop2>,
    Seg0 const& seg0,
    Seg1 const& seg1,
    Seg2 const& seg2,
    Args&&... args,
    Body&& body)
{
  constexpr Platform platform =
      detail::get_platform<camp::decay<ExecPolicy>>::value;

  auto body_c = camp::decay<Body>(std::forward<Body>(body));

  if constexpr (platform == Platform::host)
  {
    using host_ctx   = RAJA::LaunchContextT<RAJA::LaunchContextHostPolicy>;
    using launch_pol = RAJA::LaunchPolicy<RAJA::seq_launch_t>;
    return RAJA::launch<launch_pol>(
        LaunchParams {}, std::forward<Args>(args)...,
        detail::FornestLaunchBody3D<host_ctx, Loop0, Loop1, Loop2, Seg0, Seg1,
                                    Seg2, decltype(body_c)>(seg0, seg1, seg2,
                                                            std::move(body_c)));
  }
#if defined(RAJA_GPU_ACTIVE)
  else
  {
    const std::size_t e0 = detail::segment_length_host(seg0);
    const std::size_t e1 = detail::segment_length_host(seg1);
    const std::size_t e2 = detail::segment_length_host(seg2);

    constexpr int block_size = detail::exec_block_size_v<ExecPolicy>;
    static_assert(block_size > 0,
                  "RAJA::fornest mapping requires an execution policy with a "
                  "fixed block size on GPU");

    const LaunchParams launch_params =
        detail::make_launch_params_for_mapping_3d<ExecPolicy, Loop0, Loop1,
                                                  Loop2>(block_size, e0, e1,
                                                         e2);

    using launch_pol = detail::fornest_launch_policy_t<ExecPolicy>;
    return RAJA::launch<launch_pol>(
        launch_params, std::forward<Args>(args)...,
        detail::FornestLaunchBody3D<RAJA::LaunchContext, Loop0, Loop1, Loop2,
                                    Seg0, Seg1, Seg2, decltype(body_c)>(
            seg0, seg1, seg2, std::move(body_c)));
  }
#else
  else
  {
    RAJA_ABORT_OR_THROW("RAJA::fornest: device execution requested but GPU is "
                        "not enabled");
  }
#endif
}

template<typename ExecPolicy,
         typename Loop0,
         typename Loop1,
         typename Loop2,
         typename Seg0,
         typename Seg1,
         typename Seg2,
         typename... Args,
         typename Body>
  requires detail::segment_like<Seg2>
RAJA_INLINE auto fornest(
    typename resources::get_resource<camp::decay<ExecPolicy>>::type r,
    fornest_mapping_policy<ExecPolicy, Loop0, Loop1, Loop2>,
    Seg0 const& seg0,
    Seg1 const& seg1,
    Seg2 const& seg2,
    Args&&... args,
    Body&& body)
{
  constexpr Platform platform =
      detail::get_platform<camp::decay<ExecPolicy>>::value;

  auto body_c = camp::decay<Body>(std::forward<Body>(body));

  if constexpr (platform == Platform::host)
  {
    using host_ctx   = RAJA::LaunchContextT<RAJA::LaunchContextHostPolicy>;
    using launch_pol = RAJA::LaunchPolicy<RAJA::seq_launch_t>;
    return RAJA::launch<launch_pol>(
        r, LaunchParams {}, std::forward<Args>(args)...,
        detail::FornestLaunchBody3D<host_ctx, Loop0, Loop1, Loop2, Seg0, Seg1,
                                    Seg2, decltype(body_c)>(seg0, seg1, seg2,
                                                            std::move(body_c)));
  }
#if defined(RAJA_GPU_ACTIVE)
  else
  {
    const std::size_t e0 = detail::segment_length_host(seg0);
    const std::size_t e1 = detail::segment_length_host(seg1);
    const std::size_t e2 = detail::segment_length_host(seg2);

    constexpr int block_size = detail::exec_block_size_v<ExecPolicy>;
    static_assert(block_size > 0,
                  "RAJA::fornest mapping requires an execution policy with a "
                  "fixed block size on GPU");

    const LaunchParams launch_params =
        detail::make_launch_params_for_mapping_3d<ExecPolicy, Loop0, Loop1,
                                                  Loop2>(block_size, e0, e1,
                                                         e2);

    using launch_pol = detail::fornest_launch_policy_t<ExecPolicy>;
    return RAJA::launch<launch_pol>(
        r, launch_params, std::forward<Args>(args)...,
        detail::FornestLaunchBody3D<RAJA::LaunchContext, Loop0, Loop1, Loop2,
                                    Seg0, Seg1, Seg2, decltype(body_c)>(
            seg0, seg1, seg2, std::move(body_c)));
  }
#else
  else
  {
    RAJA_ABORT_OR_THROW("RAJA::fornest: device execution requested but GPU is "
                        "not enabled");
  }
#endif
}

//------------------------------------------------------------------------------
// fornest: flattened wrapper policy (2D)
//------------------------------------------------------------------------------
template<typename ExecPolicy,
         typename LayoutTag,
         typename Seg0,
         typename Seg1,
         typename... Params>
  requires(!detail::first_arg_is_segment_like_v<Params...>)
RAJA_INLINE auto fornest(fornest_flattened_policy<ExecPolicy, LayoutTag>,
                         Seg0 const& seg0,
                         Seg1 const& seg1,
                         Params&&... params)
{
  const std::size_t n0    = detail::segment_length_host(seg0);
  const std::size_t n1    = detail::segment_length_host(seg1);
  const std::size_t total = n0 * n1;

  auto f_params = expt::make_forall_param_pack(std::forward<Params>(params)...);
  std::string kernel_name =
      expt::get_kernel_name(std::forward<Params>(params)...);
  auto&& loop_body = expt::get_lambda(std::forward<Params>(params)...);
  expt::check_forall_optional_args(loop_body, f_params);

  util::PluginContext context {
      util::make_context<camp::decay<ExecPolicy>>(std::move(kernel_name))};
  util::callPreCapturePlugins(context);

  using RAJA::util::trigger_updates_before;
  auto body = trigger_updates_before(loop_body);

  util::callPostCapturePlugins(context);
  util::callPreLaunchPlugins(context);

  auto body_c = camp::decay<decltype(body)>(std::move(body));

  using Res = typename resources::get_resource<camp::decay<ExecPolicy>>::type;
  auto flat_body = detail::FornestFlattenedForallBody2D<LayoutTag, Seg0, Seg1,
                                                        decltype(body_c)>(
      n0, n1, seg0, seg1, std::move(body_c));
  auto e = wrap::forall(Res::get_default(), ExecPolicy {},
                        RAJA::TypedRangeSegment<std::size_t>(0, total),
                        std::move(flat_body), f_params);

  util::callPostLaunchPlugins(context);
  return e;
}

template<typename ExecPolicy,
         typename LayoutTag,
         typename Seg0,
         typename Seg1,
         typename... Params>
RAJA_INLINE auto fornest(
    typename resources::get_resource<camp::decay<ExecPolicy>>::type r,
    fornest_flattened_policy<ExecPolicy, LayoutTag>,
    Seg0 const& seg0,
    Seg1 const& seg1,
    Params&&... params)
{
  const std::size_t n0    = detail::segment_length_host(seg0);
  const std::size_t n1    = detail::segment_length_host(seg1);
  const std::size_t total = n0 * n1;

  auto f_params = expt::make_forall_param_pack(std::forward<Params>(params)...);
  std::string kernel_name =
      expt::get_kernel_name(std::forward<Params>(params)...);
  auto&& loop_body = expt::get_lambda(std::forward<Params>(params)...);
  expt::check_forall_optional_args(loop_body, f_params);

  util::PluginContext context {
      util::make_context<camp::decay<ExecPolicy>>(std::move(kernel_name))};
  util::callPreCapturePlugins(context);

  using RAJA::util::trigger_updates_before;
  auto body = trigger_updates_before(loop_body);

  util::callPostCapturePlugins(context);
  util::callPreLaunchPlugins(context);

  auto body_c = camp::decay<decltype(body)>(std::move(body));

  auto flat_body = detail::FornestFlattenedForallBody2D<LayoutTag, Seg0, Seg1,
                                                        decltype(body_c)>(
      n0, n1, seg0, seg1, std::move(body_c));
  auto e = wrap::forall(r, ExecPolicy {},
                        RAJA::TypedRangeSegment<std::size_t>(0, total),
                        std::move(flat_body), f_params);

  util::callPostLaunchPlugins(context);
  return e;
}

//------------------------------------------------------------------------------
// fornest: flattened wrapper policy (3D)
//------------------------------------------------------------------------------
template<typename ExecPolicy,
         typename LayoutTag,
         typename Seg0,
         typename Seg1,
         typename Seg2,
         typename... Params>
  requires detail::segment_like<Seg2>
RAJA_INLINE auto fornest(fornest_flattened_policy<ExecPolicy, LayoutTag>,
                         Seg0 const& seg0,
                         Seg1 const& seg1,
                         Seg2 const& seg2,
                         Params&&... params)
{
  const std::size_t n0    = detail::segment_length_host(seg0);
  const std::size_t n1    = detail::segment_length_host(seg1);
  const std::size_t n2    = detail::segment_length_host(seg2);
  const std::size_t total = n0 * n1 * n2;

  auto f_params = expt::make_forall_param_pack(std::forward<Params>(params)...);
  std::string kernel_name =
      expt::get_kernel_name(std::forward<Params>(params)...);
  auto&& loop_body = expt::get_lambda(std::forward<Params>(params)...);
  expt::check_forall_optional_args(loop_body, f_params);

  util::PluginContext context {
      util::make_context<camp::decay<ExecPolicy>>(std::move(kernel_name))};
  util::callPreCapturePlugins(context);

  using RAJA::util::trigger_updates_before;
  auto body = trigger_updates_before(loop_body);

  util::callPostCapturePlugins(context);
  util::callPreLaunchPlugins(context);

  auto body_c = camp::decay<decltype(body)>(std::move(body));

  using Res = typename resources::get_resource<camp::decay<ExecPolicy>>::type;
  auto flat_body = detail::FornestFlattenedForallBody3D<LayoutTag, Seg0, Seg1,
                                                        Seg2, decltype(body_c)>(
      n0, n1, n2, seg0, seg1, seg2, std::move(body_c));
  auto e = wrap::forall(Res::get_default(), ExecPolicy {},
                        RAJA::TypedRangeSegment<std::size_t>(0, total),
                        std::move(flat_body), f_params);

  util::callPostLaunchPlugins(context);
  return e;
}

template<typename ExecPolicy,
         typename LayoutTag,
         typename Seg0,
         typename Seg1,
         typename Seg2,
         typename... Params>
  requires detail::segment_like<Seg2>
RAJA_INLINE auto fornest(
    typename resources::get_resource<camp::decay<ExecPolicy>>::type r,
    fornest_flattened_policy<ExecPolicy, LayoutTag>,
    Seg0 const& seg0,
    Seg1 const& seg1,
    Seg2 const& seg2,
    Params&&... params)
{
  const std::size_t n0    = detail::segment_length_host(seg0);
  const std::size_t n1    = detail::segment_length_host(seg1);
  const std::size_t n2    = detail::segment_length_host(seg2);
  const std::size_t total = n0 * n1 * n2;

  auto f_params = expt::make_forall_param_pack(std::forward<Params>(params)...);
  std::string kernel_name =
      expt::get_kernel_name(std::forward<Params>(params)...);
  auto&& loop_body = expt::get_lambda(std::forward<Params>(params)...);
  expt::check_forall_optional_args(loop_body, f_params);

  util::PluginContext context {
      util::make_context<camp::decay<ExecPolicy>>(std::move(kernel_name))};
  util::callPreCapturePlugins(context);

  using RAJA::util::trigger_updates_before;
  auto body = trigger_updates_before(loop_body);

  util::callPostCapturePlugins(context);
  util::callPreLaunchPlugins(context);

  auto body_c = camp::decay<decltype(body)>(std::move(body));

  auto flat_body = detail::FornestFlattenedForallBody3D<LayoutTag, Seg0, Seg1,
                                                        Seg2, decltype(body_c)>(
      n0, n1, n2, seg0, seg1, seg2, std::move(body_c));
  auto e = wrap::forall(r, ExecPolicy {},
                        RAJA::TypedRangeSegment<std::size_t>(0, total),
                        std::move(flat_body), f_params);

  util::callPostLaunchPlugins(context);
  return e;
}

//------------------------------------------------------------------------------
// fornest: non-dimensional exec policy (2D)
//   - host platform: flattened forall<ExecPolicy>
//   - device platform: nested/grid launch with default mapping + tiling
//------------------------------------------------------------------------------
template<typename ExecPolicy, typename Seg0, typename Seg1, typename... Params>
  requires(!detail::first_arg_is_segment_like_v<Params...>)
RAJA_INLINE auto fornest(ExecPolicy,
                         Seg0 const& seg0,
                         Seg1 const& seg1,
                         Params&&... params)
{
  static_assert(type_traits::policy_dimensionality_v<ExecPolicy> == 0,
                "RAJA::fornest v1 requires a non-dimensional execution policy "
                "(e.g., cuda_exec<256>) or a fornest_flattened_policy.");

  constexpr Platform platform =
      detail::get_platform<camp::decay<ExecPolicy>>::value;

  if constexpr (platform == Platform::host)
  {
    return RAJA::fornest(
        fornest_flattened_policy<ExecPolicy, RAJA::layout_right> {}, seg0, seg1,
        std::forward<Params>(params)...);
  }
#if defined(RAJA_GPU_ACTIVE)
  else
  {
    const std::size_t n0 = detail::segment_length_host(seg0);
    const std::size_t n1 = detail::segment_length_host(seg1);

    constexpr int block_size = detail::exec_block_size_v<ExecPolicy>;
    static_assert(block_size > 0,
                  "RAJA::fornest requires an execution policy with a fixed "
                  "block size on GPU");

    const detail::tile3 tile = detail::choose_2d_tile(block_size, n0, n1);

    const LaunchParams launch_params(
        Teams(static_cast<int>(detail::ceil_div<std::size_t>(n0, tile.x)),
              static_cast<int>(detail::ceil_div<std::size_t>(n1, tile.y))),
        Threads(tile.x, tile.y));

    auto&& user_body = expt::get_lambda(std::forward<Params>(params)...);
    auto body_c      = camp::decay<decltype(user_body)>(
        std::forward<decltype(user_body)>(user_body));

#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE)
    using launch_policy = detail::fornest_launch_policy_t<ExecPolicy>;
    using loop0 =
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::device_global_x_direct>;
    using loop1 =
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::device_global_y_direct>;

    auto seg0_c = seg0;
    auto seg1_c = seg1;
    auto launch_body =
        detail::FornestLaunchBody2D<RAJA::LaunchContext, loop0, loop1,
                                    decltype(seg0_c), decltype(seg1_c),
                                    decltype(body_c)>(seg0_c, seg1_c,
                                                      std::move(body_c));

    auto args_tuple = camp::forward_as_tuple(std::forward<Params>(params)...);
    return detail::apply_without_last(
        [&](auto&&... opt_args) {
          return RAJA::launch<launch_policy>(
              launch_params, std::forward<decltype(opt_args)>(opt_args)...,
              launch_body);
        },
        std::move(args_tuple));
#elif defined(RAJA_SYCL_ACTIVE)
    // SYCL does not support the CUDA/HIP-style explicit mapping tags used for
    // this launch path. Fall back to flattened execution with index
    // reconstruction, which is portable across device backends.
    return RAJA::fornest(
        fornest_flattened_policy<ExecPolicy, RAJA::layout_right> {}, seg0, seg1,
        std::forward<Params>(params)...);
#else
    RAJA_ABORT_OR_THROW("RAJA::fornest: unsupported device platform");
#endif
  }
#else
  else
  {
    RAJA_ABORT_OR_THROW("RAJA::fornest: device execution requested but GPU is "
                        "not enabled");
  }
#endif
}

//------------------------------------------------------------------------------
// fornest: non-dimensional exec policy (3D)
//------------------------------------------------------------------------------
template<typename ExecPolicy,
         typename Seg0,
         typename Seg1,
         typename Seg2,
         typename... Params>
  requires(detail::segment_like<Seg0> && detail::segment_like<Seg1> &&
           detail::segment_like<Seg2>)
RAJA_INLINE auto fornest(ExecPolicy,
                         Seg0 const& seg0,
                         Seg1 const& seg1,
                         Seg2 const& seg2,
                         Params&&... params)
{
  static_assert(type_traits::policy_dimensionality_v<ExecPolicy> == 0,
                "RAJA::fornest v1 requires a non-dimensional execution policy "
                "(e.g., cuda_exec<256>) or a fornest_flattened_policy.");

  constexpr Platform platform =
      detail::get_platform<camp::decay<ExecPolicy>>::value;

  if constexpr (platform == Platform::host)
  {
    return RAJA::fornest(
        fornest_flattened_policy<ExecPolicy, RAJA::layout_right> {}, seg0, seg1,
        seg2, std::forward<Params>(params)...);
  }
#if defined(RAJA_GPU_ACTIVE)
  else
  {
    const std::size_t n0 = detail::segment_length_host(seg0);
    const std::size_t n1 = detail::segment_length_host(seg1);
    const std::size_t n2 = detail::segment_length_host(seg2);

    constexpr int block_size = detail::exec_block_size_v<ExecPolicy>;
    static_assert(block_size > 0,
                  "RAJA::fornest requires an execution policy with a fixed "
                  "block size on GPU");

    const detail::tile3 tile = detail::choose_3d_tile(block_size, n0, n1, n2);

    const LaunchParams launch_params(
        Teams(static_cast<int>(detail::ceil_div<std::size_t>(n0, tile.x)),
              static_cast<int>(detail::ceil_div<std::size_t>(n1, tile.y)),
              static_cast<int>(detail::ceil_div<std::size_t>(n2, tile.z))),
        Threads(tile.x, tile.y, tile.z));

    auto&& user_body = expt::get_lambda(std::forward<Params>(params)...);
    auto body_c      = camp::decay<decltype(user_body)>(
        std::forward<decltype(user_body)>(user_body));

#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE)
    using launch_policy = detail::fornest_launch_policy_t<ExecPolicy>;
    using loop0 =
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::device_global_x_direct>;
    using loop1 =
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::device_global_y_direct>;
    using loop2 =
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::device_global_z_direct>;

    auto seg0_c = seg0;
    auto seg1_c = seg1;
    auto seg2_c = seg2;
    auto launch_body =
        detail::FornestLaunchBody3D<RAJA::LaunchContext, loop0, loop1, loop2,
                                    decltype(seg0_c), decltype(seg1_c),
                                    decltype(seg2_c), decltype(body_c)>(
            seg0_c, seg1_c, seg2_c, std::move(body_c));

    auto args_tuple = camp::forward_as_tuple(std::forward<Params>(params)...);
    return detail::apply_without_last(
        [&](auto&&... opt_args) {
          return RAJA::launch<launch_policy>(
              launch_params, std::forward<decltype(opt_args)>(opt_args)...,
              launch_body);
        },
        std::move(args_tuple));
#elif defined(RAJA_SYCL_ACTIVE)
    return RAJA::fornest(
        fornest_flattened_policy<ExecPolicy, RAJA::layout_right> {}, seg0, seg1,
        seg2, std::forward<Params>(params)...);
#else
    RAJA_ABORT_OR_THROW("RAJA::fornest: unsupported device platform");
#endif
  }
#else
  else
  {
    RAJA_ABORT_OR_THROW("RAJA::fornest: device execution requested but GPU is "
                        "not enabled");
  }
#endif
}

//------------------------------------------------------------------------------
// fornest: typed resource overloads (exec policy)
//------------------------------------------------------------------------------
template<typename ExecPolicy, typename Seg0, typename Seg1, typename... Params>
  requires(!detail::first_arg_is_segment_like_v<Params...>)
RAJA_INLINE auto fornest(
    typename resources::get_resource<camp::decay<ExecPolicy>>::type r,
    ExecPolicy pol,
    Seg0 const& seg0,
    Seg1 const& seg1,
    Params&&... params)
{
  RAJA_UNUSED_VAR(pol);
  constexpr Platform platform =
      detail::get_platform<camp::decay<ExecPolicy>>::value;

  if constexpr (platform == Platform::host)
  {
    return RAJA::fornest(
        r, fornest_flattened_policy<ExecPolicy, RAJA::layout_right> {}, seg0,
        seg1, std::forward<Params>(params)...);
  }
#if defined(RAJA_GPU_ACTIVE)
  else
  {
    const std::size_t n0 = detail::segment_length_host(seg0);
    const std::size_t n1 = detail::segment_length_host(seg1);

    constexpr int block_size = detail::exec_block_size_v<ExecPolicy>;
    static_assert(block_size > 0,
                  "RAJA::fornest requires an execution policy with a fixed "
                  "block size on GPU");

    const detail::tile3 tile = detail::choose_2d_tile(block_size, n0, n1);

    const LaunchParams launch_params(
        Teams(static_cast<int>(detail::ceil_div<std::size_t>(n0, tile.x)),
              static_cast<int>(detail::ceil_div<std::size_t>(n1, tile.y))),
        Threads(tile.x, tile.y));

    auto&& user_body = expt::get_lambda(std::forward<Params>(params)...);
    auto body_c      = camp::decay<decltype(user_body)>(
        std::forward<decltype(user_body)>(user_body));

#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE)
    using launch_policy = detail::fornest_launch_policy_t<ExecPolicy>;
    using loop0 =
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::device_global_x_direct>;
    using loop1 =
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::device_global_y_direct>;

    auto seg0_c = seg0;
    auto seg1_c = seg1;
    auto launch_body =
        detail::FornestLaunchBody2D<RAJA::LaunchContext, loop0, loop1,
                                    decltype(seg0_c), decltype(seg1_c),
                                    decltype(body_c)>(seg0_c, seg1_c,
                                                      std::move(body_c));

    auto args_tuple = camp::forward_as_tuple(std::forward<Params>(params)...);
    return detail::apply_without_last(
        [&](auto&&... opt_args) {
          return RAJA::launch<launch_policy>(
              r, launch_params, std::forward<decltype(opt_args)>(opt_args)...,
              launch_body);
        },
        std::move(args_tuple));
#elif defined(RAJA_SYCL_ACTIVE)
    return RAJA::fornest(
        r, fornest_flattened_policy<ExecPolicy, RAJA::layout_right> {}, seg0,
        seg1, std::forward<Params>(params)...);
#else
    RAJA_ABORT_OR_THROW("RAJA::fornest: unsupported device platform");
#endif
  }
#else
  else
  {
    RAJA_ABORT_OR_THROW("RAJA::fornest: device execution requested but GPU is "
                        "not enabled");
  }
#endif
}

template<typename ExecPolicy,
         typename Seg0,
         typename Seg1,
         typename Seg2,
         typename... Params>
  requires detail::segment_like<Seg2>
RAJA_INLINE auto fornest(
    typename resources::get_resource<camp::decay<ExecPolicy>>::type r,
    ExecPolicy pol,
    Seg0 const& seg0,
    Seg1 const& seg1,
    Seg2 const& seg2,
    Params&&... params)
{
  RAJA_UNUSED_VAR(pol);
  constexpr Platform platform =
      detail::get_platform<camp::decay<ExecPolicy>>::value;

  if constexpr (platform == Platform::host)
  {
    return RAJA::fornest(
        r, fornest_flattened_policy<ExecPolicy, RAJA::layout_right> {}, seg0,
        seg1, seg2, std::forward<Params>(params)...);
  }
#if defined(RAJA_GPU_ACTIVE)
  else
  {
    const std::size_t n0 = detail::segment_length_host(seg0);
    const std::size_t n1 = detail::segment_length_host(seg1);
    const std::size_t n2 = detail::segment_length_host(seg2);

    constexpr int block_size = detail::exec_block_size_v<ExecPolicy>;
    static_assert(block_size > 0,
                  "RAJA::fornest requires an execution policy with a fixed "
                  "block size on GPU");

    const detail::tile3 tile = detail::choose_3d_tile(block_size, n0, n1, n2);

    const LaunchParams launch_params(
        Teams(static_cast<int>(detail::ceil_div<std::size_t>(n0, tile.x)),
              static_cast<int>(detail::ceil_div<std::size_t>(n1, tile.y)),
              static_cast<int>(detail::ceil_div<std::size_t>(n2, tile.z))),
        Threads(tile.x, tile.y, tile.z));

    auto&& user_body = expt::get_lambda(std::forward<Params>(params)...);
    auto body_c      = camp::decay<decltype(user_body)>(
        std::forward<decltype(user_body)>(user_body));

#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE)
    using launch_policy = detail::fornest_launch_policy_t<ExecPolicy>;
    using loop0 =
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::device_global_x_direct>;
    using loop1 =
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::device_global_y_direct>;
    using loop2 =
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::device_global_z_direct>;

    auto seg0_c = seg0;
    auto seg1_c = seg1;
    auto seg2_c = seg2;
    auto launch_body =
        detail::FornestLaunchBody3D<RAJA::LaunchContext, loop0, loop1, loop2,
                                    decltype(seg0_c), decltype(seg1_c),
                                    decltype(seg2_c), decltype(body_c)>(
            seg0_c, seg1_c, seg2_c, std::move(body_c));

    auto args_tuple = camp::forward_as_tuple(std::forward<Params>(params)...);
    return detail::apply_without_last(
        [&](auto&&... opt_args) {
          return RAJA::launch<launch_policy>(
              r, launch_params, std::forward<decltype(opt_args)>(opt_args)...,
              launch_body);
        },
        std::move(args_tuple));
#elif defined(RAJA_SYCL_ACTIVE)
    return RAJA::fornest(
        r, fornest_flattened_policy<ExecPolicy, RAJA::layout_right> {}, seg0,
        seg1, seg2, std::forward<Params>(params)...);
#else
    RAJA_ABORT_OR_THROW("RAJA::fornest: unsupported device platform");
#endif
  }
#else
  else
  {
    RAJA_ABORT_OR_THROW("RAJA::fornest: device execution requested but GPU is "
                        "not enabled");
  }
#endif
}

//------------------------------------------------------------------------------
// fornest: typed-erased resource overloads (exec + flattened)
//------------------------------------------------------------------------------
template<typename Policy, typename Seg0, typename Seg1, typename... Params>
  requires(!detail::first_arg_is_segment_like_v<Params...>)
RAJA_INLINE auto fornest(RAJA::resources::Resource r,
                         Policy policy,
                         Seg0 const& seg0,
                         Seg1 const& seg1,
                         Params&&... params)
{
  auto typed = detail::require_typed_resource<Policy>(r);
  return RAJA::fornest(typed, policy, seg0, seg1,
                       std::forward<Params>(params)...);
}

template<typename Policy,
         typename Seg0,
         typename Seg1,
         typename Seg2,
         typename... Params>
  requires detail::segment_like<Seg2>
RAJA_INLINE auto fornest(RAJA::resources::Resource r,
                         Policy policy,
                         Seg0 const& seg0,
                         Seg1 const& seg1,
                         Seg2 const& seg2,
                         Params&&... params)
{
  auto typed = detail::require_typed_resource<Policy>(r);
  return RAJA::fornest(typed, policy, seg0, seg1, seg2,
                       std::forward<Params>(params)...);
}

//------------------------------------------------------------------------------
// fornest: dynamic policy selection overloads
//------------------------------------------------------------------------------
namespace detail
{
template<camp::idx_t IDX, typename POLICY_LIST>
struct fornest_dynamic_helper
{
  template<typename... Args>
  static void invoke(const int pol, Args&&... args)
  {
    if (IDX == pol)
    {
      using t_pol = typename camp::at<POLICY_LIST, camp::num<IDX>>::type;
      RAJA::fornest(t_pol {}, std::forward<Args>(args)...);
      return;
    }
    fornest_dynamic_helper<IDX - 1, POLICY_LIST>::invoke(
        pol, std::forward<Args>(args)...);
  }

  template<typename... Args>
  static resources::EventProxy<resources::Resource> invoke(
      RAJA::resources::Resource r,
      const int pol,
      Args&&... args)
  {
    using t_pol         = typename camp::at<POLICY_LIST, camp::num<IDX>>::type;
    using exec_pol      = fornest_underlying_exec_policy_t<t_pol>;
    using resource_type = typename resources::get_resource<exec_pol>::type;

    if (IDX == pol)
    {
      RAJA::fornest(r.get<resource_type>(), t_pol {},
                    std::forward<Args>(args)...);
      return {r};
    }

    return fornest_dynamic_helper<IDX - 1, POLICY_LIST>::invoke(
        r, pol, std::forward<Args>(args)...);
  }
};

template<typename POLICY_LIST>
struct fornest_dynamic_helper<0, POLICY_LIST>
{
  template<typename... Args>
  static void invoke(const int pol, Args&&... args)
  {
    if (0 == pol)
    {
      using t_pol = typename camp::at<POLICY_LIST, camp::num<0>>::type;
      RAJA::fornest(t_pol {}, std::forward<Args>(args)...);
      return;
    }
    RAJA_ABORT_OR_THROW("Policy enum not supported");
  }

  template<typename... Args>
  static resources::EventProxy<resources::Resource> invoke(
      RAJA::resources::Resource r,
      const int pol,
      Args&&... args)
  {
    if (pol != 0) RAJA_ABORT_OR_THROW("Policy value out of range");

    using t_pol         = typename camp::at<POLICY_LIST, camp::num<0>>::type;
    using exec_pol      = fornest_underlying_exec_policy_t<t_pol>;
    using resource_type = typename resources::get_resource<exec_pol>::type;

    RAJA::fornest(r.get<resource_type>(), t_pol {},
                  std::forward<Args>(args)...);
    return {r};
  }
};
}  // namespace detail

template<typename POLICY_LIST,
         typename... Args,
         std::enable_if_t<detail::is_camp_list_v<POLICY_LIST>, int> = 0>
RAJA_INLINE void fornest(const int pol, Args&&... args)
{
  constexpr int N = camp::size<POLICY_LIST>::value;
  static_assert(N > 0, "RAJA policy list must not be empty");
  if (pol > N - 1) RAJA_ABORT_OR_THROW("Policy value out of range");
  detail::fornest_dynamic_helper<N - 1, POLICY_LIST>::invoke(
      pol, std::forward<Args>(args)...);
}

template<typename POLICY_LIST,
         typename... Args,
         std::enable_if_t<detail::is_camp_list_v<POLICY_LIST>, int> = 0>
RAJA_INLINE resources::EventProxy<resources::Resource> fornest(
    RAJA::resources::Resource r,
    const int pol,
    Args&&... args)
{
  constexpr int N = camp::size<POLICY_LIST>::value;
  static_assert(N > 0, "RAJA policy list must not be empty");
  if (pol > N - 1) RAJA_ABORT_OR_THROW("Policy value out of range");
  return detail::fornest_dynamic_helper<N - 1, POLICY_LIST>::invoke(
      r, pol, std::forward<Args>(args)...);
}

}  // namespace RAJA

#endif  // RAJA_pattern_fornest_HPP
