/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing the core components of RAJA::launch
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

#ifndef RAJA_pattern_launch_core_HPP
#define RAJA_pattern_launch_core_HPP

#include "RAJA/config.hpp"
#include "RAJA/internal/get_platform.hpp"
#include "RAJA/pattern/launch/launch_context_policy.hpp"
#include "RAJA/util/StaticLayout.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/plugins.hpp"
#include "RAJA/util/types.hpp"

// Needed to provide a default indices/dims implementation for LaunchContext
// when compiling for GPU backends. The default launch context is used by
// existing examples and user code (e.g. RAJA::LaunchContext), but device-side
// index mappers require an indices/dims object.
#if defined(RAJA_HIP_ACTIVE)
#include "RAJA/policy/hip/policy.hpp"
#elif defined(RAJA_CUDA_ACTIVE)
#include "RAJA/policy/cuda/policy.hpp"
#endif

#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

#include <iterator>

#if defined(RAJA_GPU_DEVICE_COMPILE_PASS_ACTIVE) && !defined(RAJA_SYCL_ACTIVE)
#define RAJA_TEAM_SHARED __shared__
#else
#define RAJA_TEAM_SHARED
#endif

namespace RAJA
{

// GPU or CPU threads available
// strongly type the ExecPlace (guards agaist errors)
enum struct ExecPlace : int
{
  HOST,
  DEVICE,
  NUM_PLACES
};

struct null_launch_t
{};

// Support for host, and device
template<typename HOST_POLICY
#if defined(RAJA_GPU_ACTIVE)
         ,
         typename DEVICE_POLICY = HOST_POLICY
#endif
         >

struct LoopPolicy
{
  using host_policy_t = HOST_POLICY;
#if defined(RAJA_GPU_ACTIVE)
  using device_policy_t = DEVICE_POLICY;
#endif
};

template<typename HOST_POLICY
#if defined(RAJA_GPU_ACTIVE)
         ,
         typename DEVICE_POLICY = HOST_POLICY
#endif
         >
struct LaunchPolicy
{
  using host_policy_t = HOST_POLICY;
#if defined(RAJA_GPU_ACTIVE)
  using device_policy_t = DEVICE_POLICY;
#endif
};

template<typename... SEGMENTS>
struct MultiRange
{
  using tuple_type = camp::tuple<SEGMENTS...>;

  static constexpr camp::idx_t size = sizeof...(SEGMENTS);

  tuple_type segments;

  RAJA_HOST_DEVICE constexpr MultiRange(SEGMENTS const&... segs)
      : segments(segs...)
  {}

  template<camp::idx_t Idx>
  RAJA_HOST_DEVICE RAJA_INLINE auto const& get() const
  {
    return camp::get<Idx>(segments);
  }
};

template<typename... SEGMENTS>
RAJA_HOST_DEVICE RAJA_INLINE auto make_multi_range(SEGMENTS const&... segments)
{
  return MultiRange<SEGMENTS...>(segments...);
}

template<typename... LOOP_POLICIES>
struct PerfectLoopPolicy
{
  using host_policy_t = camp::list<typename LOOP_POLICIES::host_policy_t...>;
#if defined(RAJA_GPU_ACTIVE)
  using device_policy_t =
      camp::list<typename LOOP_POLICIES::device_policy_t...>;
#endif
};

template<camp::idx_t... Indices>
struct PerfectLoopInterchange
{
  using index_seq = camp::idx_seq<Indices...>;
};

struct Teams
{
  int value[3];

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr Teams() : value {1, 1, 1} {}

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr Teams(int i) : value {i, 1, 1} {}

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr Teams(int i, int j) : value {i, j, 1} {}

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr Teams(int i, int j, int k) : value {i, j, k} {}
};

struct Threads
{
  int value[3];

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr Threads() : value {1, 1, 1} {}

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr Threads(int i) : value {i, 1, 1} {}

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr Threads(int i, int j) : value {i, j, 1} {}

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr Threads(int i, int j, int k) : value {i, j, k} {}
};

struct Lanes
{
  int value;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr Lanes() : value(0) {}

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr Lanes(int i) : value(i) {}
};

struct LaunchParams
{
public:
  Teams teams;
  Threads threads;
  size_t shared_mem_size;

  RAJA_INLINE
  LaunchParams() = default;

  LaunchParams(Teams in_teams,
               Threads in_threads,
               size_t in_shared_mem_size = 0)
      : teams(in_teams),
        threads(in_threads),
        shared_mem_size(in_shared_mem_size) {};

private:
  RAJA_HOST_DEVICE

  RAJA_INLINE
  Teams apply(Teams const& a) { return (teams = a); }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  Threads apply(Threads const& a) { return (threads = a); }
};

class LaunchContextBase
{
public:
  // Bump style allocator used to
  // get memory from the pool
  size_t shared_mem_offset;
  void* shared_mem_ptr;

// In the future move this into a derived class.
#if defined(RAJA_SYCL_ACTIVE)
  // SGS ODR issue
  mutable ::sycl::nd_item<3>* itm;
#endif

  RAJA_HOST_DEVICE LaunchContextBase()
      : shared_mem_offset(0),
        shared_mem_ptr(nullptr)
  {}

  // TODO handle alignment
  template<typename T>
  RAJA_HOST_DEVICE T* getSharedMemory(size_t bytes)
  {

    // Calculate offset in bytes with a char pointer
    void* mem_ptr = static_cast<char*>(shared_mem_ptr) + shared_mem_offset;

    shared_mem_offset += bytes * sizeof(T);

    // convert to desired type
    return static_cast<T*>(mem_ptr);
  }

  RAJA_HOST_DEVICE void releaseSharedMemory()
  {
    // On the cpu/gpu we want to restart the count
    shared_mem_offset = 0;
  }

  RAJA_HOST_DEVICE
  void teamSync()
  {
    // SGS ODR Issue
#if defined(RAJA_GPU_DEVICE_COMPILE_PASS_ACTIVE) && defined(RAJA_SYCL_ACTIVE)
    itm->barrier(::sycl::access::fence_space::local_space);
#endif

#if defined(RAJA_GPU_DEVICE_COMPILE_PASS_ACTIVE) && !defined(RAJA_SYCL_ACTIVE)
    __syncthreads();
#endif
  }
};

template<>
class LaunchContextT<LaunchContextHostPolicy> : public LaunchContextBase
{
public:
  using LaunchContextBase::LaunchContextBase;
};

// Preserve backwards compatibility
#if defined(RAJA_HIP_ACTIVE)
using LaunchContext =
    LaunchContextT<HipLaunchContextNonCachedIndicesAndDimsPolicy>;
#elif defined(RAJA_CUDA_ACTIVE)
using LaunchContext =
    LaunchContextT<CudaLaunchContextNonCachedIndicesAndDimsPolicy>;
#else
using LaunchContext = LaunchContextT<LaunchContextHostPolicy>;
#endif

template<typename LAUNCH_POLICY>
struct LaunchExecute;

// Duplicate of code above on account that we need to support the case in which
// a kernel_name is not given
template<typename LAUNCH_POLICY, typename... ReduceParams>
void launch(LaunchParams const& launch_params,
            ReduceParams&&... rest_of_launch_args)
{
  // Get reducers
  auto reducers = expt::make_forall_param_pack(
      std::forward<ReduceParams>(rest_of_launch_args)...);

  // get kernel name
  std::string kernel_name =
      expt::get_kernel_name(std::forward<ReduceParams>(rest_of_launch_args)...);

  auto&& launch_body =
      expt::get_lambda(std::forward<ReduceParams>(rest_of_launch_args)...);

  // Take the first policy as we assume the second policy is not user defined.
  // We rely on the user to pair launch and loop policies correctly.
  util::PluginContext context {
      util::make_context<typename LAUNCH_POLICY::host_policy_t>(
          std::move(kernel_name))};
  util::callPreCapturePlugins(context);

  using RAJA::util::trigger_updates_before;
  auto p_body = trigger_updates_before(launch_body);

  util::callPostCapturePlugins(context);

  util::callPreLaunchPlugins(context);

  using launch_t = LaunchExecute<typename LAUNCH_POLICY::host_policy_t>;

  using Res = typename resources::get_resource<
      typename LAUNCH_POLICY::host_policy_t>::type;

  launch_t::exec(Res::get_default(), launch_params, p_body, reducers);

  util::callPostLaunchPlugins(context);
}

//=================================================
// Run time based policy launch
//=================================================
template<typename POLICY_LIST, typename BODY>
void launch(ExecPlace place, LaunchParams const& params, BODY const& body)
{
  launch<POLICY_LIST>(place, params, body);
}

// Run-time API for new reducer interface with support of the case without a new
// kernel name
template<typename POLICY_LIST, typename... ReduceParams>
void launch(ExecPlace place,
            const LaunchParams& launch_params,
            ReduceParams&&... rest_of_launch_args)
// BODY const &body)
{

  // Forward to single policy launch API - simplifies testing of plugins
  switch (place)
  {
    case ExecPlace::HOST:
    {
      using Res = typename resources::get_resource<
          typename POLICY_LIST::host_policy_t>::type;
      launch<LaunchPolicy<typename POLICY_LIST::host_policy_t>>(
          Res::get_default(), launch_params,
          std::forward<ReduceParams>(rest_of_launch_args)...);
      break;
    }
#if defined(RAJA_GPU_ACTIVE)
    case ExecPlace::DEVICE:
    {
      using Res = typename resources::get_resource<
          typename POLICY_LIST::device_policy_t>::type;
      launch<LaunchPolicy<typename POLICY_LIST::device_policy_t>>(
          Res::get_default(), launch_params,
          std::forward<ReduceParams>(rest_of_launch_args)...);
      break;
    }
#endif
    default:
      RAJA_ABORT_OR_THROW("Unknown launch place or device is not enabled");
  }
}


// Helper function to retrieve a resource based on the run-time policy - if a
// device is active
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP) ||                   \
    defined(RAJA_ENABLE_SYCL)
template<typename T, typename U>
RAJA::resources::Resource Get_Runtime_Resource(T host_res,
                                               U device_res,
                                               RAJA::ExecPlace device)
{
  if (device == RAJA::ExecPlace::DEVICE)
  {
    return RAJA::resources::Resource(device_res);
  }
  else
  {
    return RAJA::resources::Resource(host_res);
  }
}
#endif

template<typename T>
RAJA::resources::Resource Get_Host_Resource(T host_res, RAJA::ExecPlace device)
{
  if (device == RAJA::ExecPlace::DEVICE)
  {
    RAJA_ABORT_OR_THROW("Device is not enabled");
  }

  return RAJA::resources::Resource(host_res);
}

// Launch API which takes team resource struct and supports new reducers

// Duplicate of API above on account that we need to handle the case that a
// kernel name is not provided
template<typename POLICY_LIST, typename... ReduceParams>
resources::EventProxy<resources::Resource> launch(
    RAJA::resources::Resource res,
    LaunchParams const& launch_params,
    ReduceParams&&... rest_of_launch_args)
{

  // Get reducers
  auto reducers = expt::make_forall_param_pack(
      std::forward<ReduceParams>(rest_of_launch_args)...);

  std::string kernel_name =
      expt::get_kernel_name(std::forward<ReduceParams>(rest_of_launch_args)...);

  auto&& launch_body =
      expt::get_lambda(std::forward<ReduceParams>(rest_of_launch_args)...);

  ExecPlace place;
  if (res.get_platform() == RAJA::Platform::host)
  {
    place = RAJA::ExecPlace::HOST;
  }
  else
  {
    place = RAJA::ExecPlace::DEVICE;
  }

  //
  // Configure plugins
  //
#if defined(RAJA_GPU_ACTIVE)
  util::PluginContext context {
      place == ExecPlace::HOST
          ? util::make_context<typename POLICY_LIST::host_policy_t>(
                std::move(kernel_name))
          : util::make_context<typename POLICY_LIST::device_policy_t>(
                std::move(kernel_name))};
#else
  util::PluginContext context {
      util::make_context<typename POLICY_LIST::host_policy_t>(
          std::move(kernel_name))};
#endif

  util::callPreCapturePlugins(context);

  using RAJA::util::trigger_updates_before;
  auto p_body = trigger_updates_before(launch_body);

  util::callPostCapturePlugins(context);

  util::callPreLaunchPlugins(context);

  switch (place)
  {
    case ExecPlace::HOST:
    {
      using launch_t = LaunchExecute<typename POLICY_LIST::host_policy_t>;
      resources::EventProxy<resources::Resource> e_proxy =
          launch_t::exec(res, launch_params, p_body, reducers);
      util::callPostLaunchPlugins(context);
      return e_proxy;
    }
#if defined(RAJA_GPU_ACTIVE)
    case ExecPlace::DEVICE:
    {
      using launch_t = LaunchExecute<typename POLICY_LIST::device_policy_t>;
      resources::EventProxy<resources::Resource> e_proxy =
          launch_t::exec(res, launch_params, p_body, reducers);
      util::callPostLaunchPlugins(context);
      return e_proxy;
    }
#endif
    default:
    {
      RAJA_ABORT_OR_THROW("Unknown launch place or device is not enabled");
    }
  }

  RAJA_ABORT_OR_THROW("Unknown launch place");

  //^^ RAJA will abort before getting here
  return resources::EventProxy<resources::Resource>(res);
}

template<typename POLICY_LIST>
#if defined(RAJA_GPU_DEVICE_COMPILE_PASS_ACTIVE)
using loop_policy = typename POLICY_LIST::device_policy_t;
#else
using loop_policy = typename POLICY_LIST::host_policy_t;
#endif

template<typename POLICY, typename SEGMENT>
struct LoopExecute;

template<typename POLICY, typename SEGMENT>
struct LoopICountExecute;

namespace launch_detail
{

template<typename ORDER, camp::idx_t N>
struct perfect_loop_order
{
  using type = camp::make_idx_seq_t<N>;
};

template<camp::idx_t... Indices, camp::idx_t N>
struct perfect_loop_order<PerfectLoopInterchange<Indices...>, N>
{
  using type = camp::idx_seq<Indices...>;
};

template<camp::idx_t I, typename INDEX_SEQ>
struct idx_seq_at;

template<camp::idx_t I, camp::idx_t Head, camp::idx_t... Tail>
struct idx_seq_at<I, camp::idx_seq<Head, Tail...>>
    : idx_seq_at<I - 1, camp::idx_seq<Tail...>>
{};

template<camp::idx_t Head, camp::idx_t... Tail>
struct idx_seq_at<0, camp::idx_seq<Head, Tail...>> : camp::num<Head>
{};

template<camp::idx_t Want, camp::idx_t Pos, typename INDEX_SEQ>
struct idx_seq_find;

template<camp::idx_t Want, camp::idx_t Pos, camp::idx_t Head, camp::idx_t... Tail>
struct idx_seq_find<Want, Pos, camp::idx_seq<Head, Tail...>>
    : camp::num<(Head == Want
                     ? Pos
                     : idx_seq_find<Want, Pos + 1, camp::idx_seq<Tail...>>::value)>
{};

template<camp::idx_t Want, camp::idx_t Pos, camp::idx_t Head>
struct idx_seq_find<Want, Pos, camp::idx_seq<Head>>
    : camp::num<(Head == Want ? Pos : -1)>
{};

template<typename BODY, typename ARG_TUPLE, camp::idx_t... Indices>
RAJA_HOST_DEVICE RAJA_INLINE void invoke_with_tuple_args(
    BODY const& body,
    ARG_TUPLE const& args,
    camp::idx_seq<Indices...>)
{
  body(camp::get<Indices>(args)...);
}

template<typename BODY, typename ARG_TUPLE>
RAJA_HOST_DEVICE RAJA_INLINE void invoke_with_tuple_args(BODY const& body,
                                                         ARG_TUPLE const& args)
{
  using arg_tuple_t = camp::decay<ARG_TUPLE>;

  invoke_with_tuple_args(
      body, args, camp::make_idx_seq_t<camp::tuple_size<arg_tuple_t>::value> {});
}

template<typename ORDER_SEQ,
         typename BODY,
         typename ARG_TUPLE,
         camp::idx_t... Indices>
RAJA_HOST_DEVICE RAJA_INLINE void invoke_with_ordered_tuple_args(
    BODY const& body,
    ARG_TUPLE const& args,
    camp::idx_seq<Indices...>)
{
  body(camp::get<idx_seq_find<Indices, 0, ORDER_SEQ>::value>(args)...);
}

template<typename ORDER_SEQ, typename BODY, typename ARG_TUPLE>
RAJA_HOST_DEVICE RAJA_INLINE void invoke_with_ordered_tuple_args(
    BODY const& body,
    ARG_TUPLE const& args)
{
  using arg_tuple_t = camp::decay<ARG_TUPLE>;

  invoke_with_ordered_tuple_args<ORDER_SEQ>(
      body, args, camp::make_idx_seq_t<camp::tuple_size<arg_tuple_t>::value> {});
}

template<typename ORDER_SEQ,
         typename BODY,
         typename VALUE_TUPLE,
         typename INDEX_TUPLE,
         camp::idx_t... Indices>
RAJA_HOST_DEVICE RAJA_INLINE void invoke_with_ordered_icount_args(
    BODY const& body,
    VALUE_TUPLE const& values,
    INDEX_TUPLE const& indices,
    camp::idx_seq<Indices...>)
{
  body(camp::get<idx_seq_find<Indices, 0, ORDER_SEQ>::value>(values)...,
       camp::get<idx_seq_find<Indices, 0, ORDER_SEQ>::value>(indices)...);
}

template<typename ORDER_SEQ, typename BODY, typename VALUE_TUPLE, typename INDEX_TUPLE>
RAJA_HOST_DEVICE RAJA_INLINE void invoke_with_ordered_icount_args(
    BODY const& body,
    VALUE_TUPLE const& values,
    INDEX_TUPLE const& indices)
{
  using value_tuple_t = camp::decay<VALUE_TUPLE>;

  invoke_with_ordered_icount_args<ORDER_SEQ>(
      body,
      values,
      indices,
      camp::make_idx_seq_t<camp::tuple_size<value_tuple_t>::value> {});
}

template<camp::idx_t I,
         camp::idx_t N,
         typename ORDER_SEQ,
         typename EXEC_POLICY_LIST,
         typename CONTEXT,
         typename RANGE_TUPLE,
         typename VALUE_TUPLE,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void perfect_loop_impl(camp::num<I>,
                                                    CONTEXT const& ctx,
                                                    RANGE_TUPLE const& ranges,
                                                    VALUE_TUPLE const& values,
                                                    BODY const& body)
{
  if constexpr (I == N)
  {
    invoke_with_ordered_tuple_args<ORDER_SEQ>(body, values);
  }
  else
  {
    using range_tuple_t = camp::decay<RANGE_TUPLE>;
    static constexpr camp::idx_t order_idx = idx_seq_at<I, ORDER_SEQ>::value;
    using exec_policy =
        typename camp::at<EXEC_POLICY_LIST, camp::num<order_idx>>::type;
    using segment_t = typename camp::tuple_element<order_idx, range_tuple_t>::type;
    using value_t =
        typename std::iterator_traits<typename segment_t::iterator>::value_type;

    auto const& segment = camp::get<order_idx>(ranges);

    LoopExecute<exec_policy, segment_t>::exec(
        ctx, segment, [=](value_t value) {
          auto next_values =
              camp::tuple_cat_pair(values, camp::make_tuple(value));
          perfect_loop_impl<camp::idx_t(I + 1), N, ORDER_SEQ, EXEC_POLICY_LIST>(
              camp::num<I + 1> {}, ctx, ranges, next_values, body);
        });
  }
}

template<camp::idx_t I,
         camp::idx_t N,
         typename ORDER_SEQ,
         typename EXEC_POLICY_LIST,
         typename CONTEXT,
         typename RANGE_TUPLE,
         typename VALUE_TUPLE,
         typename INDEX_TUPLE,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void perfect_loop_icount_impl(
    camp::num<I>,
    CONTEXT const& ctx,
    RANGE_TUPLE const& ranges,
    VALUE_TUPLE const& values,
    INDEX_TUPLE const& indices,
    BODY const& body)
{
  if constexpr (I == N)
  {
    invoke_with_ordered_icount_args<ORDER_SEQ>(body, values, indices);
  }
  else
  {
    using range_tuple_t = camp::decay<RANGE_TUPLE>;
    static constexpr camp::idx_t order_idx = idx_seq_at<I, ORDER_SEQ>::value;
    using exec_policy =
        typename camp::at<EXEC_POLICY_LIST, camp::num<order_idx>>::type;
    using segment_t = typename camp::tuple_element<order_idx, range_tuple_t>::type;
    using value_t =
        typename std::iterator_traits<typename segment_t::iterator>::value_type;
    using index_t = typename std::iterator_traits<
        typename segment_t::iterator>::difference_type;

    auto const& segment = camp::get<order_idx>(ranges);

    LoopICountExecute<exec_policy, segment_t>::exec(
        ctx, segment, [=](value_t value, index_t idx) {
          auto next_values =
              camp::tuple_cat_pair(values, camp::make_tuple(value));
          auto next_indices =
              camp::tuple_cat_pair(indices, camp::make_tuple(idx));
          perfect_loop_icount_impl<camp::idx_t(I + 1),
                                   N,
                                   ORDER_SEQ,
                                   EXEC_POLICY_LIST>(camp::num<I + 1> {},
                                                     ctx,
                                                     ranges,
                                                     next_values,
                                                     next_indices,
                                                     body);
        });
  }
}

}  // namespace launch_detail

RAJA_SUPPRESS_HD_WARN
template<typename POLICY_LIST,
         typename CONTEXT,
         typename SEGMENT,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void loop(CONTEXT const& ctx,
                                       SEGMENT const& segment,
                                       BODY const& body)
{

  LoopExecute<loop_policy<POLICY_LIST>, SEGMENT>::exec(ctx, segment, body);
}

template<typename POLICY_LIST,
         typename CONTEXT,
         typename SEGMENT,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void loop_icount(CONTEXT const& ctx,
                                              SEGMENT const& segment,
                                              BODY const& body)
{

  LoopICountExecute<loop_policy<POLICY_LIST>, SEGMENT>::exec(ctx, segment,
                                                             body);
}

template<typename POLICY_LIST,
         typename ORDER = void,
         typename CONTEXT,
         typename... SEGMENTS,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void perfect_loop(
    CONTEXT const& ctx,
    MultiRange<SEGMENTS...> const& ranges,
    BODY const& body)
{
  using exec_policies = loop_policy<POLICY_LIST>;
  using order_seq =
      typename launch_detail::perfect_loop_order<ORDER, sizeof...(SEGMENTS)>::type;

  static_assert(camp::size<exec_policies>::value == sizeof...(SEGMENTS),
                "RAJA::perfect_loop requires one loop policy per segment.");
  static_assert(camp::size<order_seq>::value == sizeof...(SEGMENTS),
                "RAJA::perfect_loop interchange must match the segment count.");

  launch_detail::perfect_loop_impl<0,
                                   sizeof...(SEGMENTS),
                                   order_seq,
                                   exec_policies>(camp::num<0> {},
                                                  ctx,
                                                  ranges.segments,
                                                  camp::make_tuple(),
                                                  body);
}

template<typename POLICY_LIST,
         typename ORDER = void,
         typename CONTEXT,
         typename SEGMENT0,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void perfect_loop(CONTEXT const& ctx,
                                               SEGMENT0 const& segment0,
                                               BODY const& body)
{
  perfect_loop<POLICY_LIST, ORDER>(ctx, make_multi_range(segment0), body);
}

template<typename POLICY_LIST,
         typename ORDER = void,
         typename CONTEXT,
         typename SEGMENT0,
         typename SEGMENT1,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void perfect_loop(CONTEXT const& ctx,
                                               SEGMENT0 const& segment0,
                                               SEGMENT1 const& segment1,
                                               BODY const& body)
{
  perfect_loop<POLICY_LIST, ORDER>(
      ctx, make_multi_range(segment0, segment1), body);
}

template<typename POLICY_LIST,
         typename ORDER = void,
         typename CONTEXT,
         typename SEGMENT0,
         typename SEGMENT1,
         typename SEGMENT2,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void perfect_loop(CONTEXT const& ctx,
                                               SEGMENT0 const& segment0,
                                               SEGMENT1 const& segment1,
                                               SEGMENT2 const& segment2,
                                               BODY const& body)
{
  perfect_loop<POLICY_LIST, ORDER>(
      ctx, make_multi_range(segment0, segment1, segment2), body);
}

template<typename POLICY_LIST,
         typename ORDER = void,
         typename CONTEXT,
         typename SEGMENT0,
         typename SEGMENT1,
         typename SEGMENT2,
         typename SEGMENT3,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void perfect_loop(CONTEXT const& ctx,
                                               SEGMENT0 const& segment0,
                                               SEGMENT1 const& segment1,
                                               SEGMENT2 const& segment2,
                                               SEGMENT3 const& segment3,
                                               BODY const& body)
{
  perfect_loop<POLICY_LIST, ORDER>(
      ctx, make_multi_range(segment0, segment1, segment2, segment3), body);
}

template<typename POLICY_LIST,
         typename ORDER = void,
         typename CONTEXT,
         typename SEGMENT0,
         typename SEGMENT1,
         typename SEGMENT2,
         typename SEGMENT3,
         typename SEGMENT4,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void perfect_loop(CONTEXT const& ctx,
                                               SEGMENT0 const& segment0,
                                               SEGMENT1 const& segment1,
                                               SEGMENT2 const& segment2,
                                               SEGMENT3 const& segment3,
                                               SEGMENT4 const& segment4,
                                               BODY const& body)
{
  perfect_loop<POLICY_LIST, ORDER>(
      ctx,
      make_multi_range(segment0, segment1, segment2, segment3, segment4),
      body);
}

template<typename POLICY_LIST,
         typename ORDER = void,
         typename CONTEXT,
         typename SEGMENT0,
         typename SEGMENT1,
         typename SEGMENT2,
         typename SEGMENT3,
         typename SEGMENT4,
         typename SEGMENT5,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void perfect_loop(CONTEXT const& ctx,
                                               SEGMENT0 const& segment0,
                                               SEGMENT1 const& segment1,
                                               SEGMENT2 const& segment2,
                                               SEGMENT3 const& segment3,
                                               SEGMENT4 const& segment4,
                                               SEGMENT5 const& segment5,
                                               BODY const& body)
{
  perfect_loop<POLICY_LIST, ORDER>(
      ctx,
      make_multi_range(
          segment0, segment1, segment2, segment3, segment4, segment5),
      body);
}

template<typename POLICY_LIST,
         typename ORDER = void,
         typename CONTEXT,
         typename... SEGMENTS,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void perfect_loop_icount(
    CONTEXT const& ctx,
    MultiRange<SEGMENTS...> const& ranges,
    BODY const& body)
{
  using exec_policies = loop_policy<POLICY_LIST>;
  using order_seq =
      typename launch_detail::perfect_loop_order<ORDER, sizeof...(SEGMENTS)>::type;

  static_assert(camp::size<exec_policies>::value == sizeof...(SEGMENTS),
                "RAJA::perfect_loop_icount requires one loop policy per "
                "segment.");
  static_assert(camp::size<order_seq>::value == sizeof...(SEGMENTS),
                "RAJA::perfect_loop_icount interchange must match the segment "
                "count.");

  launch_detail::perfect_loop_icount_impl<0,
                                          sizeof...(SEGMENTS),
                                          order_seq,
                                          exec_policies>(camp::num<0> {},
                                                         ctx,
                                                         ranges.segments,
                                                         camp::make_tuple(),
                                                         camp::make_tuple(),
                                                         body);
}

template<typename POLICY_LIST,
         typename ORDER = void,
         typename CONTEXT,
         typename SEGMENT0,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void perfect_loop_icount(CONTEXT const& ctx,
                                                      SEGMENT0 const& segment0,
                                                      BODY const& body)
{
  perfect_loop_icount<POLICY_LIST, ORDER>(
      ctx, make_multi_range(segment0), body);
}

template<typename POLICY_LIST,
         typename ORDER = void,
         typename CONTEXT,
         typename SEGMENT0,
         typename SEGMENT1,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void perfect_loop_icount(CONTEXT const& ctx,
                                                      SEGMENT0 const& segment0,
                                                      SEGMENT1 const& segment1,
                                                      BODY const& body)
{
  perfect_loop_icount<POLICY_LIST, ORDER>(
      ctx, make_multi_range(segment0, segment1), body);
}

template<typename POLICY_LIST,
         typename ORDER = void,
         typename CONTEXT,
         typename SEGMENT0,
         typename SEGMENT1,
         typename SEGMENT2,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void perfect_loop_icount(CONTEXT const& ctx,
                                                      SEGMENT0 const& segment0,
                                                      SEGMENT1 const& segment1,
                                                      SEGMENT2 const& segment2,
                                                      BODY const& body)
{
  perfect_loop_icount<POLICY_LIST, ORDER>(
      ctx, make_multi_range(segment0, segment1, segment2), body);
}

template<typename POLICY_LIST,
         typename ORDER = void,
         typename CONTEXT,
         typename SEGMENT0,
         typename SEGMENT1,
         typename SEGMENT2,
         typename SEGMENT3,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void perfect_loop_icount(CONTEXT const& ctx,
                                                      SEGMENT0 const& segment0,
                                                      SEGMENT1 const& segment1,
                                                      SEGMENT2 const& segment2,
                                                      SEGMENT3 const& segment3,
                                                      BODY const& body)
{
  perfect_loop_icount<POLICY_LIST, ORDER>(
      ctx, make_multi_range(segment0, segment1, segment2, segment3), body);
}

template<typename POLICY_LIST,
         typename ORDER = void,
         typename CONTEXT,
         typename SEGMENT0,
         typename SEGMENT1,
         typename SEGMENT2,
         typename SEGMENT3,
         typename SEGMENT4,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void perfect_loop_icount(CONTEXT const& ctx,
                                                      SEGMENT0 const& segment0,
                                                      SEGMENT1 const& segment1,
                                                      SEGMENT2 const& segment2,
                                                      SEGMENT3 const& segment3,
                                                      SEGMENT4 const& segment4,
                                                      BODY const& body)
{
  perfect_loop_icount<POLICY_LIST, ORDER>(
      ctx,
      make_multi_range(segment0, segment1, segment2, segment3, segment4),
      body);
}

template<typename POLICY_LIST,
         typename ORDER = void,
         typename CONTEXT,
         typename SEGMENT0,
         typename SEGMENT1,
         typename SEGMENT2,
         typename SEGMENT3,
         typename SEGMENT4,
         typename SEGMENT5,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void perfect_loop_icount(CONTEXT const& ctx,
                                                      SEGMENT0 const& segment0,
                                                      SEGMENT1 const& segment1,
                                                      SEGMENT2 const& segment2,
                                                      SEGMENT3 const& segment3,
                                                      SEGMENT4 const& segment4,
                                                      SEGMENT5 const& segment5,
                                                      BODY const& body)
{
  perfect_loop_icount<POLICY_LIST, ORDER>(
      ctx,
      make_multi_range(
          segment0, segment1, segment2, segment3, segment4, segment5),
      body);
}

namespace expt
{

RAJA_SUPPRESS_HD_WARN
template<typename POLICY_LIST,
         typename CONTEXT,
         typename SEGMENT,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void loop(CONTEXT const& ctx,
                                       SEGMENT const& segment0,
                                       SEGMENT const& segment1,
                                       BODY const& body)
{

  LoopExecute<loop_policy<POLICY_LIST>, SEGMENT>::exec(ctx, segment0, segment1,
                                                       body);
}

RAJA_SUPPRESS_HD_WARN
template<typename POLICY_LIST,
         typename CONTEXT,
         typename SEGMENT,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void loop_icount(CONTEXT const& ctx,
                                              SEGMENT const& segment0,
                                              SEGMENT const& segment1,
                                              BODY const& body)
{

  LoopICountExecute<loop_policy<POLICY_LIST>, SEGMENT>::exec(ctx, segment0,
                                                             segment1, body);
}

RAJA_SUPPRESS_HD_WARN
template<typename POLICY_LIST,
         typename CONTEXT,
         typename SEGMENT,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void loop(CONTEXT const& ctx,
                                       SEGMENT const& segment0,
                                       SEGMENT const& segment1,
                                       SEGMENT const& segment2,
                                       BODY const& body)
{

  LoopExecute<loop_policy<POLICY_LIST>, SEGMENT>::exec(ctx, segment0, segment1,
                                                       segment2, body);
}

RAJA_SUPPRESS_HD_WARN
template<typename POLICY_LIST,
         typename CONTEXT,
         typename SEGMENT,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void loop_icount(CONTEXT const& ctx,
                                              SEGMENT const& segment0,
                                              SEGMENT const& segment1,
                                              SEGMENT const& segment2,
                                              BODY const& body)
{

  LoopICountExecute<loop_policy<POLICY_LIST>, SEGMENT>::exec(
      ctx, segment0, segment1, segment2, body);
}

}  // namespace expt

template<typename POLICY, typename SEGMENT>
struct TileExecute;

template<typename POLICY, typename SEGMENT>
struct TileTCountExecute;

template<typename POLICY_LIST,
         typename CONTEXT,
         typename TILE_T,
         typename SEGMENT,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void tile(CONTEXT const& ctx,
                                       TILE_T tile_size,
                                       SEGMENT const& segment,
                                       BODY const& body)
{

  TileExecute<loop_policy<POLICY_LIST>, SEGMENT>::exec(ctx, tile_size, segment,
                                                       body);
}

template<typename POLICY_LIST,
         typename CONTEXT,
         typename TILE_T,
         typename SEGMENT,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void tile_tcount(CONTEXT const& ctx,
                                              TILE_T tile_size,
                                              SEGMENT const& segment,
                                              BODY const& body)
{
  TileTCountExecute<loop_policy<POLICY_LIST>, SEGMENT>::exec(ctx, tile_size,
                                                             segment, body);
}

namespace expt
{

template<typename POLICY_LIST,
         typename CONTEXT,
         typename TILE_T,
         typename SEGMENT,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void tile(CONTEXT const& ctx,
                                       TILE_T tile_size0,
                                       TILE_T tile_size1,
                                       SEGMENT const& segment0,
                                       SEGMENT const& segment1,
                                       BODY const& body)
{

  TileExecute<loop_policy<POLICY_LIST>, SEGMENT>::exec(
      ctx, tile_size0, tile_size1, segment0, segment1, body);
}

template<typename POLICY_LIST,
         typename CONTEXT,
         typename TILE_T,
         typename SEGMENT,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void tile_tcount(CONTEXT const& ctx,
                                              TILE_T tile_size0,
                                              TILE_T tile_size1,
                                              SEGMENT const& segment0,
                                              SEGMENT const& segment1,
                                              BODY const& body)
{

  TileTCountExecute<loop_policy<POLICY_LIST>, SEGMENT>::exec(
      ctx, tile_size0, tile_size1, segment0, segment1, body);
}

template<typename POLICY_LIST,
         typename CONTEXT,
         typename TILE_T,
         typename SEGMENT,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void tile(CONTEXT const& ctx,
                                       TILE_T tile_size0,
                                       TILE_T tile_size1,
                                       TILE_T tile_size2,
                                       SEGMENT const& segment0,
                                       SEGMENT const& segment1,
                                       SEGMENT const& segment2,
                                       BODY const& body)
{

  TileExecute<loop_policy<POLICY_LIST>, SEGMENT>::exec(
      ctx, tile_size0, tile_size1, tile_size2, segment0, segment1, segment2,
      body);
}

template<typename POLICY_LIST,
         typename CONTEXT,
         typename TILE_T,
         typename SEGMENT,
         typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void tile_tcount(CONTEXT const& ctx,
                                              TILE_T tile_size0,
                                              TILE_T tile_size1,
                                              TILE_T tile_size2,
                                              SEGMENT const& segment0,
                                              SEGMENT const& segment1,
                                              SEGMENT const& segment2,
                                              BODY const& body)
{

  TileTCountExecute<loop_policy<POLICY_LIST>, SEGMENT>::exec(
      ctx, tile_size0, tile_size1, tile_size2, segment0, segment1, segment2,
      body);
}

}  // namespace expt

}  // namespace RAJA
#endif
