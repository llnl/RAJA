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
#include "RAJA/util/StaticLayout.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/plugins.hpp"
#include "RAJA/util/types.hpp"
#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

// Odd dependecy with atomics is breaking CI builds
//#include "RAJA/util/View.hpp"

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

class LaunchContext
{
public:
  // Bump style allocator used to
  // get memory from the pool
  size_t shared_mem_offset;

  void* shared_mem_ptr;

#if defined(RAJA_SYCL_ACTIVE)
  // SGS ODR issue
  mutable ::sycl::nd_item<3>* itm;
#endif

  RAJA_HOST_DEVICE LaunchContext()
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

  /*
  //Odd dependecy with atomics is breaking CI builds
  template<typename T, size_t DIM, typename IDX_T=RAJA::Index_type, ptrdiff_t
  z_stride=DIM-1, typename arg, typename... args> RAJA_HOST_DEVICE auto
  getSharedMemoryView(size_t bytes, arg idx, args... idxs)
  {
    T * mem_ptr = &((T*) shared_mem_ptr)[shared_mem_offset];

    shared_mem_offset += bytes*sizeof(T);
    return RAJA::View<T, RAJA::Layout<DIM, IDX_T, z_stride>>(mem_ptr, idx,
  idxs...);
  }
  */

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
