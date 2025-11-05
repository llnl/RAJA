/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA workgroup Dispatcher.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_hip_WorkGroup_Dispatcher_HPP
#define RAJA_hip_WorkGroup_Dispatcher_HPP

#include "RAJA/config.hpp"

#include "camp/resource.hpp"

#include "RAJA/policy/hip/policy.hpp"

#include "RAJA/pattern/WorkGroup/Dispatcher.hpp"

#include <thread>
#include <mutex>

namespace RAJA
{

namespace detail
{

namespace hip
{

// global function that creates the value on the device using the
// factory and writes it into a pinned ptr
template<typename Factory>
__global__ void get_value_global(typename Factory::value_type* ptr,
                                 Factory factory)
{
  *ptr = factory();
}

// get the pinned ptr buffer
inline void* get_cached_value_ptr(size_t nbytes)
{
  static size_t cached_nbytes = 0;
  static void* ptr            = nullptr;
  if (nbytes > cached_nbytes)
  {
    cached_nbytes = 0;
    CAMP_HIP_API_INVOKE_AND_CHECK(hipHostFree, ptr);
    CAMP_HIP_API_INVOKE_AND_CHECK(hipHostMalloc, &ptr, nbytes);
    cached_nbytes = nbytes;
  }
  return ptr;
}

// mutex that guards against concurrent use of
// pinned buffer and get_cached_value_ptr()
inline std::mutex& get_value_mutex()
{
  static std::mutex s_mutex;
  return s_mutex;
}

// get the device function pointer by calling a global function to
// write it into a pinned ptr, beware different instantiates of this
// function may run concurrently
template<typename Factory>
inline auto get_value(Factory&& factory)
{
  using value_type = typename std::decay_t<Factory>::value_type;
  const std::lock_guard<std::mutex> lock(get_value_mutex());

  auto res = ::camp::resources::Hip::get_default();
  auto ptr = static_cast<value_type*>(get_cached_value_ptr(sizeof(value_type)));
  auto func =
      reinterpret_cast<const void*>(&get_value_global<std::decay_t<Factory>>);
  void* args[] = {(void*)&ptr, (void*)&factory};
  CAMP_HIP_API_INVOKE_AND_CHECK(hipLaunchKernel, func, 1, 1, args, 0,
                                res.get_stream());
  CAMP_HIP_API_INVOKE_AND_CHECK(hipStreamSynchronize, res.get_stream());

  return *ptr;
}

// get the device function pointer and store it so it can be used
// multiple times
template<typename Factory>
inline auto get_cached_value(Factory&& factory)
{
  static auto value = get_value(std::forward<Factory>(factory));
  return value;
}

}  // namespace hip

/*!
 * Populate and return a Dispatcher object that can be used in device code
 */
template<typename T, typename Dispatcher_T, size_t BLOCK_SIZE, bool Async>
inline const Dispatcher_T* get_Dispatcher(hip_work<BLOCK_SIZE, Async> const&)
{
  static Dispatcher_T dispatcher {
      Dispatcher_T::template makeDispatcher<T>([](auto&& factory) {
        return hip::get_cached_value(std::forward<decltype(factory)>(factory));
      })};
  return &dispatcher;
}

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
