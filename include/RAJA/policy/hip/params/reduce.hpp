//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef NEW_REDUCE_HIP_REDUCE_HPP
#define NEW_REDUCE_HIP_REDUCE_HPP

#if defined(RAJA_HIP_ACTIVE)

#include <hip/hip_runtime.h>
#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#include "RAJA/policy/hip/reduce.hpp"
#include "RAJA/pattern/params/reducer.hpp"

#include <type_traits>

namespace RAJA
{
namespace expt
{
namespace detail
{

// A single last block combines every partial result in the generic path.
// For double-precision addition, distribute block atomics over several
// independent slots instead. The sum order may vary between launches. Keep
// other operators on the existing path.
template<typename OP, typename T>
struct use_atomic_plus : std::false_type
{};

template<>
struct use_atomic_plus<RAJA::operators::plus<double, double, double>, double>
    : std::true_type
{};

constexpr unsigned int atomic_plus_slots = 64;

// Init
template<typename EXEC_POL, typename OP, typename T, typename VOp>
camp::concepts::enable_if<RAJA::type_traits::is_hip_policy<EXEC_POL>>
param_init(EXEC_POL const&,
           Reducer<OP, T, VOp>& red,
           RAJA::hip::detail::hipInfo& hi)
{
  if constexpr (use_atomic_plus<OP, T>::value)
  {
    red.devicetarget =
        RAJA::hip::device_mempool_type::getInstance().template malloc<T>(
            atomic_plus_slots);
    CAMP_HIP_API_INVOKE_AND_CHECK(hipMemsetAsync, red.devicetarget, 0,
                                  sizeof(T) * atomic_plus_slots,
                                  hi.res.get_stream());
  }
  else
  {
    red.devicetarget =
        RAJA::hip::pinned_mempool_type::getInstance().template malloc<T>(1);
    red.device_mem.allocate(hi.gridDim.x * hi.gridDim.y * hi.gridDim.z);
    red.device_count = RAJA::hip::device_zeroed_mempool_type::getInstance()
                           .template malloc<unsigned int>(1);
  }
}

// Combine
template<typename EXEC_POL, typename OP, typename T, typename VOp>
RAJA_HOST_DEVICE camp::concepts::enable_if<
    RAJA::type_traits::is_hip_policy<EXEC_POL>>
param_combine(EXEC_POL const&, Reducer<OP, T, VOp>& red)
{
  if constexpr (use_atomic_plus<OP, T>::value)
  {
    using IterationGetter = typename EXEC_POL::IterationGetter;
    using BlockGetter =
        typename RAJA::hip::get_index_block<IterationGetter>::type;
    using ThreadGetter =
        typename RAJA::hip::get_index_thread<IterationGetter>::type;
    T block_value = RAJA::hip::impl::expt::block_reduce<ThreadGetter, OP>(
        red.getVal(), OP::identity());
    if (ThreadGetter::index() == 0 && block_value != OP::identity())
    {
      RAJA::atomicAdd(RAJA::hip_atomic {},
                      red.devicetarget +
                          BlockGetter::index() % atomic_plus_slots,
                      block_value);
    }
  }
  else
  {
    RAJA::hip::impl::expt::grid_reduce<typename EXEC_POL::IterationGetter, OP>(
        red.devicetarget, red.getVal(), red.device_mem, red.device_count);
  }
}

// Resolve
template<typename EXEC_POL, typename OP, typename T, typename VOp>
camp::concepts::enable_if<RAJA::type_traits::is_hip_policy<EXEC_POL>>
param_resolve(EXEC_POL const&,
              Reducer<OP, T, VOp>& red,
              RAJA::hip::detail::hipInfo& hi)
{
  if constexpr (use_atomic_plus<OP, T>::value)
  {
    T partials[atomic_plus_slots];
    CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpyAsync, partials, red.devicetarget,
                                  sizeof(partials), hipMemcpyDeviceToHost,
                                  hi.res.get_stream());
    hi.res.wait();

    T result = OP::identity();
    for (unsigned int i = 0; i < atomic_plus_slots; ++i)
    {
      result = OP {}(result, partials[i]);
    }
    red.combineTarget(result);
    RAJA::hip::device_mempool_type::getInstance().free(red.devicetarget);
    red.devicetarget = nullptr;
  }
  else
  {
    // complete reduction
    hi.res.wait();

    red.combineTarget(*red.devicetarget);

    // free memory
    RAJA::hip::device_zeroed_mempool_type::getInstance().free(red.device_count);
    red.device_count = nullptr;
    red.device_mem.deallocate();
    RAJA::hip::pinned_mempool_type::getInstance().free(red.devicetarget);
    red.devicetarget = nullptr;
  }
}

}  //  namespace detail
}  //  namespace expt
}  //  namespace RAJA

#endif

#endif  //  NEW_REDUCE_HIP_REDUCE_HPP
