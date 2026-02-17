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

namespace RAJA
{
namespace expt
{
namespace detail
{

// Init
template<typename EXEC_POL, typename OP, typename T, typename VOp>
camp::concepts::enable_if<RAJA::type_traits::is_hip_policy<EXEC_POL>>
param_init(EXEC_POL const&,
           Reducer<OP, T, VOp>& red,
           RAJA::hip::detail::hipInfo& hi)
{
  red.devicetarget =
      RAJA::hip::pinned_mempool_type::getInstance().template malloc<T>(1);
  red.device_mem.allocate(hi.gridDim.x * hi.gridDim.y * hi.gridDim.z);
  red.device_count = RAJA::hip::device_zeroed_mempool_type::getInstance()
                         .template malloc<unsigned int>(1);
}

// Combine
template<typename EXEC_POL, typename OP, typename T, typename VOp>
RAJA_HOST_DEVICE camp::concepts::enable_if<
    RAJA::type_traits::is_hip_policy<EXEC_POL>>
param_combine(EXEC_POL const&, Reducer<OP, T, VOp>& red)
{
  RAJA::hip::impl::expt::grid_reduce<typename EXEC_POL::IterationGetter, OP>(
      red.devicetarget, red.getVal(), red.device_mem, red.device_count);
}

// Resolve
template<typename EXEC_POL, typename OP, typename T, typename VOp>
camp::concepts::enable_if<RAJA::type_traits::is_hip_policy<EXEC_POL>>
param_resolve(EXEC_POL const&,
              Reducer<OP, T, VOp>& red,
              RAJA::hip::detail::hipInfo& hi)
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

}  //  namespace detail
}  //  namespace expt
}  //  namespace RAJA

#endif

#endif  //  NEW_REDUCE_HIP_REDUCE_HPP
