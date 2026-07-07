/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing device policy aliases.
 *
 *          These aliases are intended to reduce downstream preprocessor
 *          conditionals when targeting different GPU backends.
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

#ifndef RAJA_policy_device_HPP
#define RAJA_policy_device_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/types.hpp"

#if defined(RAJA_CUDA_ACTIVE)
#include "RAJA/policy/cuda/policy.hpp"
#elif defined(RAJA_HIP_ACTIVE)
#include "RAJA/policy/hip/policy.hpp"
#elif defined(RAJA_SYCL_ACTIVE)
#include "RAJA/policy/sycl/policy.hpp"
#include "RAJA/policy/sycl/launch.hpp"
#endif

namespace RAJA
{

/*!
 * Generic device policy aliases.
 *
 * These aliases select the active GPU backend (CUDA/HIP/SYCL) at compile time.
 *
 * For SYCL, CUDA-like x/y/z naming is used to match CUDA/HIP conventions:
 *   x -> dim2, y -> dim1, z -> dim0
 */

#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE)

#if defined(RAJA_CUDA_ACTIVE)
#define RAJA_DEVICE_BACKEND_PREFIX cuda
#elif defined(RAJA_HIP_ACTIVE)
#define RAJA_DEVICE_BACKEND_PREFIX hip
#endif

#define RAJA_DEVICE_CONCAT_IMPL(prefix, suffix) prefix##suffix
#define RAJA_DEVICE_CONCAT(prefix, suffix) RAJA_DEVICE_CONCAT_IMPL(prefix, suffix)
#define RAJA_DEVICE_ALIAS(name) RAJA_DEVICE_CONCAT(RAJA_DEVICE_BACKEND_PREFIX, _##name)

// forall
template<size_t BLOCK_SIZE, bool Async = false>
using device_exec = RAJA_DEVICE_ALIAS(exec)<BLOCK_SIZE, Async>;

template<size_t BLOCK_SIZE>
using device_exec_async = device_exec<BLOCK_SIZE, true>;

// launch
template<bool Async, int num_threads = RAJA::named_usage::unspecified>
using device_launch_t = RAJA_DEVICE_ALIAS(launch_t)<Async, num_threads>;

// kernel (For) index mapping
template<int nx_threads>
using device_global_size_x_direct =
    RAJA_DEVICE_ALIAS(global_size_x_direct)<nx_threads>;
template<int ny_threads>
using device_global_size_y_direct =
    RAJA_DEVICE_ALIAS(global_size_y_direct)<ny_threads>;
template<int nz_threads>
using device_global_size_z_direct =
    RAJA_DEVICE_ALIAS(global_size_z_direct)<nz_threads>;

// launch (loop) index mapping
using device_global_thread_x = RAJA_DEVICE_ALIAS(global_thread_x);
using device_global_thread_y = RAJA_DEVICE_ALIAS(global_thread_y);
using device_global_thread_z = RAJA_DEVICE_ALIAS(global_thread_z);

// kernel (loop) index mapping
using device_thread_x_direct = RAJA_DEVICE_ALIAS(thread_x_direct);
using device_thread_y_direct = RAJA_DEVICE_ALIAS(thread_y_direct);
using device_thread_z_direct = RAJA_DEVICE_ALIAS(thread_z_direct);

using device_thread_x_loop = RAJA_DEVICE_ALIAS(thread_x_loop);
using device_thread_y_loop = RAJA_DEVICE_ALIAS(thread_y_loop);
using device_thread_z_loop = RAJA_DEVICE_ALIAS(thread_z_loop);

using device_block_x_direct = RAJA_DEVICE_ALIAS(block_x_direct);
using device_block_y_direct = RAJA_DEVICE_ALIAS(block_y_direct);
using device_block_z_direct = RAJA_DEVICE_ALIAS(block_z_direct);

using device_block_x_loop = RAJA_DEVICE_ALIAS(block_x_loop);
using device_block_y_loop = RAJA_DEVICE_ALIAS(block_y_loop);
using device_block_z_loop = RAJA_DEVICE_ALIAS(block_z_loop);

#undef RAJA_DEVICE_ALIAS
#undef RAJA_DEVICE_CONCAT
#undef RAJA_DEVICE_CONCAT_IMPL
#undef RAJA_DEVICE_BACKEND_PREFIX

#elif defined(RAJA_SYCL_ACTIVE)

// forall
template<size_t WORK_GROUP_SIZE, bool Async = false>
using device_exec = RAJA::sycl_exec<WORK_GROUP_SIZE, Async>;

template<size_t WORK_GROUP_SIZE>
using device_exec_async = device_exec<WORK_GROUP_SIZE, true>;

// launch
template<bool Async, int num_threads = RAJA::named_usage::unspecified>
using device_launch_t = RAJA::sycl_launch_t<Async, num_threads>;

// kernel (For) index mapping (x/y/z -> dim2/dim1/dim0)
template<int nx_threads>
using device_global_size_x_direct = RAJA::sycl_global_2<nx_threads>;
template<int ny_threads>
using device_global_size_y_direct = RAJA::sycl_global_1<ny_threads>;
template<int nz_threads>
using device_global_size_z_direct = RAJA::sycl_global_0<nz_threads>;

// launch (loop) index mapping (x/y/z -> dim2/dim1/dim0)
using device_global_thread_x = RAJA::sycl_global_item_2;
using device_global_thread_y = RAJA::sycl_global_item_1;
using device_global_thread_z = RAJA::sycl_global_item_0;

// kernel (loop) index mapping (x/y/z -> dim2/dim1/dim0)
using device_thread_x_direct = RAJA::sycl_local_2_direct;
using device_thread_y_direct = RAJA::sycl_local_1_direct;
using device_thread_z_direct = RAJA::sycl_local_0_direct;

using device_thread_x_loop = RAJA::sycl_local_2_loop;
using device_thread_y_loop = RAJA::sycl_local_1_loop;
using device_thread_z_loop = RAJA::sycl_local_0_loop;

using device_block_x_direct = RAJA::sycl_group_2_direct;
using device_block_y_direct = RAJA::sycl_group_1_direct;
using device_block_z_direct = RAJA::sycl_group_0_direct;

using device_block_x_loop = RAJA::sycl_group_2_loop;
using device_block_y_loop = RAJA::sycl_group_1_loop;
using device_block_z_loop = RAJA::sycl_group_0_loop;

#endif  // active backend

}  // namespace RAJA

#endif  // RAJA_policy_device_HPP
