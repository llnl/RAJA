/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel
 *          traversals on GPU with CUDA.
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

#ifndef RAJA_policy_cuda_kernel_Lambda_HPP
#define RAJA_policy_cuda_kernel_Lambda_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <cassert>
#include <climits>

#include "camp/camp.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel.hpp"
#include "RAJA/pattern/kernel/Lambda.hpp"

namespace RAJA
{
namespace internal
{

template<typename Data,
         camp::idx_t LambdaIndex,
         typename... Args,
         typename Types>
struct CudaStatementExecutor<Data,
                             statement::Lambda<LambdaIndex, Args...>,
                             Types>
{

  static inline RAJA_DEVICE void exec(Data& data, bool thread_active)
  {
    // Only execute the lambda if it hasn't been masked off
    if (thread_active)
    {
      StatementExecutor<statement::Lambda<LambdaIndex, Args...>, Types>::exec(
          data);
    }
  }

  static inline LaunchDims calculateDimensions(
      Data const& RAJA_UNUSED_ARG(data))
  {
    return LaunchDims();
  }
};


}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
