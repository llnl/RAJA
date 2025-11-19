/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_openmp_kernel_ompsyncthreads_HPP
#define RAJA_policy_openmp_kernel_ompsyncthreads_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_OPENMP)

#include "RAJA/pattern/kernel/internal.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/openmp/policy.hpp"

namespace RAJA
{

namespace statement
{
struct OmpSyncThreads : public internal::Statement<camp::nil>
{};

}  // namespace statement

namespace internal
{


// Statement executor to synchronize omp threads inside a kernel region
template<typename Types>
struct StatementExecutor<statement::OmpSyncThreads, Types>
{

  template<typename Data>
  static RAJA_INLINE void exec(Data&&)
  {
#pragma omp barrier
  }
};


}  // namespace internal
}  // namespace RAJA


#endif  // closing endif for RAJA_ENABLE_OPENMP guard

#endif  // closing endif for header file include guard
