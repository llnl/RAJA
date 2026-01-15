/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel::forall
 *          traversals on GPU with SYCL.
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

#ifndef RAJA_policy_sycl_kernel_HPP
#define RAJA_policy_sycl_kernel_HPP

#include "RAJA/policy/sycl/kernel/Conditional.hpp"
#include "RAJA/policy/sycl/kernel/SyclKernel.hpp"
#include "RAJA/policy/sycl/kernel/For.hpp"
#include "RAJA/policy/sycl/kernel/ForICount.hpp"
//#include "RAJA/policy/sycl/kernel/Hyperplane.hpp"
//#include "RAJA/policy/sycl/kernel/InitLocalMem.hpp"
#include "RAJA/policy/sycl/kernel/Lambda.hpp"
//#include "RAJA/policy/sycl/kernel/Reduce.hpp"
//#include "RAJA/policy/sycl/kernel/Sync.hpp"
#include "RAJA/policy/sycl/kernel/Tile.hpp"
#include "RAJA/policy/sycl/kernel/TileTCount.hpp"
#include "RAJA/policy/sycl/kernel/internal.hpp"

#endif  // closing endif for header file include guard
