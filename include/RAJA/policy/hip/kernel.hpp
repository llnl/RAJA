/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel::forall
 *          traversals on GPU with HIP.
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


#ifndef RAJA_policy_hip_kernel_HPP
#define RAJA_policy_hip_kernel_HPP

#include "RAJA/policy/hip/kernel/Conditional.hpp"
#include "RAJA/policy/hip/kernel/For.hpp"
#include "RAJA/policy/hip/kernel/ForICount.hpp"
#include "RAJA/policy/hip/kernel/HipKernel.hpp"
#include "RAJA/policy/hip/kernel/Hyperplane.hpp"
#include "RAJA/policy/hip/kernel/InitLocalMem.hpp"
#include "RAJA/policy/hip/kernel/Lambda.hpp"
#include "RAJA/policy/hip/kernel/Reduce.hpp"
#include "RAJA/policy/hip/kernel/Sync.hpp"
#include "RAJA/policy/hip/kernel/Tile.hpp"
#include "RAJA/policy/hip/kernel/TileTCount.hpp"
#include "RAJA/policy/hip/kernel/internal.hpp"

#endif  // closing endif for header file include guard
