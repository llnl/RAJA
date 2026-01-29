/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel::forall
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


#ifndef RAJA_policy_cuda_kernel_HPP
#define RAJA_policy_cuda_kernel_HPP

#include "RAJA/policy/cuda/kernel/Conditional.hpp"
#include "RAJA/policy/cuda/kernel/CudaKernel.hpp"
#include "RAJA/policy/cuda/kernel/For.hpp"
#include "RAJA/policy/cuda/kernel/ForICount.hpp"
#include "RAJA/policy/cuda/kernel/Hyperplane.hpp"
#include "RAJA/policy/cuda/kernel/InitLocalMem.hpp"
#include "RAJA/policy/cuda/kernel/Lambda.hpp"
#include "RAJA/policy/cuda/kernel/Reduce.hpp"
#include "RAJA/policy/cuda/kernel/Sync.hpp"
#include "RAJA/policy/cuda/kernel/Tile.hpp"
#include "RAJA/policy/cuda/kernel/TileTCount.hpp"
#include "RAJA/policy/cuda/kernel/internal.hpp"

#endif  // closing endif for header file include guard
