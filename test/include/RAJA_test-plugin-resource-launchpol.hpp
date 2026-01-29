//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Kernel execution policy lists used throughout plugin tests
//

#ifndef __RAJA_test_plugin_resource_launchpol_HPP__
#define __RAJA_test_plugin_resource_launchpol_HPP__

#include "RAJA/RAJA.hpp"

#include "camp/list.hpp"

// Sequential execution policy types
using SequentialPluginResourceLaunchExecPols = camp::list<RAJA::LaunchPolicy<RAJA::seq_launch_t>>;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPPluginResourceLaunchExecPols = camp::list<RAJA::LaunchPolicy<RAJA::omp_launch_t>>;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaPluginResourceLaunchExecPols = camp::list<RAJA::LaunchPolicy<RAJA::seq_launch_t, RAJA::cuda_launch_t<false>>>;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipPluginResourceLaunchExecPols = camp::list<RAJA::LaunchPolicy<RAJA::seq_launch_t, RAJA::hip_launch_t<false>>>;

#endif

#endif  // __RAJA_test_plugin_kernelpol_HPP__
