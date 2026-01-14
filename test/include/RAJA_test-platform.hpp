//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Platform header includes and helpers used throughout RAJA tests.
//

#ifndef __RAJA_test_platform_HPP__
#define __RAJA_test_platform_HPP__

#include "RAJA/RAJA.hpp"

#include "camp/list.hpp"

template < RAJA::Platform PLATFORM >
struct PlatformHolder
{
   static const RAJA::Platform platform = PLATFORM;
};

//
// Platform types
//
using HostPlatformList = camp::list<PlatformHolder<RAJA::Platform::host>>;

using SequentialPlatformList = HostPlatformList;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPPlatformList = HostPlatformList;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaPlatformList = camp::list<PlatformHolder<RAJA::Platform::cuda>>;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetPlatformList = camp::list<PlatformHolder<RAJA::Platform::omp_target>>;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipPlatformList = camp::list<PlatformHolder<RAJA::Platform::hip>>;
#endif

#endif // __RAJA_test_platform_HPP__
