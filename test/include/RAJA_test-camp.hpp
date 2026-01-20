//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Camp header includes and helpers used throughout RAJA tests.
//

#ifndef __RAJA_test_camp_HPP__
#define __RAJA_test_camp_HPP__

#include "camp/resource.hpp"
#include "camp/list.hpp"

//
// Memory resource types for back-end memory management
//
using HostResourceList = camp::list<camp::resources::Host>;

using SequentialResourceList = HostResourceList;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPResourceList = HostResourceList;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaResourceList = camp::list<camp::resources::Cuda>;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetResourceList = camp::list<camp::resources::Omp>;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipResourceList = camp::list<camp::resources::Hip>;
#endif

#if defined(RAJA_ENABLE_SYCL)
using SyclResourceList = camp::list<camp::resources::Sycl>;
#endif

#endif // __RAJA_test_camp_HPP__
