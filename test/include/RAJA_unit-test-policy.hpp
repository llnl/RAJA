//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Header defining "for one" unit test utility so that constructs can be
// tested outside of standard RAJA kernel launch utilities (forall, kernel).
//

#ifndef __RAJA_test_policy_HPP__
#define __RAJA_test_policy_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/camp.hpp"

#include <type_traits>


// base classes to represent host or device in exec_dispatcher
struct RunOnHost {};
struct RunOnDevice {};

// sequential test policy
struct test_seq : public RunOnHost  { };

// struct with specializations containing information about test policies
template < typename test_policy >
struct test_policy_info;

// alias for equivalent RAJA exec policy to given test policy
template < typename test_policy >
using test_equivalent_exec_policy = typename test_policy_info<test_policy>::type;

// alias for platform of given test policy
template < typename test_policy >
using test_platform = typename test_policy_info<test_policy>::platform;

// alias for platform of given test policy
template < typename test_policy >
using test_resource = typename test_policy_info<test_policy>::resource;

template < typename test_policy >
test_resource<test_policy> get_test_resource()
{
  return test_resource<test_policy>::get_default();
}

template < typename dst_resource, typename src_resource, typename T >
inline T* test_reallocate(dst_resource dst_res, src_resource src_res, T* src, size_t len)
{
  T* dst = nullptr;
  if (dst_res.get_platform() == camp::resources::Platform::host) {
    dst = dst_res.template allocate<T>(len);
    src_res.memcpy(dst, src, len*sizeof(T));
    src_res.wait();
  } else if (src_res.get_platform() == camp::resources::Platform::host) {
    dst = dst_res.template allocate<T>(len);
    dst_res.memcpy(dst, src, len*sizeof(T));
    dst_res.wait();
  } else {
    throw std::runtime_error("Expected source or destination resource to be host");
  }
  src_res.deallocate(src);
  return dst;
}


// test_seq policy information
template < >
struct test_policy_info<test_seq>
{
  using resource = camp::resources::Host;
  using type = RAJA::seq_exec;
  using platform = RunOnHost;
  static const char* name() { return "test_seq"; }
};

#if defined(RAJA_ENABLE_TARGET_OPENMP)

// cuda test policy
struct test_openmp_target : public RunOnHost { };

// test_openmp_target policy information
template < >
struct test_policy_info<test_openmp_target>
{
  using resource = camp::resources::Omp;
  using type = RAJA::omp_target_parallel_for_exec<1>;
  using platform = RunOnHost;
  static const char* name() { return "test_openmp_target"; }
};

#endif

#if defined(RAJA_ENABLE_CUDA)

// cuda test policy
struct test_cuda : public RunOnDevice { };

// test_cuda policy information
template < >
struct test_policy_info<test_cuda>
{
  using resource = camp::resources::Cuda;
  using type = RAJA::cuda_exec<1>;
  using platform = RunOnDevice;
  static const char* name() { return "test_cuda"; }
};

#endif

#if defined(RAJA_ENABLE_HIP)

// hip test policy
struct test_hip : public RunOnDevice { };

// test_hip policy information
template < >
struct test_policy_info<test_hip>
{
  using resource = camp::resources::Hip;
  using type = RAJA::hip_exec<1>;
  using platform = RunOnDevice;
  static const char* name() { return "test_hip"; }
};

#endif


//
// unit test policies
//
using SequentialUnitTestPolicyList = camp::list<test_seq>;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPUnitTestPolicyList = SequentialUnitTestPolicyList;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaUnitTestPolicyList = camp::list<test_cuda>;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetUnitTestPolicyList = camp::list<test_openmp_target>;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipUnitTestPolicyList = camp::list<test_hip>;
#endif

#endif // RAJA_test_policy_HPP__
