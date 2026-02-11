//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __WARP_THREAD_WARPLOOP_IMPL_HPP__
#define __WARP_THREAD_WARPLOOP_IMPL_HPP__

#include <numeric>

template<typename EXEC_POL, bool USE_RESOURCE,
         typename SEGMENTS,
         typename WORKING_RES,
         typename... Args>
typename std::enable_if< USE_RESOURCE >::type call_kernel(SEGMENTS&& segs, WORKING_RES work_res, Args&&... args) {
  RAJA::kernel_resource<EXEC_POL>( segs, work_res, args...);
}

template<typename EXEC_POL, bool USE_RESOURCE,
         typename SEGMENTS,
         typename WORKING_RES,
         typename... Args>
typename std::enable_if< !USE_RESOURCE >::type call_kernel(SEGMENTS&& segs, WORKING_RES, Args&&... args) {
  RAJA::kernel<EXEC_POL>( segs, args...);
}

template<typename EXEC_POL, bool USE_RESOURCE,
         typename SEGMENTS,
         typename PARAMS,
         typename WORKING_RES,
         typename... Args>
typename std::enable_if< USE_RESOURCE >::type call_kernel_param(SEGMENTS&& segs, PARAMS&& params, WORKING_RES work_res, Args&&... args) {
  RAJA::kernel_param_resource<EXEC_POL>( segs, params, work_res, args...);
}

template<typename EXEC_POL, bool USE_RESOURCE,
         typename SEGMENTS,
         typename PARAMS,
         typename WORKING_RES,
         typename... Args>
typename std::enable_if< !USE_RESOURCE >::type call_kernel_param(SEGMENTS&& segs, PARAMS&& params, WORKING_RES, Args&&... args) {
  RAJA::kernel_param<EXEC_POL>( segs, params, args...);
}

//
//
// Define list of nested loop types the WarpLoop test supports.
//
//
using WarpLoopSupportedLoopTypeList = camp::list<
  DEVICE_DEPTH_1_REDUCESUM_WARP,
  DEVICE_DEPTH_1_REDUCESUM_WARPDIRECT_TILE,
  DEVICE_DEPTH_2_REDUCESUM_WARP
>;

//
//
// Sum of array of elements with GPU-specific policies.
//
//
template <typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POL, bool USE_RESOURCE>
void KernelWarpThreadTest(const DEVICE_DEPTH_1_REDUCESUM_WARP&,
                          const RAJA::Index_type len)
{
  WORKING_RES work_res{WORKING_RES::get_default()};
  camp::resources::Resource erased_work_res{work_res};

  RAJA::Index_type* work_array;
  RAJA::Index_type* check_array;
  RAJA::Index_type* test_array;

  allocateForallTestData<RAJA::Index_type>(len,
                                     erased_work_res,
                                     &work_array,
                                     &check_array,
                                     &test_array);

  RAJA::TypedRangeSegment<RAJA::Index_type> rangelen(0, len);

  RAJA::ReduceSum<REDUCE_POL, RAJA::Index_type> worksum(0);

  call_kernel<EXEC_POLICY, USE_RESOURCE>(RAJA::make_tuple(RAJA::TypedRangeSegment<RAJA::Index_type>(0, len)), work_res,
                            [=] RAJA_HOST_DEVICE (RAJA::Index_type i) {
                              worksum += i;
                            });

  ASSERT_EQ(worksum.get(), len*(len-1)/2);

  deallocateForallTestData<RAJA::Index_type>(erased_work_res,
                                       work_array,
                                       check_array,
                                       test_array);
}

template <typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POL, bool USE_RESOURCE>
void KernelWarpThreadTest(const DEVICE_DEPTH_2_REDUCESUM_WARP&,
                          const RAJA::Index_type numtiles)
{
  WORKING_RES work_res{WORKING_RES::get_default()};
  camp::resources::Resource erased_work_res{work_res};

  RAJA::Index_type flatSize = 32 * numtiles;
  RAJA::Index_type* work_array;
  RAJA::Index_type* check_array;
  RAJA::Index_type* test_array;

  allocateForallTestData<RAJA::Index_type>(flatSize,
                                     erased_work_res,
                                     &work_array,
                                     &check_array,
                                     &test_array);

  RAJA::TypedRangeSegment<RAJA::Index_type> rangelen(0, flatSize);

  RAJA::ReduceSum<REDUCE_POL, RAJA::Index_type> worksum(0);

  call_kernel_param<EXEC_POLICY, USE_RESOURCE>(
                            RAJA::make_tuple(RAJA::TypedRangeSegment<RAJA::Index_type>(0, flatSize)),
                            RAJA::make_tuple((RAJA::Index_type)0),
                            work_res,
                            [=] RAJA_HOST_DEVICE (RAJA::Index_type RAJA_UNUSED_ARG(i), RAJA::Index_type j) {
                              worksum += j; // j should only be 0..31
                            });

  ASSERT_EQ(worksum.get(), numtiles*32*(32-1)/2);

  deallocateForallTestData<RAJA::Index_type>(erased_work_res,
                                       work_array,
                                       check_array,
                                       test_array);
}

// More specific execution policies that use the above DEVICE_DEPTH_1_REDUCESUM_WARP test.
template <typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POL, bool USE_RESOURCE, typename... Args>
void KernelWarpThreadTest(const DEVICE_DEPTH_1_REDUCESUM_WARPDIRECT_TILE&, Args... args){
  KernelWarpThreadTest<WORKING_RES, EXEC_POLICY, REDUCE_POL, USE_RESOURCE>(DEVICE_DEPTH_1_REDUCESUM_WARP(), args...);
}

//
//
// Defining the Kernel Loop structure for WarpLoop Nested Loop Tests.
//
//
template<typename POLICY_TYPE, typename REDUCE_POL, typename POLICY_DATA>
struct WarpThreadExec;

#if defined(RAJA_ENABLE_CUDA) or defined(RAJA_ENABLE_HIP)

template<typename REDUCE_POL, typename POLICY_DATA>
struct WarpThreadExec<DEVICE_DEPTH_1_REDUCESUM_WARP, REDUCE_POL, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::DEVICE_KERNEL<
        RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<0>>::type,
          RAJA::statement::Lambda<0>
        >
      > // end DEVICE_KERNEL
    >;
};

template<typename REDUCE_POL, typename POLICY_DATA>
struct WarpThreadExec<DEVICE_DEPTH_1_REDUCESUM_WARPDIRECT_TILE, REDUCE_POL, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::DEVICE_KERNEL<
        RAJA::statement::Tile<0, RAJA::tile_fixed<32>, RAJA::seq_exec,
          RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<0>>::type,
            RAJA::statement::Lambda<0>
          >
        >
      > // end DEVICE_KERNEL
    >;
};

template<typename REDUCE_POL, typename POLICY_DATA>
struct WarpThreadExec<DEVICE_DEPTH_2_REDUCESUM_WARP, REDUCE_POL, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::DEVICE_KERNEL<
        RAJA::statement::Tile<0, RAJA::tile_fixed<32>, RAJA::seq_exec,
          RAJA::statement::ForICount<0, RAJA::statement::Param<0>, typename camp::at<POLICY_DATA, camp::num<0>>::type,
            RAJA::statement::Lambda<0>
          >
        >
      > // end DEVICE_KERNEL
    >;
};

#endif  // RAJA_ENABLE_CUDA or RAJA_ENABLE_HIP

#endif  // __WARP_THREAD_WARPLOOP_IMPL_HPP__
