//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __WARP_THREAD_REDUCEMASK_IMPL_HPP__
#define __WARP_THREAD_REDUCEMASK_IMPL_HPP__

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
// Define list of nested loop types the ReduceMask test supports.
//
//
using ReduceMaskSupportedLoopTypeList = camp::list<
  DEVICE_DEPTH_2_REDUCESUM_WARPMASK,
  DEVICE_DEPTH_2_REDUCESUM_WARPMASK_FORI
>;

//
//
// Sum of array of elements with GPU-specific policies.
//
//
template <typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POL, bool USE_RESOURCE>
void KernelWarpThreadTest(const DEVICE_DEPTH_2_REDUCESUM_WARPMASK&,
                          const RAJA::Index_type directlen,
                          const RAJA::Index_type looplen)
{
  WORKING_RES work_res{WORKING_RES::get_default()};
  camp::resources::Resource erased_work_res{work_res};

  RAJA::Index_type* work_array;
  RAJA::Index_type* check_array;
  RAJA::Index_type* test_array;

  allocateForallTestData<RAJA::Index_type>(directlen*looplen,
                                     erased_work_res,
                                     &work_array,
                                     &check_array,
                                     &test_array);

  RAJA::ReduceMax<REDUCE_POL, int> max_thread(0);
  RAJA::ReduceSum<REDUCE_POL, RAJA::Index_type> trip_count(0);
  RAJA::ReduceSum<REDUCE_POL, RAJA::Index_type> worksum(0);

  call_kernel<EXEC_POLICY, USE_RESOURCE>(
                            RAJA::make_tuple(RAJA::TypedRangeSegment<RAJA::Index_type>(0, directlen), RAJA::TypedRangeSegment<RAJA::Index_type>(0, looplen)),
                            work_res,
                            [=] RAJA_DEVICE (RAJA::Index_type i, RAJA::Index_type RAJA_UNUSED_ARG(j)) {
                              trip_count += 1;
                              worksum += i; // i should only be 0..directlen-1
                              max_thread.max(threadIdx.x);
                            });

  ASSERT_EQ(max_thread.get(), 255);
  ASSERT_EQ(trip_count.get(), looplen*directlen);
  ASSERT_EQ(worksum.get(), looplen*directlen*(directlen-1)/2);

  deallocateForallTestData<RAJA::Index_type>(erased_work_res,
                                       work_array,
                                       check_array,
                                       test_array);
}

template <typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POL, bool USE_RESOURCE>
void KernelWarpThreadTest(const DEVICE_DEPTH_2_REDUCESUM_WARPMASK_FORI&,
                          const RAJA::Index_type directlen,
                          const RAJA::Index_type looplen)
{
  WORKING_RES work_res{WORKING_RES::get_default()};
  camp::resources::Resource erased_work_res{work_res};

  RAJA::Index_type* work_array;
  RAJA::Index_type* check_array;
  RAJA::Index_type* test_array;

  allocateForallTestData<RAJA::Index_type>(directlen*looplen,
                                     erased_work_res,
                                     &work_array,
                                     &check_array,
                                     &test_array);

  RAJA::ReduceMax<REDUCE_POL, int> max_thread(0);
  RAJA::ReduceSum<REDUCE_POL, RAJA::Index_type> trip_count(0);
  RAJA::ReduceSum<REDUCE_POL, RAJA::Index_type> worksum(0);

  call_kernel_param<EXEC_POLICY, USE_RESOURCE>(
                            RAJA::make_tuple(RAJA::TypedRangeSegment<RAJA::Index_type>(0, directlen), RAJA::TypedRangeSegment<RAJA::Index_type>(0, looplen)),
                            RAJA::make_tuple((RAJA::Index_type)0, (RAJA::Index_type)0),
                            work_res,
                            [=] RAJA_DEVICE (RAJA::Index_type RAJA_UNUSED_ARG(i), RAJA::Index_type RAJA_UNUSED_ARG(j), RAJA::Index_type RAJA_UNUSED_ARG(x), RAJA::Index_type y) {
                              trip_count += 1;
                              worksum += y; // y should only be 0..3
                              max_thread.max(threadIdx.x);
                            });

  ASSERT_EQ(max_thread.get(), 255);
  ASSERT_EQ(trip_count.get(), looplen*directlen);
  ASSERT_EQ(worksum.get(), looplen*directlen*(looplen-1)/2);

  deallocateForallTestData<RAJA::Index_type>(erased_work_res,
                                       work_array,
                                       check_array,
                                       test_array);
}

//
//
// Defining the Kernel Loop structure for ReduceMask Nested Loop Tests.
//
//
template<typename POLICY_TYPE, typename REDUCE_POL, typename POLICY_DATA>
struct WarpThreadExec;

#if defined(RAJA_ENABLE_CUDA) or defined(RAJA_ENABLE_HIP)

template<typename REDUCE_POL, typename POLICY_DATA>
struct WarpThreadExec<DEVICE_DEPTH_2_REDUCESUM_WARPMASK, REDUCE_POL, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::DEVICE_KERNEL<
        RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<0>>::type,
          RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<1>>::type,
            RAJA::statement::Lambda<0>
          >
        >
      > // end DEVICE_KERNEL
    >;
};

template<typename REDUCE_POL, typename POLICY_DATA>
struct WarpThreadExec<DEVICE_DEPTH_2_REDUCESUM_WARPMASK_FORI, REDUCE_POL, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::DEVICE_KERNEL<
        RAJA::statement::ForICount<0, RAJA::statement::Param<0>, typename camp::at<POLICY_DATA, camp::num<0>>::type,
          RAJA::statement::ForICount<1, RAJA::statement::Param<1>, typename camp::at<POLICY_DATA, camp::num<1>>::type,
            RAJA::statement::Lambda<0>
          >
        >
      > // end DEVICE_KERNEL
    >;
};

#endif  // RAJA_ENABLE_CUDA or RAJA_ENABLE_HIP

#endif  // __WARP_THREAD_REDUCEMASK_IMPL_HPP__
