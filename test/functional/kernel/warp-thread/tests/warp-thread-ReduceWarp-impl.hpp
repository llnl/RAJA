//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __WARP_THREAD_REDUCEWARP_IMPL_HPP__
#define __WARP_THREAD_REDUCEWARP_IMPL_HPP__

#include <numeric>

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
// Define list of nested loop types the ReduceWarp test supports.
//
//
using ReduceWarpSupportedLoopTypeList = camp::list<
  DEVICE_DEPTH_1_REDUCESUM_WARPREDUCE,
  DEVICE_DEPTH_2_REDUCESUM_WARPREDUCE,
  DEVICE_DEPTH_3_REDUCESUM_WARPREDUCE
>;

//
//
// Sum of array of elements with GPU-specific policies.
//
//
template <typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POL, bool USE_RESOURCE>
void KernelWarpThreadTest(const DEVICE_DEPTH_1_REDUCESUM_WARPREDUCE&,
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

  RAJA::ReduceSum<REDUCE_POL, RAJA::Index_type> worksum(0);
  RAJA::ReduceSum<REDUCE_POL, RAJA::Index_type> reduce_count(0);

  call_kernel_param<EXEC_POLICY, USE_RESOURCE>(
                            RAJA::make_tuple(RAJA::TypedRangeSegment<RAJA::Index_type>(0, len)),
                            RAJA::make_tuple((RAJA::Index_type)0),
                            work_res,

                            [=] RAJA_HOST_DEVICE (RAJA::Index_type i, RAJA::Index_type &value) {
                              value += i;
                            },

                            [=] RAJA_HOST_DEVICE (RAJA::Index_type &value) {
                              // This only gets executed on the "root" thread which received the reduced value.
                              worksum += value;
                              reduce_count += 1;
                            });

  ASSERT_EQ(worksum.get(), len*(len-1)/2);
  ASSERT_EQ(reduce_count.get(), 1);

  deallocateForallTestData<RAJA::Index_type>(erased_work_res,
                                       work_array,
                                       check_array,
                                       test_array);
}

template <typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POL, bool USE_RESOURCE>
void KernelWarpThreadTest(const DEVICE_DEPTH_2_REDUCESUM_WARPREDUCE&,
                          const RAJA::Index_type len) // len needs to be divisible by 10 and 16
{
  WORKING_RES work_res{WORKING_RES::get_default()};
  camp::resources::Resource erased_work_res{work_res};

  RAJA::Index_type* work_array;
  RAJA::Index_type* check_array;
  RAJA::Index_type* test_array;

  RAJA::Index_type innerlen = 10;
  RAJA::Index_type outerlen = len / innerlen;

  allocateForallTestData<RAJA::Index_type>(len,
                                     erased_work_res,
                                     &work_array,
                                     &check_array,
                                     &test_array);

  RAJA::ReduceSum<REDUCE_POL, RAJA::Index_type> worksum(0);
  RAJA::ReduceSum<REDUCE_POL, RAJA::Index_type> reduce_count(0);

  call_kernel_param<EXEC_POLICY, USE_RESOURCE>(
                            RAJA::make_tuple(RAJA::TypedRangeSegment<RAJA::Index_type>(0, outerlen),
                                             RAJA::TypedRangeSegment<RAJA::Index_type>(0, innerlen)),
                            RAJA::make_tuple((RAJA::Index_type)0),
                            work_res,

                            [=] RAJA_HOST_DEVICE (RAJA::Index_type i, RAJA::Index_type j, RAJA::Index_type &value) {
                              value += i + j * outerlen;
                            },

                            [=] RAJA_HOST_DEVICE (RAJA::Index_type &value) {
                              // This only gets executed on the "root" thread which received the reduced value.
                              worksum += value;
                              reduce_count += 1;
                            });

  ASSERT_EQ(worksum.get(), outerlen*innerlen*(outerlen*innerlen-1)/2);
  ASSERT_EQ(reduce_count.get(), innerlen);

  deallocateForallTestData<RAJA::Index_type>(erased_work_res,
                                       work_array,
                                       check_array,
                                       test_array);
}

template <typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POL, bool USE_RESOURCE>
void KernelWarpThreadTest(const DEVICE_DEPTH_3_REDUCESUM_WARPREDUCE&,
                          const RAJA::Index_type len) // len needs to be divisible by 10 and 16
{
  WORKING_RES work_res{WORKING_RES::get_default()};
  camp::resources::Resource erased_work_res{work_res};

  RAJA::Index_type* work_array;
  RAJA::Index_type* check_array;
  RAJA::Index_type* test_array;

  RAJA::Index_type innerlen = 10;
  RAJA::Index_type middlelen = 16;
  RAJA::Index_type outerlen = len / (innerlen*middlelen);

  allocateForallTestData<RAJA::Index_type>(len,
                                     erased_work_res,
                                     &work_array,
                                     &check_array,
                                     &test_array);

  RAJA::ReduceSum<REDUCE_POL, RAJA::Index_type> worksum(0);
  RAJA::ReduceSum<REDUCE_POL, RAJA::Index_type> reduce_count(0);

  call_kernel_param<EXEC_POLICY, USE_RESOURCE>(
                            RAJA::make_tuple(RAJA::TypedRangeSegment<RAJA::Index_type>(0, outerlen),
                                             RAJA::TypedRangeSegment<RAJA::Index_type>(0, middlelen),
                                             RAJA::TypedRangeSegment<RAJA::Index_type>(0, innerlen)),
                            RAJA::make_tuple((RAJA::Index_type)0),
                            work_res,

                            [=] RAJA_HOST_DEVICE (RAJA::Index_type i, RAJA::Index_type j, RAJA::Index_type k, RAJA::Index_type &value) {
                              value += i + j * outerlen + k * outerlen * middlelen;
                            },

                            [=] RAJA_HOST_DEVICE (RAJA::Index_type &value) {
                              // This only gets executed on the "root" thread which received the reduced value.
                              worksum += value;
                              reduce_count += 1;
                            });

  ASSERT_EQ(worksum.get(), outerlen*middlelen*innerlen*(outerlen*middlelen*innerlen-1)/2);
  ASSERT_EQ(reduce_count.get(), middlelen*innerlen);

  deallocateForallTestData<RAJA::Index_type>(erased_work_res,
                                       work_array,
                                       check_array,
                                       test_array);
}

//
//
// Defining the Kernel Loop structure for ReduceWarp Nested Loop Tests.
//
//
template<typename POLICY_TYPE, typename REDUCE_POL, typename POLICY_DATA>
struct WarpThreadExec;

#if defined(RAJA_ENABLE_CUDA) or defined(RAJA_ENABLE_HIP)

template<typename REDUCE_POL, typename POLICY_DATA>
struct WarpThreadExec<DEVICE_DEPTH_1_REDUCESUM_WARPREDUCE, REDUCE_POL, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::DEVICE_KERNEL<
        RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<0>>::type, RAJA::statement::Lambda<0>>,
        RAJA::statement::Reduce<typename camp::at<POLICY_DATA, camp::num<1>>::type, RAJA::operators::plus, RAJA::statement::Param<0>,
          RAJA::statement::Lambda<1, RAJA::Params<0>>
        >
      > // end DEVICE_KERNEL
    >;
};

template<typename REDUCE_POL, typename POLICY_DATA>
struct WarpThreadExec<DEVICE_DEPTH_2_REDUCESUM_WARPREDUCE, REDUCE_POL, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::DEVICE_KERNEL<
        RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<0>>::type,
          RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<1>>::type, RAJA::statement::Lambda<0>
          >
        >,
        RAJA::statement::Reduce<typename camp::at<POLICY_DATA, camp::num<2>>::type, RAJA::operators::plus, RAJA::statement::Param<0>,
          RAJA::statement::Lambda<1, RAJA::Params<0>>
        >
      > // end DEVICE_KERNEL
    >;
};

template<typename REDUCE_POL, typename POLICY_DATA>
struct WarpThreadExec<DEVICE_DEPTH_3_REDUCESUM_WARPREDUCE, REDUCE_POL, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::DEVICE_KERNEL<
        RAJA::statement::For<2, typename camp::at<POLICY_DATA, camp::num<0>>::type,
          RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<1>>::type,
            RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<2>>::type, RAJA::statement::Lambda<0>
            > // end For 0
          >,  // end For 1
          typename camp::at<POLICY_DATA, camp::num<3>>::type, // warp synchronize
          RAJA::statement::Reduce<typename camp::at<POLICY_DATA, camp::num<4>>::type, RAJA::operators::plus, RAJA::statement::Param<0>,
            RAJA::statement::Lambda<1, RAJA::Params<0>>
          >
        > // end For 2
      > // end DEVICE_KERNEL
    >;
};

#endif  // RAJA_ENABLE_CUDA or RAJA_ENABLE_HIP

#endif  // __WARP_THREAD_REDUCEWARP_IMPL_HPP__
