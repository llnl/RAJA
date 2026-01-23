//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __NESTED_LOOP_MULTI_LAMBDA_PARAM_REDUCE_SUM_IMPL_HPP__
#define __NESTED_LOOP_MULTI_LAMBDA_PARAM_REDUCE_SUM_IMPL_HPP__

#include <numeric>

template<typename EXEC_POL, bool USE_RESOURCE,
         typename SEGMENTS,
         typename PARAMS,
         typename WORKING_RES,
         typename... Args>
typename std::enable_if< USE_RESOURCE >::type call_kernel(SEGMENTS&& segs, PARAMS&& params, WORKING_RES work_res, Args&&... args) {
  RAJA::kernel_param_resource<EXEC_POL>( segs, params, work_res, args...);
}

template<typename EXEC_POL, bool USE_RESOURCE,
         typename SEGMENTS,
         typename PARAMS,
         typename WORKING_RES,
         typename... Args>
typename std::enable_if< !USE_RESOURCE >::type call_kernel(SEGMENTS&& segs, PARAMS&& params, WORKING_RES, Args&&... args) {
  RAJA::kernel_param<EXEC_POL>( segs, params, args...);
}

//
//
// Define list of nested loop types the Block test supports.
//
//
using BlockReduceSumSupportedLoopTypeList = camp::list<
  DEPTH_1_REDUCESUM,
  DEVICE_DEPTH_1_REDUCESUM
  >;

//
//
// Nest loop trip count test.
//
//
template <typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POL, bool USE_RESOURCE>
void KernelNestedLoopTest(const DEPTH_1_REDUCESUM&, const int N){

  WORKING_RES work_res{WORKING_RES::get_default()};
  camp::resources::Resource erased_work_res{work_res};

  // Allocate Tests Data
  int * work_array;
  int * check_array;
  int * test_array;

  allocateForallTestData<int>(N,
                              erased_work_res,
                              &work_array,
                              &check_array,
                              &test_array);

  RAJA::TypedRangeSegment<int> range(0,N);

  // Initialize Data
  std::iota(test_array, test_array + RAJA::stripIndexType(N), 0);

  erased_work_res.memcpy(work_array, test_array, sizeof(int) * RAJA::stripIndexType(N));
  
  RAJA::ReduceSum<REDUCE_POL, int> worksum(0);

  // Calculate Working data
  call_kernel<EXEC_POLICY, USE_RESOURCE>(
    RAJA::make_tuple(RAJA::RangeSegment(0, N)),
    RAJA::make_tuple<int>(0),

    // Resource
    work_res,

    // lambda 0, only runs for sequential
    [=] RAJA_HOST_DEVICE (RAJA::Index_type i, int & value) {
       value = work_array[i];
    },

    // lambda 1, only runs for device
    [=] RAJA_HOST_DEVICE (RAJA::Index_type i, int & value) {
       value += work_array[i];
    },

    // lambda 2, (reduction) runs for both sequential and device
    // Device: This only gets executed on the "root" thread which received the reduced value.
    [=] RAJA_HOST_DEVICE (int & value) {
       worksum += value;
    }

  );

  ASSERT_EQ(worksum.get(), N*(N-1)/2);

  deallocateForallTestData<int>(erased_work_res,
                                work_array,
                                check_array,
                                test_array);
}

// DEVICE_ and DEPTH_1_REDUCESUM execution policies use the above DEPTH_1_REDUCESUM test.
template <typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POL, bool USE_RESOURCE, typename... Args>
void KernelNestedLoopTest(const DEVICE_DEPTH_1_REDUCESUM&, Args... args){
  KernelNestedLoopTest<WORKING_RES, EXEC_POLICY, REDUCE_POL, USE_RESOURCE>(DEPTH_1_REDUCESUM(), args...);
}

//
//
// Defining the Kernel Loop structure for Block Nested Loop Tests.
//
//
template<typename POLICY_TYPE, typename REDUCE_POL, typename POLICY_DATA>
struct BlockNestedLoopExec;

template<typename REDUCE_POL, typename POLICY_DATA>
struct BlockNestedLoopExec<DEPTH_1_REDUCESUM, REDUCE_POL, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<0>>::type, RAJA::statement::Lambda<0>,
        RAJA::statement::Reduce<typename camp::at<POLICY_DATA, camp::num<1>>::type, RAJA::operators::plus, RAJA::statement::Param<0>,
          RAJA::statement::Lambda<2, RAJA::Params<0>>
        >
      >
    >;
};

#if defined(RAJA_ENABLE_CUDA) or defined(RAJA_ENABLE_HIP)

template<typename REDUCE_POL, typename POLICY_DATA>
struct BlockNestedLoopExec<DEVICE_DEPTH_1_REDUCESUM, REDUCE_POL, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::DEVICE_KERNEL<
      RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<0>>::type, RAJA::statement::Lambda<1>>,
        RAJA::statement::Reduce<typename camp::at<POLICY_DATA, camp::num<1>>::type, RAJA::operators::plus, RAJA::statement::Param<0>,
          RAJA::statement::Lambda<2, RAJA::Params<0>>
          // Device: Lambda 2 only gets executed on the "root" thread which received the reduced value.
        >
      > // end DEVICE_KERNEL
    >;
};

#endif  // RAJA_ENABLE_CUDA or RAJA_ENABLE_HIP

#endif  // __NESTED_LOOP_MULTI_LAMBDA_PARAM_REDUCE_SUM_IMPL_HPP__
