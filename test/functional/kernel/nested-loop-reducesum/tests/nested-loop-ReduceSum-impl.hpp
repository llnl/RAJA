//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __NESTED_LOOP_REDUCESUM_IMPL_HPP__
#define __NESTED_LOOP_REDUCESUM_IMPL_HPP__

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

//
//
// Define list of nested loop types the ReduceSum test supports.
//
//
using ReduceSumSupportedLoopTypeList = camp::list<
  DEPTH_3_REDUCESUM,
  DEPTH_3_REDUCESUM_SEQ_INNER,
  DEPTH_3_REDUCESUM_SEQ_OUTER,
  DEVICE_DEPTH_3_REDUCESUM,
  DEVICE_DEPTH_3_REDUCESUM_SEQ_INNER,
  DEVICE_DEPTH_3_REDUCESUM_SEQ_OUTER>;

//
//
// ReduceSum 3D Matrix index calculation per element.
//
//
template <typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POL, bool USE_RESOURCE>
void KernelNestedLoopTest(const DEPTH_3_REDUCESUM&,
                          const RAJA::Index_type dim0,
                          const RAJA::Index_type dim1,
                          const RAJA::Index_type dim2){
  WORKING_RES work_res{WORKING_RES::get_default()};
  camp::resources::Resource erased_work_res{work_res};

  RAJA::Index_type flatSize = dim0 * dim1 * dim2;
  RAJA::Index_type* work_array;
  RAJA::Index_type* check_array;
  RAJA::Index_type* test_array;

  allocateForallTestData<RAJA::Index_type>(flatSize,
                                     erased_work_res,
                                     &work_array,
                                     &check_array,
                                     &test_array);

  RAJA::TypedRangeSegment<RAJA::Index_type> rangeflat(0,flatSize);
  RAJA::TypedRangeSegment<RAJA::Index_type> range0(0, dim0);
  RAJA::TypedRangeSegment<RAJA::Index_type> range1(0, dim1);
  RAJA::TypedRangeSegment<RAJA::Index_type> range2(0, dim2);

  std::iota(test_array, test_array + RAJA::stripIndexType(flatSize), 0);

  erased_work_res.memcpy(work_array, test_array, sizeof(RAJA::Index_type) * RAJA::stripIndexType(flatSize));

  constexpr int Depth = 3;
  RAJA::View< RAJA::Index_type, RAJA::Layout<Depth> > work_view(work_array, dim0, dim1, dim2);

  RAJA::ReduceSum<RAJA::seq_reduce, RAJA::Index_type> hostsum(0);
  RAJA::ReduceSum<REDUCE_POL, RAJA::Index_type> worksum(0);

  call_kernel<EXEC_POLICY, USE_RESOURCE>(RAJA::make_tuple(range0, range1, range2), work_res,
                            [=] RAJA_HOST_DEVICE (RAJA::Index_type i, RAJA::Index_type j, RAJA::Index_type k) {
                              worksum += work_view(i,j,k);
                            });

  RAJA::forall<RAJA::seq_exec>(rangeflat, [=] (RAJA::Index_type i) {
    hostsum += test_array[RAJA::stripIndexType(i)];
  });

  ASSERT_EQ(hostsum.get(), worksum.get());

  deallocateForallTestData<RAJA::Index_type>(erased_work_res,
                                       work_array,
                                       check_array,
                                       test_array);
}

// DEVICE_ and DEPTH_3_REDUCESUM_SEQ_ execution policies use the above DEPTH_3_REDUCESUM test.
template <typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POL, bool USE_RESOURCE, typename... Args>
void KernelNestedLoopTest(const DEPTH_3_REDUCESUM_SEQ_OUTER&, Args... args){
  KernelNestedLoopTest<WORKING_RES, EXEC_POLICY, REDUCE_POL, USE_RESOURCE>(DEPTH_3_REDUCESUM(), args...);
}

template <typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POL, bool USE_RESOURCE, typename... Args>
void KernelNestedLoopTest(const DEPTH_3_REDUCESUM_SEQ_INNER&, Args... args){
  KernelNestedLoopTest<WORKING_RES, EXEC_POLICY, REDUCE_POL, USE_RESOURCE>(DEPTH_3_REDUCESUM(), args...);
}

template <typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POL, bool USE_RESOURCE, typename... Args>
void KernelNestedLoopTest(const DEVICE_DEPTH_3_REDUCESUM&, Args... args){
  KernelNestedLoopTest<WORKING_RES, EXEC_POLICY, REDUCE_POL, USE_RESOURCE>(DEPTH_3_REDUCESUM(), args...);
}

template <typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POL, bool USE_RESOURCE, typename... Args>
void KernelNestedLoopTest(const DEVICE_DEPTH_3_REDUCESUM_SEQ_OUTER&, Args... args){
  KernelNestedLoopTest<WORKING_RES, EXEC_POLICY, REDUCE_POL, USE_RESOURCE>(DEPTH_3_REDUCESUM(), args...);
}

template <typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POL, bool USE_RESOURCE, typename... Args>
void KernelNestedLoopTest(const DEVICE_DEPTH_3_REDUCESUM_SEQ_INNER&, Args... args){
  KernelNestedLoopTest<WORKING_RES, EXEC_POLICY, REDUCE_POL, USE_RESOURCE>(DEPTH_3_REDUCESUM(), args...);
}

//
//
// Defining the Kernel Loop structure for ReduceSum Nested Loop Tests.
//
//
template<typename POLICY_TYPE, typename REDUCE_POL, typename POLICY_DATA>
struct ReduceSumNestedLoopExec;

template<typename REDUCE_POL, typename POLICY_DATA>
struct ReduceSumNestedLoopExec<DEPTH_3_REDUCESUM, REDUCE_POL, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<0>>::type,
        RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<1>>::type,
          RAJA::statement::For<2, typename camp::at<POLICY_DATA, camp::num<2>>::type,
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;
};

template<typename REDUCE_POL, typename POLICY_DATA>
struct ReduceSumNestedLoopExec<DEPTH_3_REDUCESUM_SEQ_OUTER, REDUCE_POL, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::For<0, RAJA::seq_exec,
        RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<1>>::type,
          RAJA::statement::For<2, typename camp::at<POLICY_DATA, camp::num<2>>::type,
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;
};

template<typename REDUCE_POL, typename POLICY_DATA>
struct ReduceSumNestedLoopExec<DEPTH_3_REDUCESUM_SEQ_INNER, REDUCE_POL, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<0>>::type,
        RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<1>>::type,
          RAJA::statement::For<2, RAJA::seq_exec,
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;
};

#if defined(RAJA_ENABLE_CUDA) or defined(RAJA_ENABLE_HIP)

template<typename REDUCE_POL, typename POLICY_DATA>
struct ReduceSumNestedLoopExec<DEVICE_DEPTH_3_REDUCESUM, REDUCE_POL, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::DEVICE_KERNEL<
        RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<0>>::type,
          RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<1>>::type,
            RAJA::statement::For<2, typename camp::at<POLICY_DATA, camp::num<2>>::type,
              RAJA::statement::Lambda<0>
            >
          >
        >
      > // end DEVICE_KERNEL
    >;
};

template<typename REDUCE_POL, typename POLICY_DATA>
struct ReduceSumNestedLoopExec<DEVICE_DEPTH_3_REDUCESUM_SEQ_OUTER, REDUCE_POL, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::DEVICE_KERNEL<
        RAJA::statement::For<0, RAJA::seq_exec,
          RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<1>>::type,
            RAJA::statement::For<2, typename camp::at<POLICY_DATA, camp::num<2>>::type,
              RAJA::statement::Lambda<0>
            >
          >
        >
      > // end DEVICE_KERNEL
    >;
};

template<typename REDUCE_POL, typename POLICY_DATA>
struct ReduceSumNestedLoopExec<DEVICE_DEPTH_3_REDUCESUM_SEQ_INNER, REDUCE_POL, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::DEVICE_KERNEL<
        RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<0>>::type,
          RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<1>>::type,
            RAJA::statement::For<2, RAJA::seq_exec,
              RAJA::statement::Lambda<0>
            >
          >
        >
      > // end DEVICE_KERNEL
    >;
};

#endif  // RAJA_ENABLE_CUDA or RAJA_ENABLE_HIP

#endif  // __NESTED_LOOP_REDUCESUM_IMPL_HPP__
