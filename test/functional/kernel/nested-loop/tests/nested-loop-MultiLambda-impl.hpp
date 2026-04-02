//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __NESTED_LOOP_MULTI_LAMBDA_IMPL_HPP__
#define __NESTED_LOOP_MULTI_LAMBDA_IMPL_HPP__

#include "RAJA_test-abs.hpp"

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
// Define list of nested loop types the MultiLambda test supports.
//
//
using MultiLambdaSupportedLoopTypeList = camp::list<
  DEPTH_2,
  DEPTH_2_COLLAPSE,
  DEVICE_DEPTH_2>;

//
//
// Simple 5 point matrix calculation test.
//
//
template <typename WORKING_RES, typename EXEC_POLICY, bool USE_RESOURCE>
void KernelNestedLoopTest(){
  constexpr static int N = 1000;
  constexpr static int DIM = 2;

  camp::resources::Resource host_res{camp::resources::Host::get_default()};
  WORKING_RES work_res{WORKING_RES::get_default()};

  // Allocate Tests Data
  double* work_arrA = work_res.template allocate<double>(N*N);
  double* work_arrB = work_res.template allocate<double>(N*N);

  double* test_arrA = host_res.allocate<double>(N*N);
  double* test_arrB = host_res.allocate<double>(N*N);

  double* check_arrA = host_res.allocate<double>(N*N);
  double* check_arrB = host_res.allocate<double>(N*N);

  // Initialize Data
  for (RAJA::Index_type i = 0; i < N*N; i++) {
    test_arrA[i] = i * 1.2;  test_arrB[i] = i * 0.5;
  }
  work_res.memcpy(work_arrA, test_arrA, sizeof(double) * RAJA::stripIndexType(N*N));
  work_res.memcpy(work_arrB, test_arrB, sizeof(double) * RAJA::stripIndexType(N*N));

  // Initialize RAJA Views
  RAJA::View< double, RAJA::Layout<DIM> > test_viewA(test_arrA, N, N);
  RAJA::View< double, RAJA::Layout<DIM> > test_viewB(test_arrB, N, N);

  // Calculate Test data
  for (RAJA::Index_type i = 1; i < N-1; ++i ) {
    for (RAJA::Index_type j = 1; j < N-1; ++j ) {
      test_viewB(i,j) = 0.2 * (test_viewA(i,j) + test_viewA(i,j-1) + test_viewA(i,j+1) + test_viewA(i+1,j) + test_viewA(i-1,j));
    }
  }
  for (RAJA::Index_type i = 1; i < N-1; ++i ) {
    for (RAJA::Index_type j = 1; j < N-1; ++j ) {
      test_viewA(i,j) = 0.2 * (test_viewB(i,j) + test_viewB(i,j-1) + test_viewB(i,j+1) + test_viewB(i+1,j) + test_viewB(i-1,j));
    }
  } 

  RAJA::View< double, RAJA::Layout<DIM> > work_viewA(work_arrA, N, N);
  RAJA::View< double, RAJA::Layout<DIM> > work_viewB(work_arrB, N, N);

  call_kernel<EXEC_POLICY, USE_RESOURCE>(
    RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                     RAJA::RangeSegment{1, N-1}),

    // Resource
    work_res,

    // lambda 0
    [=] RAJA_HOST_DEVICE (RAJA::Index_type i, RAJA::Index_type j) {
      work_viewB(i,j) = 0.2 * (work_viewA(i,j) + work_viewA(i,j-1) + work_viewA(i,j+1) + work_viewA(i+1,j) + work_viewA(i-1,j));
    },

    // lambda 1
    [=] RAJA_HOST_DEVICE (RAJA::Index_type i, RAJA::Index_type j) {
      work_viewA(i,j) = 0.2 * (work_viewB(i,j) + work_viewB(i,j-1) + work_viewB(i,j+1) + work_viewB(i+1,j) + work_viewB(i-1,j));
    }
  );

  work_res.memcpy(check_arrA, work_arrA, sizeof(double) * RAJA::stripIndexType(N*N));
  work_res.memcpy(check_arrB, work_arrB, sizeof(double) * RAJA::stripIndexType(N*N));

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment{0, N*N}, [=] (RAJA::Index_type i) {
    ASSERT_TRUE( RAJA::test_abs(test_arrA[i] - check_arrA[i]) < 10e-8 );
    ASSERT_TRUE( RAJA::test_abs(test_arrB[i] - check_arrB[i]) < 10e-8 );
  });

  work_res.deallocate(work_arrA);
  work_res.deallocate(work_arrB);

  host_res.deallocate(test_arrA);
  host_res.deallocate(test_arrB);

  host_res.deallocate(check_arrA);
  host_res.deallocate(check_arrB);
}

//
//
// Defining the Kernel Loop structure for MultiLambda Nested Loop Tests.
//
//
template<typename POLICY_TYPE, typename POLICY_DATA>
struct MultiLambdaNestedLoopExec;

template<typename POLICY_DATA>
struct MultiLambdaNestedLoopExec<DEPTH_2, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<0>>::type,
        RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<1>>::type,
          RAJA::statement::Lambda<0>
        >
      >,
      RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<0>>::type,
        RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<1>>::type,
          RAJA::statement::Lambda<1>
        >
      >
    >;
};

template<typename POLICY_DATA>
struct MultiLambdaNestedLoopExec<DEPTH_2_COLLAPSE, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::Collapse< typename camp::at<POLICY_DATA, camp::num<0>>::type,
        RAJA::ArgList<1,0>,
        RAJA::statement::Lambda<0>
      >,
      RAJA::statement::Collapse< typename camp::at<POLICY_DATA, camp::num<0>>::type,
        RAJA::ArgList<1,0>,
        RAJA::statement::Lambda<1>
      >
    >;
};

#if defined(RAJA_ENABLE_CUDA) or defined(RAJA_ENABLE_HIP) or defined(RAJA_ENABLE_SYCL)

template<typename POLICY_DATA>
struct MultiLambdaNestedLoopExec<DEVICE_DEPTH_2, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::DEVICE_KERNEL<
        RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<0>>::type,
          RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<1>>::type,
            RAJA::statement::Lambda<0>
          >
        >
      >,
      RAJA::statement::DEVICE_KERNEL<
        RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<0>>::type,
          RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<1>>::type,
            RAJA::statement::Lambda<1>
          >
        >
      >
    >;
};

#endif  // RAJA_ENABLE_CUDA or RAJA_ENABLE_HIP

#endif  // __NESTED_LOOP_MULTI_LAMBDA_IMPL_HPP__
