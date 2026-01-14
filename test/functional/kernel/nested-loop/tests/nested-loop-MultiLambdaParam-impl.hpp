//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __NESTED_LOOP_MULTI_LAMBDA_PARAM_IMPL_HPP__
#define __NESTED_LOOP_MULTI_LAMBDA_PARAM_IMPL_HPP__

#include "RAJA_test-abs.hpp"

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
// Define list of nested loop types the MultiLambdaParam test supports.
//
//
using MultiLambdaParamSupportedLoopTypeList = camp::list<
  DEPTH_3,
  DEVICE_DEPTH_3
  >;

//
//
// Matrix-Matrix Multiplication test.
//
//
template <typename WORKING_RES, typename EXEC_POLICY, bool USE_RESOURCE>
void KernelNestedLoopTest(){

  constexpr static int N = 100;
  constexpr static int DIM = 2;

  camp::resources::Resource host_res{camp::resources::Host::get_default()};
  WORKING_RES work_res{WORKING_RES::get_default()};

  // Allocate Tests Data
  double* work_arrA = work_res.template allocate<double>(N*N);
  double* work_arrB = work_res.template allocate<double>(N*N);
  double* work_arrC = work_res.template allocate<double>(N*N);

  double* test_arrA = host_res.allocate<double>(N*N);
  double* test_arrB = host_res.allocate<double>(N*N);
  double* test_arrC = host_res.allocate<double>(N*N);

  double* check_arrC = host_res.allocate<double>(N*N);

  // Initialize RAJA Views
  RAJA::View< double, RAJA::Layout<DIM> > test_viewA(test_arrA, N, N);
  RAJA::View< double, RAJA::Layout<DIM> > test_viewB(test_arrB, N, N);
  RAJA::View< double, RAJA::Layout<DIM> > test_viewC(test_arrC, N, N);

  RAJA::View< double, RAJA::Layout<DIM> > work_viewA(work_arrA, N, N);
  RAJA::View< double, RAJA::Layout<DIM> > work_viewB(work_arrB, N, N);
  RAJA::View< double, RAJA::Layout<DIM> > work_viewC(work_arrC, N, N);

  // Initialize Data
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      test_viewA(row, col) = row;
      test_viewB(row, col) = col;
      test_viewB(row, col) = 0;
    }
  }

  work_res.memcpy(work_arrA, test_arrA, sizeof(double) * RAJA::stripIndexType(N*N));
  work_res.memcpy(work_arrB, test_arrB, sizeof(double) * RAJA::stripIndexType(N*N));
  work_res.memcpy(work_arrC, test_arrC, sizeof(double) * RAJA::stripIndexType(N*N));
  work_res.wait();

  // Calculate Test data
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {

      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += test_viewA(row, k) * test_viewB(k, col);
      }
      test_viewC(row, col) = dot;

    }
  }
  
  // Calculate Working data
  call_kernel<EXEC_POLICY, USE_RESOURCE>(
    RAJA::make_tuple(RAJA::RangeSegment{0, N},
                     RAJA::RangeSegment{0, N},
                     RAJA::RangeSegment{0, N}),

    RAJA::tuple<double>{0.0},

    // Resource
    work_res,

    // lambda 0
    [=] RAJA_HOST_DEVICE (double& dot) {
       dot = 0.0;
    },

    // lambda 1
    [=] RAJA_HOST_DEVICE (int col, int row, int k, double& dot) {
       dot += work_viewA(row, k) * work_viewB(k, col);
    },

    // lambda 2
    [=] RAJA_HOST_DEVICE (int col, int row, double& dot) {
       work_viewC(row, col) = dot;
    }

  );

  work_res.memcpy(check_arrC, work_arrC, sizeof(double) * RAJA::stripIndexType(N*N));
  work_res.wait();

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment{0, N*N}, [=] (RAJA::Index_type i) {
    ASSERT_TRUE( RAJA::test_abs(test_arrC[i] - check_arrC[i]) < 10e-8 );
  });

  work_res.deallocate(work_arrA);
  work_res.deallocate(work_arrB);
  work_res.deallocate(work_arrC);

  host_res.deallocate(test_arrA);
  host_res.deallocate(test_arrB);
  host_res.deallocate(test_arrC);

  host_res.deallocate(check_arrC);
}

//
//
// Defining the Kernel Loop structure for MultiLambdaParam Nested Loop Tests.
//
//
template<typename POLICY_TYPE, typename POLICY_DATA>
struct MultiLambdaParamNestedLoopExec;

template<typename POLICY_DATA>
struct MultiLambdaParamNestedLoopExec<DEPTH_3, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<0>>::type,
        RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<1>>::type,
          RAJA::statement::Lambda<0, RAJA::Params<0>>,  // dot = 0.0
          RAJA::statement::For<2, typename camp::at<POLICY_DATA, camp::num<2>>::type,
            RAJA::statement::Lambda<1> // inner loop: dot += ...
          >,
          RAJA::statement::Lambda<2, RAJA::Segs<0, 1>, RAJA::Params<0>>   // set C(row, col) = dot
        >
      >
    >;
};

#if defined(RAJA_ENABLE_CUDA) or defined(RAJA_ENABLE_HIP) or defined(RAJA_ENABLE_SYCL)

template<typename POLICY_DATA>
struct MultiLambdaParamNestedLoopExec<DEVICE_DEPTH_3, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::DEVICE_KERNEL<
      RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<0>>::type,
        RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<1>>::type,
          RAJA::statement::Lambda<0, RAJA::Params<0>>,  // dot = 0.0
          RAJA::statement::For<2, typename camp::at<POLICY_DATA, camp::num<2>>::type,
            RAJA::statement::Lambda<1> // inner loop: dot += ...
          >,
          RAJA::statement::Lambda<2, RAJA::Segs<0, 1>, RAJA::Params<0>>   // set C(row, col) = dot
        >
      >
      > // end CudaKernel
    >;
};

#endif  // RAJA_ENABLE_CUDA or RAJA_ENABLE_HIP

#endif  // __NESTED_LOOP_MULTI_LAMBDA_PARAM_IMPL_HPP__
