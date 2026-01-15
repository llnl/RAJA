//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_VECTOR_ForallVectorRef1d_HPP__
#define __TEST_TENSOR_VECTOR_ForallVectorRef1d_HPP__

#include<RAJA/RAJA.hpp>

RAJA_INDEX_VALUE( TX, "TX" );

template <typename VECTOR_TYPE>
void ForallVectorRef1dImpl()
{

  using vector_t = VECTOR_TYPE;
  using policy_t = typename vector_t::register_policy;
  using element_t = typename vector_t::element_type;


  size_t N = 10*vector_t::s_num_elem+1;
  // If we are not using fixed vectors, add some random number of elements
  // to the array to test some postamble code generation.
    //N += (size_t)(100*NO_OPT_RAND);

  std::vector<element_t> A(N);
  std::vector<element_t> B(N);
  std::vector<element_t> C(N);

  element_t * A_ptr = tensor_malloc<policy_t>(A);
  element_t * B_ptr = tensor_malloc<policy_t>(B);
  element_t * C_ptr = tensor_malloc<policy_t>(C);

  for(size_t i = 0;i < N; ++ i){
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
    B[i] = (element_t)(NO_OPT_RAND*1000.0);
    C[i] = 0.0;
  }

  RAJA::TypedView<element_t, RAJA::Layout<1>, TX> X(A.data(), N);
  RAJA::TypedView<element_t, RAJA::Layout<1>, TX> Y(B.data(), N);
  RAJA::TypedView<element_t, RAJA::Layout<1>, TX> Z(C.data(), N);

  RAJA::TypedView<element_t, RAJA::Layout<1>, TX> X_d(A_ptr, N);
  RAJA::TypedView<element_t, RAJA::Layout<1>, TX> Y_d(B_ptr, N);
  RAJA::TypedView<element_t, RAJA::Layout<1>, TX> Z_d(C_ptr, N);

  using idx_t = RAJA::expt::VectorIndex<int, vector_t>;

  auto all = idx_t::all();

  // evaluate on all() range
  tensor_copy_to_device<policy_t>(A_ptr, A);
  tensor_copy_to_device<policy_t>(B_ptr, B);
  tensor_copy_to_device<policy_t>(C_ptr, C);

  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
    Z_d[all] = 3 + (X_d[all]*(5/Y_d[all])) + 9;
  });

  tensor_copy_to_host<policy_t>(C, C_ptr);

//  for(size_t i = 0;i < N; ++ i){
//    printf("%lf ", (double)C[i]);
//  }
//  printf("\n\n");

  for(size_t i = 0;i < N;i ++){
    ASSERT_SCALAR_EQ(element_t(3+(A[i]*(5/B[i]))+9), C[i]);
  }


  // evaluate complex left side division on all() range
  for(size_t i = 0;i < N; ++ i){
    C[i] = 0.0;
  }

  tensor_copy_to_device<policy_t>(C_ptr, C);

  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
    Z_d[all] = 3 + ((X_d[all]*Y_d[all])/Y_d[all]) + 9;
  });

  tensor_copy_to_host<policy_t>(C, C_ptr);

  for(size_t i = 0;i < N;i ++){
    ASSERT_SCALAR_EQ(element_t(3+((A[i]*B[i])/B[i]))+9, C[i]);
  }

  // evaluate on a subrange [N/2, N)
  for(size_t i = 0;i < N; ++ i){
    C[i] = 0.0;
  }

  tensor_copy_to_device<policy_t>(C_ptr, C);

  // evaluate on a subrange [N/2, N)
  auto some = idx_t::range(N/2, N);
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
    Z_d[some] = 3.+ (X_d[some]*(5/Y_d[some])) + 9;
  });

  tensor_copy_to_host<policy_t>(A, A_ptr);
  tensor_copy_to_host<policy_t>(B, B_ptr);
  tensor_copy_to_host<policy_t>(C, C_ptr);

  for(size_t i = 0;i < N/2;i ++){
    ASSERT_SCALAR_EQ(0, C[i]);
  }
  for(size_t i = N/2;i < N;i ++){
    ASSERT_SCALAR_EQ(element_t(3+(A[i]*(5/B[i]))+9), C[i]);
  }




  // evaluate on a subrange [0, N/2) using a forall statement
  for(size_t i = 0;i < N; ++ i){
    C[i] = 0.0;
  }

  // vector_exec only works on the host due to its use of RAJA::seq_exec
  RAJA::forall<RAJA::expt::vector_exec<vector_t>>(RAJA::TypedRangeSegment<TX>(0,N/2),
      [=](TX i){

     Z[i] = 3 + (X[i]*(5/Y[i])) + 9;
  });

  for(size_t i = 0;i < N/2;i ++){
    ASSERT_SCALAR_EQ(element_t(3+(A[i]*(5/B[i]))+9), C[i]);
  }
  for(size_t i = N/2;i < N;i ++){
    ASSERT_SCALAR_EQ(0, C[i]);
  }

  tensor_free<policy_t>(A_ptr);
  tensor_free<policy_t>(B_ptr);
  tensor_free<policy_t>(C_ptr);
}



TYPED_TEST_P(TestTensorVector, ForallVectorRef1d)
{
  ForallVectorRef1dImpl<TypeParam>();
}


#endif
