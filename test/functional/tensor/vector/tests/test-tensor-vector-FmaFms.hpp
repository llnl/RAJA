//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_VECTOR_FmaFms_HPP__
#define __TEST_TENSOR_VECTOR_FmaFms_HPP__

#include<RAJA/RAJA.hpp>

template <typename VECTOR_TYPE>
void FmaFmsImpl()
{

  using vector_t = VECTOR_TYPE;
  using policy_t = typename vector_t::register_policy;
  using element_t = typename vector_t::element_type;

  std::vector<element_t> A(vector_t::s_num_elem);
  std::vector<element_t> B(vector_t::s_num_elem);
  std::vector<element_t> C(vector_t::s_num_elem);
  std::vector<element_t> fma(vector_t::s_num_elem);
  std::vector<element_t> fms(vector_t::s_num_elem);

  element_t * A_ptr = tensor_malloc<policy_t>(A);
  element_t * B_ptr = tensor_malloc<policy_t>(B);
  element_t * C_ptr = tensor_malloc<policy_t>(C);
  element_t * fma_ptr = tensor_malloc<policy_t>(fma);
  element_t * fms_ptr = tensor_malloc<policy_t>(fms);

  for(camp::idx_t i = 0;i < vector_t::s_num_elem;++ i){
    A[i] = (element_t)i;
    B[i] = (element_t)i*2;
    C[i] = (element_t)i*3;
    fma[i] = 0;
    fms[i] = 0;
  }

  tensor_copy_to_device<policy_t>(A_ptr, A);
  tensor_copy_to_device<policy_t>(B_ptr, B);
  tensor_copy_to_device<policy_t>(C_ptr, C);
  tensor_copy_to_device<policy_t>(fma_ptr, fma);
  tensor_copy_to_device<policy_t>(fms_ptr, fms);

  // For Fixed vectors, only try with fixed length
  // For Stream vectors, try all lengths
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
    for(camp::idx_t N = 1; N <= vector_t::s_num_elem; ++ N){

      // load arrays as vectors
      vector_t vec_A;
      vec_A.load_packed_n(A_ptr, N);

      vector_t vec_B;
      vec_B.load_packed_n(B_ptr, N);

      vector_t vec_C;
      vec_C.load_packed_n(C_ptr, N);


      // try FMA (A*B+C)

      vector_t _fma = vec_A.multiply_add(vec_B, vec_C);
      for(camp::idx_t i = 0;i < N;++ i){
        fma_ptr[i] = _fma.get(i);
      }

      // try FMS (A*B-C)
      vector_t _fms = vec_A.multiply_subtract(vec_B, vec_C);
      for(camp::idx_t i = 0;i < N;++ i){
        fms_ptr[i] = _fms.get(i);
      }
    }
  });

  tensor_copy_to_host<policy_t>(fma, fma_ptr);
  tensor_copy_to_host<policy_t>(fms, fms_ptr);

  // For Fixed vectors, only try with fixed length
  // For Stream vectors, try all lengths
  for(camp::idx_t N = 1; N <= vector_t::s_num_elem; ++ N){

    // check FMA (A*B+C)

    for(camp::idx_t i = 0;i < N;++ i){
      ASSERT_SCALAR_EQ(fma[i], A[i]*B[i]+C[i]);
    }

    // check FMS (A*B-C)
    for(camp::idx_t i = 0;i < N;++ i){
      ASSERT_SCALAR_EQ(fms[i], A[i]*B[i]-C[i]);
    }

  }

  tensor_free<policy_t>(A_ptr);
  tensor_free<policy_t>(B_ptr);
  tensor_free<policy_t>(C_ptr);
  tensor_free<policy_t>(fma_ptr);
  tensor_free<policy_t>(fms_ptr);
}



TYPED_TEST_P(TestTensorVector, FmaFms)
{
  FmaFmsImpl<TypeParam>();
}


#endif
