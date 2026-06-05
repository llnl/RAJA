//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_VECTOR_SumDot_HPP__
#define __TEST_TENSOR_VECTOR_SumDot_HPP__

#include<RAJA/RAJA.hpp>

template <typename VECTOR_TYPE>
void SumDotImpl()
{

  using vector_t = VECTOR_TYPE;
  using policy_t = typename vector_t::register_policy;
  using element_t = typename vector_t::element_type;

  std::vector<element_t> A(vector_t::s_num_elem);
  std::vector<element_t> ex_sum(1);
  std::vector<element_t> ex_dot(1);

  element_t host_sum = 0;
  element_t host_dot = 0;

  element_t * A_ptr = tensor_malloc<policy_t>(A);
  element_t * ex_sum_ptr = tensor_malloc<policy_t>(ex_sum);
  element_t * ex_dot_ptr = tensor_malloc<policy_t>(ex_dot);

  for(camp::idx_t i = 0;i < vector_t::s_num_elem;++ i){
    A[i] = (element_t)i;
  }

  ex_sum[0] = (element_t)0;
  ex_dot[0] = (element_t)0;

  // compute expected values on host
  for(camp::idx_t i = 0; i < vector_t::s_num_elem; ++i){
    host_sum += A[i];
    host_dot += A[i]*A[i];
  }

  tensor_copy_to_device<policy_t>(A_ptr, A);
  tensor_copy_to_device<policy_t>(ex_sum_ptr, ex_sum);
  tensor_copy_to_device<policy_t>(ex_dot_ptr, ex_dot);

  // For Fixed vectors, only try with fixed length
  // For Stream vectors, try all lengths
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
    // load array A as vector
    vector_t vec;
    vec.load_packed_n(A_ptr, vector_t::s_num_elem);

    ex_sum_ptr[0] = vec.sum();
    ex_dot_ptr[0] = vec.dot(vec);
  });

  tensor_copy_to_host<policy_t>(ex_sum, ex_sum_ptr);
  tensor_copy_to_host<policy_t>(ex_dot, ex_dot_ptr);

  // check sum
  ASSERT_SCALAR_EQ(ex_sum[0], host_sum);

  // check dot
  ASSERT_SCALAR_EQ(ex_dot[0], host_dot);

  tensor_free<policy_t>(A_ptr);
  tensor_free<policy_t>(ex_sum_ptr);
  tensor_free<policy_t>(ex_dot_ptr);
}



TYPED_TEST_P(TestTensorVector, SumDot)
{
  SumDotImpl<TypeParam>();
}


#endif
