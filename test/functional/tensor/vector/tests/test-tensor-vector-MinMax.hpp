//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_VECTOR_MinMax_HPP__
#define __TEST_TENSOR_VECTOR_MinMax_HPP__

#include<RAJA/RAJA.hpp>

template <typename VECTOR_TYPE>
void MinMaxImpl()
{

  using vector_t = VECTOR_TYPE;
  using policy_t = typename vector_t::register_policy;
  using element_t = typename vector_t::element_type;

  std::vector<element_t> A(vector_t::s_num_elem);
  std::vector<element_t> ex_min(1);
  std::vector<element_t> ex_max(1);

  element_t * A_ptr = tensor_malloc<policy_t>(A);
  element_t * ex_min_ptr = tensor_malloc<policy_t>(ex_min);
  element_t * ex_max_ptr = tensor_malloc<policy_t>(ex_max);

  for(camp::idx_t i = 0;i < vector_t::s_num_elem;++ i){
    A[i] = (element_t)i;
  }
  ex_min[0] = (element_t)99999999;
  ex_max[0] = (element_t)0;


  tensor_copy_to_device<policy_t>(A_ptr, A);
  tensor_copy_to_device<policy_t>(ex_min_ptr, ex_min);
  tensor_copy_to_device<policy_t>(ex_max_ptr, ex_max);

  // For Fixed vectors, only try with fixed length
  // For Stream vectors, try all lengths
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
    for(camp::idx_t N = 1; N <= vector_t::s_num_elem; ++ N){

      // load array A as vector
      vector_t vec;
      vec.load_packed_n(A_ptr, N);

      ex_min_ptr[0] = vec.min_n(N);
      ex_max_ptr[0] = vec.max_n(N);
    }
  });

  tensor_copy_to_host<policy_t>(ex_min, ex_min_ptr);
  tensor_copy_to_host<policy_t>(ex_max, ex_max_ptr);

  // check min
  ASSERT_SCALAR_EQ(ex_min[0], (element_t)0);

  // check max
  ASSERT_SCALAR_EQ(ex_max[0], (element_t)(vector_t::s_num_elem-1));

  tensor_free<policy_t>(A_ptr);
  tensor_free<policy_t>(ex_min_ptr);
  tensor_free<policy_t>(ex_max_ptr);
}



TYPED_TEST_P(TestTensorVector, MinMax)
{
  MinMaxImpl<TypeParam>();
}


#endif
