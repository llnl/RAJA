//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_VECTOR_CtorGetSet_HPP__
#define __TEST_TENSOR_VECTOR_CtorGetSet_HPP__

#include<RAJA/RAJA.hpp>

template <typename VECTOR_TYPE>
void CtorGetSetImpl()
{

  using vector_t = VECTOR_TYPE;
  using policy_t = typename vector_t::register_policy;
  using element_t = typename vector_t::element_type;


  std::vector<element_t> A(vector_t::s_num_elem);
  std::vector<element_t> get(vector_t::s_num_elem);
  std::vector<element_t> set(vector_t::s_num_elem);

  element_t * A_ptr = tensor_malloc<policy_t>(A);
  element_t * get_ptr = tensor_malloc<policy_t>(get);
  element_t * set_ptr = tensor_malloc<policy_t>(set);

  for(camp::idx_t i = 0;i < vector_t::s_num_elem;++ i){
    A[i] = (element_t)(i*2);
    get[i] = 0;
    set[i] = 0;
  }

  tensor_copy_to_device<policy_t>(A_ptr, A);
  tensor_copy_to_device<policy_t>(get_ptr, get);
  tensor_copy_to_device<policy_t>(set_ptr, set);

  // For Fixed vectors, only try with fixed length
  // For Stream vectors, try all lengths
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
    for(camp::idx_t N = 1; N <= vector_t::s_num_elem; ++ N){
      // load array A as vector
      vector_t vec;
      vec.load_packed_n(A_ptr, N);

      // try get operations
      for(camp::idx_t i = 0;i < N;++ i){
        get_ptr[i] = vec.get(i);
      }

      // try set and get operations
      for(camp::idx_t i = 0;i < N;++ i){
        vec.set((element_t)(i+1), i);
        set_ptr[i] = vec.get(i);
      }
    }
  });


  tensor_copy_to_host<policy_t>(get, get_ptr);
  tensor_copy_to_host<policy_t>(set, set_ptr);

  // For Fixed vectors, only try with fixed length
  // For Stream vectors, try all lengths
  for(camp::idx_t N = 1; N <= vector_t::s_num_elem; ++ N){

    // check get operations
    for(camp::idx_t i = 0;i < N;++ i){
      ASSERT_SCALAR_EQ(get[i], (element_t)(i*2));
    }

    for(camp::idx_t i = 0;i < N;++ i){
      ASSERT_SCALAR_EQ(set[i], (element_t)(i+1));
    }

  }

  tensor_free<policy_t>(A_ptr);
  tensor_free<policy_t>(get_ptr);
  tensor_free<policy_t>(set_ptr);
}



TYPED_TEST_P(TestTensorVector, CtorGetSet)
{
  CtorGetSetImpl<TypeParam>();
}


#endif
