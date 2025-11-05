//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_REGISTER_Load_HPP__
#define __TEST_TENSOR_REGISTER_Load_HPP__

#include<RAJA/RAJA.hpp>

template <typename REGISTER_TYPE>
void LoadImpl()
{
  using reg_t = REGISTER_TYPE;
  using element_t = typename reg_t::element_type;
  using policy_t = typename reg_t::register_policy;

  static constexpr camp::idx_t num_elem = reg_t::s_num_elem;

  // Allocate
  std::vector<element_t> input0_vec(10*num_elem);
  element_t *input0_hptr = input0_vec.data();
  element_t *input0_dptr = tensor_malloc<policy_t, element_t>(10*num_elem);

  std::vector<element_t> output0_vec(num_elem);
  element_t *output0_dptr = tensor_malloc<policy_t, element_t>(num_elem);

  // Initialize input data
  for(camp::idx_t i = 0;i < 10*num_elem; ++ i){
   input0_hptr[i] = (element_t)(i+1+NO_OPT_RAND);
  }

  tensor_copy_to_device<policy_t>(input0_dptr, input0_vec);




  // load stride-1 from pointer
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    // fill x using set
    reg_t x;
    x.load_packed(input0_dptr);

    // extract from x using get
    for(camp::idx_t i = 0;i < num_elem; ++ i){
      output0_dptr[i] = x.get(i);
    }

  });
  tensor_copy_to_host<policy_t>(output0_vec, output0_dptr);

  // check that we were able to copy using set/get
  for(camp::idx_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(output0_vec[i], input0_vec[i]);
  }



  for(camp::idx_t N = 0;N < num_elem; ++ N){
    // load stride-1 from pointer
    tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

      // fill x using set
      reg_t x;
      x.load_packed_n(input0_dptr, N);

      // extract from x using get
      for(camp::idx_t i = 0;i < num_elem; ++ i){
        output0_dptr[i] = x.get(i);
      }

    });
    tensor_copy_to_host<policy_t>(output0_vec, output0_dptr);

    // check that we were able to copy using set/get
    for(camp::idx_t i = 0;i < num_elem; ++ i){
      if(i < N){
        ASSERT_SCALAR_EQ(output0_vec[i], input0_vec[i]);
      }
      else{
        ASSERT_SCALAR_EQ(output0_vec[i], (element_t)0);
      }
    }
  }




  // load stride-2 from pointer
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    // fill x using set
    reg_t x;
    x.load_strided(input0_dptr, 2);

    // extract from x using get
    for(camp::idx_t i = 0;i < num_elem; ++ i){
      output0_dptr[i] = x.get(i);
    }

  });
  tensor_copy_to_host<policy_t>(output0_vec, output0_dptr);

  // check that we were able to copy using set/get
  for(camp::idx_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(output0_vec[i], input0_vec[i*2]);
  }



  for(camp::idx_t N = 0;N < num_elem; ++ N){
    // load stride-2 from pointer
    tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

      // fill x using set
      reg_t x;
      x.load_strided_n(input0_dptr, 2, N);

      // extract from x using get
      for(camp::idx_t i = 0;i < num_elem; ++ i){
        output0_dptr[i] = x.get(i);
      }

    });
    tensor_copy_to_host<policy_t>(output0_vec, output0_dptr);

    // check that we were able to copy using set/get
    for(camp::idx_t i = 0;i < num_elem; ++ i){
      if(i < N){
        ASSERT_SCALAR_EQ(output0_vec[i], input0_vec[i*2]);
      }
      else{
        ASSERT_SCALAR_EQ(output0_vec[i], (element_t)0);
      }
    }
  }


  //
  // Cleanup
  //
  tensor_free<policy_t>(input0_dptr);
  tensor_free<policy_t>(output0_dptr);
}



TYPED_TEST_P(TestTensorRegister, Load)
{
  LoadImpl<TypeParam>();
}


#endif
