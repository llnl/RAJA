//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_REGISTER_Store_HPP__
#define __TEST_TENSOR_REGISTER_Store_HPP__

#include<RAJA/RAJA.hpp>

template <typename REGISTER_TYPE>
void StoreImpl()
{
  using reg_t = REGISTER_TYPE;
  using element_t = typename reg_t::element_type;
  using policy_t = typename reg_t::register_policy;

  static constexpr camp::idx_t num_elem = reg_t::s_num_elem;

  // Allocate
  std::vector<element_t> input0_vec(num_elem);
  element_t *input0_hptr = input0_vec.data();
  element_t *input0_dptr = tensor_malloc<policy_t, element_t>(num_elem);

  std::vector<element_t> output0_vec(10*num_elem);
  element_t *output0_dptr = tensor_malloc<policy_t, element_t>(10*num_elem);

  // Initialize input data
  for(camp::idx_t i = 0;i < num_elem; ++ i){
   input0_hptr[i] = (element_t)(i+1+NO_OPT_RAND);
  }

  tensor_copy_to_device<policy_t>(input0_dptr, input0_vec);


  // Initialize output
  for(camp::idx_t i = 0;i < 10*num_elem; ++ i){
    output0_vec[i] = (element_t)0;
  }
  tensor_copy_to_device<policy_t>(output0_dptr, output0_vec);


  // store stride-1 to pointer
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    // fill x
    reg_t x;
    for(camp::idx_t i = 0;i < num_elem; ++ i){
      x.set(input0_dptr[i], i);
    }

    x.store_packed(output0_dptr);

  });
  tensor_copy_to_host<policy_t>(output0_vec, output0_dptr);

  // check that we were able to copy
  for(camp::idx_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(output0_vec[i], input0_vec[i]);
  }



  for(camp::idx_t N = 0;N < num_elem; ++ N){

    // Initialize output
    for(camp::idx_t i = 0;i < 10*num_elem; ++ i){
      output0_vec[i] = (element_t)0;
    }
    tensor_copy_to_device<policy_t>(output0_dptr, output0_vec);


    // load stride-1 from pointer
    tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

      // fill x
      reg_t x;
      for(camp::idx_t i = 0;i < num_elem; ++ i){
        x.set(input0_dptr[i], i);
      }

      x.store_packed_n(output0_dptr, N);

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



  // Initialize output
  for(camp::idx_t i = 0;i < 10*num_elem; ++ i){
    output0_vec[i] = (element_t)0;
  }
  tensor_copy_to_device<policy_t>(output0_dptr, output0_vec);


  // load stride-2 from pointer
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    // fill x
    reg_t x;
    for(camp::idx_t i = 0;i < num_elem; ++ i){
      x.set(input0_dptr[i], i);
    }

    x.store_strided(output0_dptr, 2);

  });
  tensor_copy_to_host<policy_t>(output0_vec, output0_dptr);

  // check that we were able to copy using set/get
  for(camp::idx_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(output0_vec[2*i], input0_vec[i]);
  }



  for(camp::idx_t N = 0;N < num_elem; ++ N){

    // Initialize output
    for(camp::idx_t i = 0;i < 10*num_elem; ++ i){
      output0_vec[i] = (element_t)0;
    }
    tensor_copy_to_device<policy_t>(output0_dptr, output0_vec);



    // load stride-2 from pointer
    tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

      // fill x
      reg_t x;
      for(camp::idx_t i = 0;i < num_elem; ++ i){
        x.set(input0_dptr[i], i);
      }

      x.store_strided_n(output0_dptr, 2, N);

    });
    tensor_copy_to_host<policy_t>(output0_vec, output0_dptr);

    // check that we were able to copy using set/get
    for(camp::idx_t i = 0;i < num_elem; ++ i){
      if(i < N){
        ASSERT_SCALAR_EQ(output0_vec[2*i], input0_vec[i]);
      }
      else{
        ASSERT_SCALAR_EQ(output0_vec[2*i], (element_t)0);
      }
    }
  }


  //
  // Cleanup
  //
  tensor_free<policy_t>(input0_dptr);
  tensor_free<policy_t>(output0_dptr);
}



TYPED_TEST_P(TestTensorRegister, Store)
{
  StoreImpl<TypeParam>();
}


#endif
