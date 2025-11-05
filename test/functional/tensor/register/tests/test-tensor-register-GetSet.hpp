//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_REGISTER_GetSet_HPP__
#define __TEST_TENSOR_REGISTER_GetSet_HPP__

#include<RAJA/RAJA.hpp>

template <typename REGISTER_TYPE>
void GetSetImpl()
{
  using reg_t = REGISTER_TYPE;
  using element_t = typename reg_t::element_type;
  using policy_t = typename reg_t::register_policy;

  static constexpr camp::idx_t num_elem = reg_t::s_num_elem;

  // Allocate
  std::vector<element_t> input0_vec(num_elem);
  element_t *input0_hptr = input0_vec.data();
  element_t *input0_dptr = tensor_malloc<policy_t, element_t>(num_elem);

  std::vector<element_t> output0_vec(num_elem);
  element_t *output0_dptr = tensor_malloc<policy_t, element_t>(num_elem);

  // Initialize input data
  for(camp::idx_t i = 0;i < num_elem; ++ i){
   input0_hptr[i] = (element_t)(i+1+NO_OPT_RAND);
  }

  tensor_copy_to_device<policy_t>(input0_dptr, input0_vec);

  // Test set and get operations
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    // fill x using set
    reg_t x;
    for(camp::idx_t i = 0;i < num_elem; ++ i){
      x.set(input0_dptr[i], i);
    }

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


  //
  // test copy construction
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    // fill x using set
    reg_t x;
    for(camp::idx_t i = 0;i < num_elem; ++ i){
      x.set(input0_dptr[i], i);
    }

    reg_t cc(x);

    // extract from x using get
    for(camp::idx_t i = 0;i < num_elem; ++ i){
      output0_dptr[i] = cc.get(i);
    }

  });
  tensor_copy_to_host<policy_t>(output0_vec, output0_dptr);

  // check that we were able to copy using set/get
  for(camp::idx_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(output0_vec[i], input0_vec[i]);
  }




  //
  // test explicit copy
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    // fill x using set
    reg_t x;
    for(camp::idx_t i = 0;i < num_elem; ++ i){
      x.set(input0_dptr[i], i);
    }

    reg_t cc;
    cc.copy(x);

    // extract from x using get
    for(camp::idx_t i = 0;i < num_elem; ++ i){
      output0_dptr[i] = cc.get(i);
    }

  });
  tensor_copy_to_host<policy_t>(output0_vec, output0_dptr);

  // check that we were able to copy using set/get
  for(camp::idx_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(output0_vec[i], input0_vec[i]);
  }




  //
  // test assignment
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    // fill x using set
    reg_t x;
    for(camp::idx_t i = 0;i < num_elem; ++ i){
      x.set(input0_dptr[i], i);
    }

    reg_t cc = x;

    // extract from x using get
    for(camp::idx_t i = 0;i < num_elem; ++ i){
      output0_dptr[i] = cc.get(i);
    }

  });
  tensor_copy_to_host<policy_t>(output0_vec, output0_dptr);

  // check that we were able to copy using set/get
  for(camp::idx_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(output0_vec[i], input0_vec[i]);
  }




  //
  // test scalar construction (broadcast)
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){


    reg_t cc = (element_t) 5;

    // extract from x using get
    for(camp::idx_t i = 0;i < num_elem; ++ i){
      output0_dptr[i] = cc.get(i);
    }

  });
  tensor_copy_to_host<policy_t>(output0_vec, output0_dptr);

  // check that we were able to copy using set/get
  for(camp::idx_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(output0_vec[i], (element_t)5);
  }





  //
  // test scalar broadcast by assignment
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){


    reg_t cc = (element_t) 0;
    cc = (element_t) 11.0;

    // extract from x using get
    for(camp::idx_t i = 0;i < num_elem; ++ i){
      output0_dptr[i] = cc.get(i);
    }

  });
  tensor_copy_to_host<policy_t>(output0_vec, output0_dptr);

  // check that we were able to copy using set/get
  for(camp::idx_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(output0_vec[i], (element_t)11);
  }



  //
  // test scalar explicit broadcast
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    reg_t cc = (element_t) 0;
    cc.broadcast(13.0);

    // extract from x using get
    for(camp::idx_t i = 0;i < num_elem; ++ i){
      output0_dptr[i] = cc.get(i);
    }

  });
  tensor_copy_to_host<policy_t>(output0_vec, output0_dptr);

  // check that we were able to copy using set/get
  for(camp::idx_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(output0_vec[i], (element_t)13);
  }


  //
  // Cleanup
  //
  tensor_free<policy_t>(input0_dptr);
  tensor_free<policy_t>(output0_dptr);
}



TYPED_TEST_P(TestTensorRegister, GetSet)
{
  GetSetImpl<TypeParam>();
}


#endif
