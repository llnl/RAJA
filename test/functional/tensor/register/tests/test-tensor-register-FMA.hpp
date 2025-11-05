//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_REGISTER_FMA_HPP__
#define __TEST_TENSOR_REGISTER_FMA_HPP__

#include<RAJA/RAJA.hpp>

template <typename REGISTER_TYPE>
void FMAImpl()
{
  using reg_t = REGISTER_TYPE;
  using element_t = typename reg_t::element_type;
  using policy_t = typename reg_t::register_policy;

  static constexpr camp::idx_t num_elem = reg_t::s_num_elem;

  // Allocate

  std::vector<element_t> input0_vec(num_elem);
  element_t *input0_hptr = input0_vec.data();
  element_t *input0_dptr = tensor_malloc<policy_t, element_t>(num_elem);

  std::vector<element_t> input1_vec(num_elem);
  element_t *input1_hptr = input1_vec.data();
  element_t *input1_dptr = tensor_malloc<policy_t, element_t>(num_elem);

  std::vector<element_t> input2_vec(num_elem);
  element_t *input2_hptr = input2_vec.data();
  element_t *input2_dptr = tensor_malloc<policy_t, element_t>(num_elem);

  std::vector<element_t> output0_vec(num_elem);
  element_t *output0_dptr = tensor_malloc<policy_t, element_t>(num_elem);


  // Initialize input data
  for(camp::idx_t i = 0;i < num_elem; ++ i){
   input0_hptr[i] = (element_t)(i+1+NO_OPT_RAND);
   input1_hptr[i] = (element_t)(i*i+1+NO_OPT_RAND);
   input2_hptr[i] = (element_t)(i+i+1+NO_OPT_RAND);
  }

  tensor_copy_to_device<policy_t>(input0_dptr, input0_vec);
  tensor_copy_to_device<policy_t>(input1_dptr, input1_vec);
  tensor_copy_to_device<policy_t>(input2_dptr, input2_vec);


  //
  //  Check full-length operations
  //

  // operator z = a*b+c
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    reg_t a;
    a.load_packed(input0_dptr);

    reg_t b;
    b.load_packed(input1_dptr);

    reg_t c;
    c.load_packed(input2_dptr);

    reg_t z = a.multiply_add(b,c);

    z.store_packed(output0_dptr);
  });

  tensor_copy_to_host<policy_t>(output0_vec, output0_dptr);

  for(camp::idx_t lane = 0;lane < num_elem;++ lane){
    ASSERT_SCALAR_EQ(input0_vec[lane] * input1_vec[lane] + input2_vec[lane], output0_vec[lane]);
  }




  // Cleanup
  tensor_free<policy_t>(input0_dptr);
  tensor_free<policy_t>(input1_dptr);
  tensor_free<policy_t>(input2_dptr);
  tensor_free<policy_t>(output0_dptr);
}



TYPED_TEST_P(TestTensorRegister, FMA)
{
  FMAImpl<TypeParam>();
}


#endif
