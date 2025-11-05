//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_REGISTER_Gather_HPP__
#define __TEST_TENSOR_REGISTER_Gather_HPP__

#include<RAJA/RAJA.hpp>

template <typename REGISTER_TYPE>
void GatherImpl()
{
  using reg_t = REGISTER_TYPE;
  using element_t = typename reg_t::element_type;
  using policy_t = typename reg_t::register_policy;

  static constexpr camp::idx_t num_elem = reg_t::s_num_elem;

  // get the integer indexing types
  using int_register_t = typename reg_t::int_vector_type;
  using index_t = typename int_register_t::element_type;

  // Allocate

  // Data to be read (10x larger than output)
  std::vector<element_t> input0_vec(10*num_elem);
  element_t *input0_hptr = input0_vec.data();
  element_t *input0_dptr = tensor_malloc<policy_t, element_t>(10*num_elem);

  // Indexing into input0
  std::vector<index_t> input1_vec(num_elem);
  index_t *input1_hptr = input1_vec.data();
  index_t *input1_dptr = tensor_malloc<policy_t, index_t>(num_elem);

  std::vector<element_t> output0_vec(num_elem);
  element_t *output0_dptr = tensor_malloc<policy_t, element_t>(num_elem);


  // Initialize input data
  for(camp::idx_t i = 0;i < 10*num_elem; ++ i){
   input0_hptr[i] = (element_t)(i+1+NO_OPT_RAND);
  }
  for(camp::idx_t i = 0;i < num_elem; ++ i){
   input1_hptr[i] = (index_t)(3*i+1+NO_OPT_RAND);
  }

  tensor_copy_to_device<policy_t>(input0_dptr, input0_vec);
  tensor_copy_to_device<policy_t>(input1_dptr, input1_vec);


  //
  //  Check full-length operations
  //

  // operator z[i] = a[b[i]]
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    // get offsets
    int_register_t idx;
    idx.load_packed(input1_dptr);

    // gather elements from a given offsets in idx
    reg_t a;
    a.gather(input0_dptr, idx);

    // write out gathered elements in packed order
    a.store_packed(output0_dptr);
  });

  tensor_copy_to_host<policy_t>(output0_vec, output0_dptr);

  for(camp::idx_t lane = 0;lane < num_elem;++ lane){
    ASSERT_SCALAR_EQ(input0_vec[input1_vec[lane]], output0_vec[lane]);
  }


  //
  // Check partial length operations
  //

  for(camp::idx_t N = 0;N <= num_elem;++ N){

    // operator z[i] = a[b[i]]
    tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

      // get offsets
      int_register_t idx;
      idx.load_packed_n(input1_dptr, N);

      // gather elements from a given offsets in idx
      reg_t a;
      a.gather_n(input0_dptr, idx, N);

      // write out gathered elements in packed order
      // we're writing out entire length to check the zeroing
      a.store_packed(output0_dptr);
    });

    tensor_copy_to_host<policy_t>(output0_vec, output0_dptr);


    for(camp::idx_t lane = 0;lane < num_elem;++ lane){
      if(lane < N){
        ASSERT_SCALAR_EQ(input0_vec[input1_vec[lane]], output0_vec[lane]);
      }
      else{
        ASSERT_SCALAR_EQ((element_t)0, output0_vec[lane]);
      }
    }

  }


  // Cleanup
  tensor_free<policy_t>(input0_dptr);
  tensor_free<policy_t>(input1_dptr);
  tensor_free<policy_t>(output0_dptr);
}



TYPED_TEST_P(TestTensorRegister, Gather)
{
  GatherImpl<TypeParam>();
}


#endif
