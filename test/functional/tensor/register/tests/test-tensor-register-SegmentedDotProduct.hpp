//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_REGISTER_SegmentedDotProduct_HPP__
#define __TEST_TENSOR_REGISTER_SegmentedDotProduct_HPP__

#include<RAJA/RAJA.hpp>

template <typename REGISTER_TYPE>
void SegmentedDotProductImpl()
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

  std::vector<element_t> output0_vec(num_elem);
  element_t *output0_dptr = tensor_malloc<policy_t, element_t>(num_elem);


  // Initialize input data
  for(camp::idx_t i = 0;i < num_elem; ++ i){
   input0_hptr[i] = (element_t)(i+1+NO_OPT_RAND);
   input1_hptr[i] = (element_t)(i*i+1+NO_OPT_RAND);
  }

  tensor_copy_to_device<policy_t>(input0_dptr, input0_vec);
  tensor_copy_to_device<policy_t>(input1_dptr, input1_vec);



  // run segmented dot products for all segments allowed by the vector
  for(camp::idx_t segbits = 0;(1<<segbits) <= num_elem;++ segbits){

    camp::idx_t num_output_segments = 1<<segbits;

    for(camp::idx_t output_segment = 0;output_segment < num_output_segments;++output_segment){


      tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

        reg_t x;
        x.load_packed(input0_dptr);

        reg_t y;
        y.load_packed(input1_dptr);

        reg_t dp = x.segmented_dot(segbits, output_segment, y);
        dp.store_packed(output0_dptr);

      });


      // Move result to host
      tensor_copy_to_host<policy_t>(output0_vec, output0_dptr);

      // Compute expected values
      std::vector<element_t> expected(num_elem);

      camp::idx_t offset = output_segment * num_elem/(1<<segbits);

      for(camp::idx_t i = 0;i < num_elem; ++ i){
        expected[i] = 0;
      }
      for(camp::idx_t i = 0;i < num_elem; ++ i){
        expected[(i>>segbits) + offset] += input0_vec[i]*input1_vec[i];
      }

      for(camp::idx_t i = 0;i < num_elem; ++ i){
        ASSERT_SCALAR_EQ(expected[i], output0_vec[i]);
      }

    } // output_segment

  } // segbits



  // Cleanup
  tensor_free<policy_t>(input0_dptr);
  tensor_free<policy_t>(input1_dptr);
  tensor_free<policy_t>(output0_dptr);
}



TYPED_TEST_P(TestTensorRegister, SegmentedDotProduct)
{
  SegmentedDotProductImpl<TypeParam>();
}


#endif
