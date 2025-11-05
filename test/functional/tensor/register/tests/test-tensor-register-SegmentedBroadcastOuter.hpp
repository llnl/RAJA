//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_REGISTER_SegmentedBroadcastOuter_HPP__
#define __TEST_TENSOR_REGISTER_SegmentedBroadcastOuter_HPP__

#include<RAJA/RAJA.hpp>

template <typename REGISTER_TYPE>
void SegmentedBroadcastOuterImpl()
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
//  printf("input: ");
  for(camp::idx_t i = 0;i < num_elem; ++ i){
    input0_hptr[i] = (element_t)(i+1+NO_OPT_RAND);
//    printf("%lf ", (double)input0_hptr[i]);
  }
//  printf("\n");
  tensor_copy_to_device<policy_t>(input0_dptr, input0_vec);



  // run segmented dot products for all segments allowed by the vector
  for(camp::idx_t segbits = 0;(1<<segbits) <= num_elem;++ segbits){

    camp::idx_t num_segments = (1<<segbits);

    for(camp::idx_t input_segment = 0;input_segment < num_segments;++ input_segment){

      // Execute segmented broadcast
      tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

        reg_t x;
        x.load_packed(input0_dptr);

        reg_t y = x.segmented_broadcast_outer(segbits, input_segment);

        y.store_packed(output0_dptr);

      });

      // Move result to host
      tensor_copy_to_host<policy_t>(output0_vec, output0_dptr);


      // Check result

      // Compute expected values
//      printf("explode: segbits=%d, input_segment=%d\n", segbits, input_segment);
//      printf("  expected:  ");

      element_t expected[num_elem];
      for(camp::idx_t i = 0;i < num_elem; ++ i){
        camp::idx_t seg = i>>segbits;

        camp::idx_t off = (num_elem>>segbits)*input_segment + seg;

        expected[i] = input0_hptr[off];
//        printf("%lf ", (double)expected[i]);
      }
//      printf("\n");


      for(camp::idx_t i = 0;i < num_elem; ++ i){
        ASSERT_SCALAR_EQ(expected[i], output0_vec[i]);
      }

    } // segment

  } // segbits


  // Cleanup
  tensor_free<policy_t>(input0_dptr);
  tensor_free<policy_t>(output0_dptr);
}



TYPED_TEST_P(TestTensorRegister, SegmentedBroadcastOuter)
{
  SegmentedBroadcastOuterImpl<TypeParam>();
}


#endif
