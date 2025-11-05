//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_REGISTER_Scatter_HPP__
#define __TEST_TENSOR_REGISTER_Scatter_HPP__

#include<RAJA/RAJA.hpp>

template <typename REGISTER_TYPE>
void ScatterImpl()
{
  using reg_t = REGISTER_TYPE;
  using element_t = typename reg_t::element_type;
  using policy_t = typename reg_t::register_policy;

  static constexpr camp::idx_t num_elem = reg_t::s_num_elem;

  // get the integer indexing types
  using int_register_t = typename reg_t::int_vector_type;
  using index_t = typename int_register_t::element_type;

  // Allocate

  // Data to be read
  std::vector<element_t> input0_vec(num_elem);
  element_t *input0_hptr = input0_vec.data();
  element_t *input0_dptr = tensor_malloc<policy_t, element_t>(num_elem);

  // Indexing into output0
  std::vector<index_t> input1_vec(num_elem);
  index_t *input1_hptr = input1_vec.data();
  index_t *input1_dptr = tensor_malloc<policy_t, index_t>(num_elem);

  // Scattered output (10x larger than output)
  std::vector<element_t> output0_vec(10*num_elem);
  element_t *output0_dptr = tensor_malloc<policy_t, element_t>(10*num_elem);

  // precomputed expected output
  std::vector<element_t> expected(10*num_elem);

  // Initialize input data
  for(camp::idx_t i = 0;i < num_elem; ++ i){
   input0_hptr[i] = (element_t)(i+1+NO_OPT_RAND);
   input1_hptr[i] = (index_t)(3*i+1+NO_OPT_RAND);
  }

  tensor_copy_to_device<policy_t>(input0_dptr, input0_vec);
  tensor_copy_to_device<policy_t>(input1_dptr, input1_vec);


  // Initialize output
  for(camp::idx_t i = 0;i < num_elem; ++ i){
   output0_vec[i] = (element_t)0;
  }
  tensor_copy_to_device<policy_t>(output0_dptr, output0_vec);


  //
  //  Check full-length operations
  //

  // operator z[b[i]] = a[i]
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    int_register_t idx;
    idx.load_packed(input1_dptr);

    reg_t a;
    a.load_packed(input0_dptr);

    a.scatter(output0_dptr, idx);
  });

  tensor_copy_to_host<policy_t>(output0_vec, output0_dptr);

  // compute expected value
  for(camp::idx_t lane = 0;lane < 10*num_elem;++ lane){
    expected[lane] = 0;
  }
  for(camp::idx_t lane = 0;lane < num_elem;++ lane){
    expected[input1_vec[lane]] = input0_vec[lane];
  }

  // check result
  for(camp::idx_t lane = 0;lane < num_elem;++ lane){
    ASSERT_SCALAR_EQ(expected[lane], output0_vec[lane]);
  }


  //
  // Check partial length operations
  //

  for(camp::idx_t N = 0;N <= num_elem;++ N){

    // Initialize output
    for(camp::idx_t i = 0;i < num_elem; ++ i){
     output0_vec[i] = (element_t)0;
    }
    tensor_copy_to_device<policy_t>(output0_dptr, output0_vec);



    // operator z[i] = a[b[i]]
    tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

      int_register_t idx;
      idx.load_packed(input1_dptr);

      reg_t a;
      a.load_packed(input0_dptr);

      a.scatter_n(output0_dptr, idx, N);
    });

    tensor_copy_to_host<policy_t>(output0_vec, output0_dptr);


    // compute expected value
    for(camp::idx_t lane = 0;lane < 10*num_elem;++ lane){
      expected[lane] = 0;
    }
    for(camp::idx_t lane = 0;lane < N;++ lane){
      expected[input1_vec[lane]] = input0_vec[lane];
    }

    // check result
    for(camp::idx_t lane = 0;lane < num_elem;++ lane){
      ASSERT_SCALAR_EQ(expected[lane], output0_vec[lane]);
    }

  }


  // Cleanup
  tensor_free<policy_t>(input0_dptr);
  tensor_free<policy_t>(input1_dptr);
  tensor_free<policy_t>(output0_dptr);
}



TYPED_TEST_P(TestTensorRegister, Scatter)
{
  ScatterImpl<TypeParam>();
}


#endif
