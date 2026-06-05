//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_MATRIX_Transpose_HPP__
#define __TEST_TENSOR_MATRIX_Transpose_HPP__

#include<RAJA/RAJA.hpp>

template <typename MATRIX_TYPE>
void TransposeImpl()
{

  using matrix_t = MATRIX_TYPE;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;

  using transpose_t = typename matrix_t::transpose_type;


  static constexpr camp::idx_t N = matrix_t::s_num_rows;
  static constexpr camp::idx_t M = matrix_t::s_num_columns;

//  bool is_row_major = matrix_t::layout_type::is_row_major();

  //
  // Allocate Row-Major Data
  //

  // alloc input0

  std::vector<element_t> input0_vec(N*M);
  RAJA::View<element_t, RAJA::Layout<2>> input0_h(input0_vec.data(), N, M);

  element_t *input0_ptr = tensor_malloc<policy_t>(input0_vec);
  RAJA::View<element_t, RAJA::Layout<2>> input0_d(input0_ptr,  N, M);



  // alloc output0

  std::vector<element_t> output0_vec(N*M);
  RAJA::View<element_t, RAJA::Layout<2>> output0_h(output0_vec.data(),  M, N);

  element_t *output0_ptr = tensor_malloc<policy_t>(output0_vec);
  RAJA::View<element_t, RAJA::Layout<2>> output0_d(output0_ptr,  M, N);



  // Fill input0 and output0
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < M; ++ j){
      input0_h(i,j) = i*matrix_t::s_num_columns+j;
    }
  }

  tensor_copy_to_device<policy_t>(input0_ptr, input0_vec);




  //
  // Do Operation: transpose
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    // load original matrix
    matrix_t A;
    A.load_strided(input0_ptr, M, 1);

    // transpose matrix
    transpose_t B = A.transpose();

    // store transposed matrix
    B.store_strided(output0_ptr, N, 1);

  });

  tensor_copy_to_host<policy_t>(output0_vec, output0_ptr);


  printf("gtest result:\n");
  for(camp::idx_t i = 0;i < M; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      printf("%3d ", (int)output0_h(i,j));
    }
    printf("\n");
  }



  //
  // Check results
  //
  for(camp::idx_t i = 0;i < M; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      ASSERT_SCALAR_EQ(output0_h(i,j), input0_h(j,i));
    }
  }



  //
  // Free data
  //
  tensor_free<policy_t>(input0_ptr);
  tensor_free<policy_t>(output0_ptr);

}



TYPED_TEST_P(TestTensorMatrix, Transpose)
{
  TransposeImpl<TypeParam>();
}


#endif
