//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_MATRIX_ET_Transpose_HPP__
#define __TEST_TENSOR_MATRIX_ET_Transpose_HPP__

#include<RAJA/RAJA.hpp>

template <typename MATRIX_TYPE>
void ET_TransposeImpl()
{

  using matrix_t = MATRIX_TYPE;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;

  using transpose_t = typename matrix_t::transpose_type;

//  static constexpr camp::idx_t N = RAJA::max<camp::idx_t>(matrix_t::s_num_rows, matrix_t::s_num_columns)*2;
//  static constexpr camp::idx_t M = RAJA::max<camp::idx_t>(matrix_t::s_num_rows, matrix_t::s_num_columns)*2;

  static constexpr camp::idx_t N = matrix_t::s_num_rows;
  static constexpr camp::idx_t M = matrix_t::s_num_columns;

  //
  // Allocate Row-Major Data
  //

  // alloc input0

  std::vector<element_t> input0_vec(N*M);
  RAJA::View<element_t, RAJA::Layout<2>> input0_h(input0_vec.data(), N, M);

  element_t *input0_ptr = tensor_malloc<policy_t>(input0_vec);
  RAJA::View<element_t, RAJA::Layout<2>> input0_d(input0_ptr,  N, M);


  // alloc input1 with StaticLayout

  std::vector<element_t> input1_vec(N*M);
  RAJA::View<element_t, RAJA::StaticLayout<RAJA::PERM_IJ,N,M>> input1_h(input1_vec.data());

  element_t *input1_ptr = tensor_malloc<policy_t>(input1_vec);
  RAJA::View<element_t, RAJA::StaticLayout<RAJA::PERM_IJ,N,M>> input1_d(input1_ptr);



  // alloc output0

  std::vector<element_t> output0_vec(N*M);
  RAJA::View<element_t, RAJA::Layout<2>> output0_h(output0_vec.data(),  M, N);

  element_t *output0_ptr = tensor_malloc<policy_t>(output0_vec);
  RAJA::View<element_t, RAJA::Layout<2>> output0_d(output0_ptr,  M, N);


  // alloc output1

  std::vector<element_t> output1_vec(N*M);
  RAJA::View<element_t, RAJA::Layout<2>> output1_h(output1_vec.data(),  M, N);

  element_t *output1_ptr = tensor_malloc<policy_t>(output1_vec);
  RAJA::View<element_t, RAJA::Layout<2>> output1_d(output1_ptr,  M, N);


  // alloc output2

  std::vector<element_t> output2_vec(N*M);
  RAJA::View<element_t, RAJA::Layout<2>> output2_h(output2_vec.data(),  M, N);

  element_t *output2_ptr = tensor_malloc<policy_t>(output2_vec);
  RAJA::View<element_t, RAJA::Layout<2>> output2_d(output2_ptr,  M, N);


  // alloc output3

  std::vector<element_t> output3_vec(N*M);
  RAJA::View<element_t, RAJA::Layout<2>> output3_h(output3_vec.data(),  M, N);

  element_t *output3_ptr = tensor_malloc<policy_t>(output3_vec);
  RAJA::View<element_t, RAJA::Layout<2>> output3_d(output3_ptr,  M, N);


  // alloc output4

  std::vector<element_t> output4_vec(N*M);
  RAJA::View<element_t, RAJA::Layout<2>> output4_h(output4_vec.data(),  M, N);

  element_t *output4_ptr = tensor_malloc<policy_t>(output4_vec);
  RAJA::View<element_t, RAJA::Layout<2>> output4_d(output4_ptr,  M, N);



  // Fill input0 and input1
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < M; ++ j){
      input0_h(i,j) = i*matrix_t::s_num_columns+j;
      input1_h(i,j) = i*matrix_t::s_num_columns+j;
    }
  }

  tensor_copy_to_device<policy_t>(input0_ptr, input0_vec);
  tensor_copy_to_device<policy_t>(input1_ptr, input1_vec);


  //
  // Do Operation: transpose
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    auto rows = RAJA::expt::RowIndex<int, matrix_t>::all();
    auto cols = RAJA::expt::ColIndex<int, matrix_t>::all();

    auto rows_tr = RAJA::expt::RowIndex<int, transpose_t>::all();
    auto cols_tr = RAJA::expt::ColIndex<int, transpose_t>::all();

    auto SArows = RAJA::expt::RowIndex<int, matrix_t>::static_all();
    auto SAcols = RAJA::expt::ColIndex<int, matrix_t>::static_all();

    auto SRrows = RAJA::expt::RowIndex<int, matrix_t>::template static_range<0,N>();
    auto SRcols = RAJA::expt::ColIndex<int, matrix_t>::template static_range<0,M>();

    output0_d(rows_tr, cols_tr) = input0_d(rows, cols).transpose();

    output1_d(rows_tr, cols_tr) = input1_d(SArows, SRcols).transpose();  // mixed static_all and static_range
    output2_d(rows_tr, cols_tr) = input1_d(SArows, SAcols).transpose();  // static_all
    output3_d(rows_tr, cols_tr) = input1_d(SRrows, SRcols).transpose();  // static_range
    output4_d(rows_tr, cols_tr) = input1_d(rows, SRcols).transpose();    // mixed static_range and non-static

  });

  tensor_copy_to_host<policy_t>(output0_vec, output0_ptr);
  tensor_copy_to_host<policy_t>(output1_vec, output1_ptr);
  tensor_copy_to_host<policy_t>(output2_vec, output2_ptr);
  tensor_copy_to_host<policy_t>(output3_vec, output3_ptr);
  tensor_copy_to_host<policy_t>(output4_vec, output4_ptr);



//  for(camp::idx_t i = 0;i < M; ++ i){
//    for(camp::idx_t j = 0;j < N; ++ j){
//      printf("%3d ", (int)output0_h(i,j));
//    }
//    printf("\n");
//  }



  //
  // Check results
  //
  for(camp::idx_t i = 0;i < M; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      ASSERT_SCALAR_EQ(output0_h(i,j), input0_h(j,i));
      ASSERT_SCALAR_EQ(output1_h(i,j), input1_h(j,i));
      ASSERT_SCALAR_EQ(output2_h(i,j), input1_h(j,i));
      ASSERT_SCALAR_EQ(output3_h(i,j), input1_h(j,i));
      ASSERT_SCALAR_EQ(output4_h(i,j), input1_h(j,i));
    }
  }



  //
  // Free data
  //
  tensor_free<policy_t>(input0_ptr);
  tensor_free<policy_t>(input1_ptr);
  tensor_free<policy_t>(output0_ptr);
  tensor_free<policy_t>(output1_ptr);
  tensor_free<policy_t>(output2_ptr);
  tensor_free<policy_t>(output3_ptr);
  tensor_free<policy_t>(output4_ptr);

}



TYPED_TEST_P(TestTensorMatrix, ET_Transpose)
{
  ET_TransposeImpl<TypeParam>();
}


#endif
