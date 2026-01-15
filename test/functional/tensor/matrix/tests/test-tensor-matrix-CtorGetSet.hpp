//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_REGISTER_CtorGetSet_HPP__
#define __TEST_TENSOR_REGISTER_CtorGetSet_HPP__

#include<RAJA/RAJA.hpp>

template <typename MATRIX_TYPE>
void CtorGetSetImpl()
{

  using matrix_t = MATRIX_TYPE;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;


  //
  // Allocate Data
  //
  std::vector<element_t> data1_vec(matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data1_h(data1_vec.data(), matrix_t::s_num_rows, matrix_t::s_num_columns);

  element_t *data1_ptr = tensor_malloc<policy_t>(data1_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data1_d(data1_ptr, matrix_t::s_num_rows, matrix_t::s_num_columns);


  std::vector<element_t> data2_vec(matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data2_h(data2_vec.data(), matrix_t::s_num_rows, matrix_t::s_num_columns);

  element_t *data2_ptr = tensor_malloc<policy_t>(data2_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data2_d(data2_ptr, matrix_t::s_num_rows, matrix_t::s_num_columns);



  //
  // Do Operation: broadcast-ctor and copy-ctor
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    // create a matrix that contains all 3's
    matrix_t m1(element_t(3));

    // copy to another matrix
    matrix_t m2(m1);

    // write out both matrices
    for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
      for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
        data1_d(i,j) = m1.get(i,j);
        data2_d(i,j) = m2.get(i,j);
      }
    }

  });

  // copy data back to host
  tensor_copy_to_host<policy_t>(data1_vec, data1_ptr);
  tensor_copy_to_host<policy_t>(data2_vec, data2_ptr);


  //
  // Check results
  //
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      ASSERT_SCALAR_EQ(3, data1_h(i,j));
      ASSERT_SCALAR_EQ(3, data2_h(i,j));
//      printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1(i,j), data2(i,j));
    }
  }


  //
  // Free data
  //
  tensor_free<policy_t>(data1_ptr);
  tensor_free<policy_t>(data2_ptr);

}



TYPED_TEST_P(TestTensorMatrix, CtorGetSet)
{
  CtorGetSetImpl<TypeParam>();
}


#endif
