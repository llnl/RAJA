//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_REGISTER_Load_RowMajor_HPP__
#define __TEST_TENSOR_REGISTER_Load_RowMajor_HPP__

#include<RAJA/RAJA.hpp>

template <typename MATRIX_TYPE>
void Load_RowMajorImpl()
{

  using matrix_t = MATRIX_TYPE;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;


  //
  // Allocate Row-Major Data
  //

  // alloc data1

  std::vector<element_t> data1_vec(4*matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data1_h(data1_vec.data(), 2*matrix_t::s_num_rows, 2*matrix_t::s_num_columns);

  element_t *data1_ptr = tensor_malloc<policy_t>(data1_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data1_d(data1_ptr, 2*matrix_t::s_num_rows, 2*matrix_t::s_num_columns);


  // alloc data2

  std::vector<element_t> data2_vec(matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data2_h(data2_vec.data(), matrix_t::s_num_rows, matrix_t::s_num_columns);

  element_t *data2_ptr = tensor_malloc<policy_t>(data2_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data2_d(data2_ptr, matrix_t::s_num_rows, matrix_t::s_num_columns);



  // Fill data
  for(camp::idx_t i = 0;i < 2*matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < 2*matrix_t::s_num_columns; ++ j){
      data1_h(i,j) = 2*i*matrix_t::s_num_columns+j;
    }
  }

  tensor_copy_to_device<policy_t>(data1_ptr, data1_vec);


  //
  // Do Operation: Full load
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
    matrix_t m;
    if(matrix_t::layout_type::is_row_major()){
      m.load_packed(data1_ptr, 2*matrix_t::s_num_columns, 1);
    }
    else{
      m.load_strided(data1_ptr, 2*matrix_t::s_num_columns, 1);
    }

    // write out to a second view so we can check it on the host
    // on GPU's we'll write way too much, but it should stil be correct
    for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
      for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
        data2_d(i,j) = m.get(i,j);
      }
    }

  });

  tensor_copy_to_host<policy_t>(data2_vec, data2_ptr);


  //
  // Check results
  //
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      ASSERT_SCALAR_EQ(data1_h(i,j), data2_h(i,j));
//      printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1(i,j), data2(i,j));
    }
  }



  //
  // Loop over all possible sub-matrix sizes using the load_*_nm routines
  //
  for(camp::idx_t n_size = 0;n_size <= matrix_t::s_num_rows; ++ n_size){
    for(camp::idx_t m_size = 0;m_size <= matrix_t::s_num_columns; ++ m_size){
//      printf("Running %d x %d\n", (int)n_size, (int)m_size);
      //
      // Clear data2
      //
      for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
        for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
          data2_h(i,j) = -1;
        }
      }
      tensor_copy_to_device<policy_t>(data2_ptr, data2_vec);


      //
      // Do Operation: Partial load
      //
      tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
        matrix_t m;
        if(matrix_t::layout_type::is_row_major()){
          m.load_packed_nm(data1_ptr, 2*matrix_t::s_num_columns, 1, n_size, m_size);
        }
        else{
          m.load_strided_nm(data1_ptr, 2*matrix_t::s_num_columns, 1, n_size, m_size);
        }

        // write out to a second view so we can check it on the host
        // on GPU's we'll write way too much, but it should stil be correct
        for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
          for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
            data2_d(i,j) = m.get(i,j);
          }
        }

      });

      tensor_copy_to_host<policy_t>(data2_vec, data2_ptr);


      //
      // Check results
      //
      for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
        for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
//          printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1(i,j), data2(i,j));
          if(i < n_size && j < m_size){
            ASSERT_SCALAR_EQ(data1_h(i,j), data2_h(i,j));
          }
          else{
            ASSERT_SCALAR_EQ(element_t(0), data2_h(i,j));
          }
        }
      }


    }
  }


  //
  // Free data
  //
  tensor_free<policy_t>(data1_ptr);
  tensor_free<policy_t>(data2_ptr);
}



TYPED_TEST_P(TestTensorMatrix, Load_RowMajor)
{
  Load_RowMajorImpl<TypeParam>();
}


#endif
