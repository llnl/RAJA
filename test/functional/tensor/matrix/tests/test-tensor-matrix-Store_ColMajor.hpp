//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_MATRIX_Store_ColMajor_HPP__
#define __TEST_TENSOR_MATRIX_Store_ColMajor_HPP__

#include<RAJA/RAJA.hpp>

template <typename MATRIX_TYPE>
void Store_ColMajorImpl()
{

  using matrix_t = MATRIX_TYPE;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;


  //
  // Allocate Column-Major Data
  //

  // alloc data1 - matrix data will be generated on device, stored into data1

  std::vector<element_t> data1_vec(4*matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data1_h(data1_vec.data(), 2*matrix_t::s_num_columns, 2*matrix_t::s_num_rows);

  element_t *data1_ptr = tensor_malloc<policy_t>(data1_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data1_d(data1_ptr, 2*matrix_t::s_num_rows, 2*matrix_t::s_num_columns);


  // alloc data2 - reference data to compare with data1 on host

  std::vector<element_t> data2_vec(matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data2_h(data2_vec.data(), matrix_t::s_num_columns, matrix_t::s_num_rows);


  //
  // Fill reference data
  //
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      data2_h(j,i) = 2*i*matrix_t::s_num_columns+j;
    }
  }

  //
  // Clear data1
  //
  for(camp::idx_t i = 0;i < 2*matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < 2*matrix_t::s_num_columns; ++ j){
      data1_h(j,i) = element_t(-2);
    }
  }
  tensor_copy_to_device<policy_t>(data1_ptr, data1_vec);


  //
  // Do Operation: Full store
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    // fill out matrix
    matrix_t m(-1.0);

    for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
      for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
        m.set(2*i*matrix_t::s_num_columns+j, i, j);
      }
    }

    // Store matrix to memory
    if(matrix_t::layout_type::is_column_major()){
      m.store_packed(data1_ptr, 1, 2*matrix_t::s_num_rows);
    }
    else{
      m.store_strided(data1_ptr, 1, 2*matrix_t::s_num_rows);
    }
  });

  tensor_copy_to_host<policy_t>(data1_vec, data1_ptr);


  //
  // Check results
  //
  for(camp::idx_t i = 0;i < 2*matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < 2*matrix_t::s_num_columns; ++ j){
      if(i < matrix_t::s_num_rows && j < matrix_t::s_num_columns){
//        printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1_h(i,j), data2_h(i,j));
        ASSERT_SCALAR_EQ(data1_h(j,i), data2_h(j,i));
      }
      else{
//        printf("%d,%d:  %lf, -2\n", (int)i, (int)j, data1_h(i,j));
        ASSERT_SCALAR_EQ(data1_h(j,i), element_t(-2));
      }
    }
  }



  //
  // Loop over all possible sub-matrix sizes using the load_*_nm routines
  //
  for(camp::idx_t n_size = 0;n_size <= matrix_t::s_num_rows; ++ n_size){
    for(camp::idx_t m_size = 0;m_size <= matrix_t::s_num_columns; ++ m_size){

      //
      // Clear data1
      //
      for(camp::idx_t i = 0;i < 2*matrix_t::s_num_rows; ++ i){
        for(camp::idx_t j = 0;j < 2*matrix_t::s_num_columns; ++ j){
          data1_h(j,i) = element_t(-2);
        }
      }
      tensor_copy_to_device<policy_t>(data1_ptr, data1_vec);


      //
      // Do Operation: Partial Store
      //
      tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
        // fill out matrix
        matrix_t m(-1.0);

        for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
          for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
            m.set(2*i*matrix_t::s_num_columns+j, i, j);
          }
        }

        // Store matrix to memory
        if(matrix_t::layout_type::is_column_major()){
          m.store_packed_nm(data1_ptr, 1, 2*matrix_t::s_num_rows, n_size, m_size);
        }
        else{
          m.store_strided_nm(data1_ptr, 1, 2*matrix_t::s_num_rows, n_size, m_size);
        }

      });


      tensor_copy_to_host<policy_t>(data1_vec, data1_ptr);


      //
      // Check results
      //
      for(camp::idx_t i = 0;i < 2*matrix_t::s_num_rows; ++ i){
        for(camp::idx_t j = 0;j < 2*matrix_t::s_num_columns; ++ j){
          if(i < n_size && j < m_size){
//            printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1_h(i,j), data2_h(i,j));
            ASSERT_SCALAR_EQ(data1_h(j,i), data2_h(j,i));
          }
          else{
//            printf("%d,%d:  %lf, -2\n", (int)i, (int)j, data1_h(i,j));
            ASSERT_SCALAR_EQ(data1_h(j,i), element_t(-2));
          }
        }
      }


    }
  }


  //
  // Free data
  //
  tensor_free<policy_t>(data1_ptr);
}



TYPED_TEST_P(TestTensorMatrix, Store_ColMajor)
{
  Store_ColMajorImpl<TypeParam>();
}


#endif
