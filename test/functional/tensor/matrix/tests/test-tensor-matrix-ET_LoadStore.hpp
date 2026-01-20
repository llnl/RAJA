//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_REGISTER_ET_LoadStore_HPP__
#define __TEST_TENSOR_REGISTER_ET_LoadStore_HPP__

#include<RAJA/RAJA.hpp>

template <typename MATRIX_TYPE>
void ET_LoadStoreImpl()
{

  using matrix_t = MATRIX_TYPE;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;


  //
  // Allocate Row-Major Data
  //

  // alloc data1
  std::vector<element_t> data1_vec(matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data1_h(data1_vec.data(), matrix_t::s_num_rows, matrix_t::s_num_columns);

  element_t *data1_ptr = tensor_malloc<policy_t>(data1_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data1_d(data1_ptr, matrix_t::s_num_rows, matrix_t::s_num_columns);


  // alloc data2
  std::vector<element_t> data2_vec(matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data2_h(data2_vec.data(), matrix_t::s_num_columns, matrix_t::s_num_rows);

  element_t *data2_ptr = tensor_malloc<policy_t>(data2_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data2_d(data2_ptr, matrix_t::s_num_columns, matrix_t::s_num_rows);


  // alloc data3 with StaticLayout
  std::vector<element_t> data3_vec(matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::StaticLayout<RAJA::PERM_IJ,matrix_t::s_num_rows,matrix_t::s_num_columns>> data3_h(data3_vec.data());

  element_t *data3_ptr = tensor_malloc<policy_t>(data3_vec);
  RAJA::View<element_t, RAJA::StaticLayout<RAJA::PERM_IJ,matrix_t::s_num_rows,matrix_t::s_num_columns>> data3_d(data3_ptr);


  // alloc data4
  std::vector<element_t> data4_vec(matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data4_h(data4_vec.data(), matrix_t::s_num_columns, matrix_t::s_num_rows);

  element_t *data4_ptr = tensor_malloc<policy_t>(data4_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data4_d(data4_ptr, matrix_t::s_num_columns, matrix_t::s_num_rows);


  // alloc data5
  std::vector<element_t> data5_vec(matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data5_h(data5_vec.data(), matrix_t::s_num_columns, matrix_t::s_num_rows);

  element_t *data5_ptr = tensor_malloc<policy_t>(data5_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data5_d(data5_ptr, matrix_t::s_num_columns, matrix_t::s_num_rows);


  // alloc data6
  std::vector<element_t> data6_vec(matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data6_h(data6_vec.data(), matrix_t::s_num_columns, matrix_t::s_num_rows);

  element_t *data6_ptr = tensor_malloc<policy_t>(data6_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data6_d(data6_ptr, matrix_t::s_num_columns, matrix_t::s_num_rows);


  // alloc data7
  std::vector<element_t> data7_vec(matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data7_h(data7_vec.data(), matrix_t::s_num_columns, matrix_t::s_num_rows);

  element_t *data7_ptr = tensor_malloc<policy_t>(data7_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data7_d(data7_ptr, matrix_t::s_num_columns, matrix_t::s_num_rows);



  // Fill data
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      data1_h(i,j) = i*matrix_t::s_num_columns+j;
      data3_h(i,j) = i*matrix_t::s_num_columns+j;
    }
  }

  tensor_copy_to_device<policy_t>(data1_ptr, data1_vec);
  tensor_copy_to_device<policy_t>(data3_ptr, data3_vec);


  //
  // Do Operation: Load/Store full matrix from one view to another
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    auto rows = RAJA::expt::RowIndex<int, matrix_t>::all();
    auto cols = RAJA::expt::ColIndex<int, matrix_t>::all();

    auto SArows = RAJA::expt::RowIndex<int, matrix_t>::static_all();
    auto SAcols = RAJA::expt::ColIndex<int, matrix_t>::static_all();

    auto SRrows = RAJA::expt::RowIndex<int, matrix_t>::template static_range<0,matrix_t::s_num_rows>();
    auto SRcols = RAJA::expt::ColIndex<int, matrix_t>::template static_range<0,matrix_t::s_num_columns>();

    data2_d(cols, rows) = data1_d(rows, cols);

    data4_d(cols, rows) = data3_d(SArows, SRcols);  // mixed static_all and static_range
    data5_d(cols, rows) = data3_d(SArows, SAcols);  // static_all
    data6_d(cols, rows) = data3_d(SRrows, SRcols);  // static_range
    data7_d(cols, rows) = data3_d(rows, SRcols);    // mixed static_range and non-static

  });

  tensor_copy_to_host<policy_t>(data2_vec, data2_ptr);
  tensor_copy_to_host<policy_t>(data4_vec, data4_ptr);
  tensor_copy_to_host<policy_t>(data5_vec, data5_ptr);
  tensor_copy_to_host<policy_t>(data6_vec, data6_ptr);
  tensor_copy_to_host<policy_t>(data7_vec, data7_ptr);


  //
  // Check results
  //
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      //printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1_h(i,j), data2_h(j,i));
      ASSERT_SCALAR_EQ(data1_h(i,j), data2_h(j,i));
      ASSERT_SCALAR_EQ(data3_h(i,j), data4_h(j,i));
      ASSERT_SCALAR_EQ(data3_h(i,j), data5_h(j,i));
      ASSERT_SCALAR_EQ(data3_h(i,j), data6_h(j,i));
      ASSERT_SCALAR_EQ(data3_h(i,j), data7_h(j,i));
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
          data2_h(j,i) = -1;
        }
      }
      tensor_copy_to_device<policy_t>(data2_ptr, data2_vec);


      //
      // Do Operation: Load/Store partial matrix from one view to another
      //
      tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
        // Load data using a View
        auto rows = RAJA::expt::RowIndex<int, matrix_t>::range(0, n_size);
        auto cols = RAJA::expt::ColIndex<int, matrix_t>::range(0, m_size);

        data2_d(cols, rows) = data1_d(rows, cols);
      });

      tensor_copy_to_host<policy_t>(data2_vec, data2_ptr);


      //
      // Check results
      //
      for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
        for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
//          printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1(i,j), data2(i,j));
          if(i < n_size && j < m_size){
            ASSERT_SCALAR_EQ(data1_h(i,j), data2_h(j,i));
          }
          else{
            ASSERT_SCALAR_EQ(element_t(-1), data2_h(j,i));
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
  tensor_free<policy_t>(data3_ptr);
  tensor_free<policy_t>(data4_ptr);
  tensor_free<policy_t>(data5_ptr);
  tensor_free<policy_t>(data6_ptr);
  tensor_free<policy_t>(data7_ptr);
}



TYPED_TEST_P(TestTensorMatrix, ET_LoadStore)
{
  ET_LoadStoreImpl<TypeParam>();
}


#endif
