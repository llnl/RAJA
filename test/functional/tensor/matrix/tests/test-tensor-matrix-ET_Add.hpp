//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_MATRIX_ET_Add_HPP__
#define __TEST_TENSOR_MATRIX_ET_Add_HPP__

#include<RAJA/RAJA.hpp>

template <typename MATRIX_TYPE>
void ET_AddImpl()
{

  using matrix_t = MATRIX_TYPE;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;


  static constexpr camp::idx_t N = RAJA::max<camp::idx_t>(matrix_t::s_num_rows, matrix_t::s_num_columns)*2;

  //
  // Allocate Row-Major Data
  //

  // alloc data1
  std::vector<element_t> data1_vec(N*N);
  RAJA::View<element_t, RAJA::Layout<2>> data1_h(data1_vec.data(), N, N);

  element_t *data1_ptr = tensor_malloc<policy_t>(data1_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data1_d(data1_ptr,  N, N);


  // alloc data2
  std::vector<element_t> data2_vec(N*N);
  RAJA::View<element_t, RAJA::Layout<2>> data2_h(data2_vec.data(),  N, N);

  element_t *data2_ptr = tensor_malloc<policy_t>(data2_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data2_d(data2_ptr,  N, N);


  // alloc data3 with StaticLayout
  std::vector<element_t> data3_vec(N*N);
  RAJA::View<element_t, RAJA::StaticLayout<RAJA::PERM_IJ,N,N>> data3_h(data3_vec.data());

  element_t *data3_ptr = tensor_malloc<policy_t>(data3_vec);
  RAJA::View<element_t, RAJA::StaticLayout<RAJA::PERM_IJ,N,N>> data3_d(data3_ptr);


  // alloc data4 with StaticLayout
  std::vector<element_t> data4_vec(N*N);
  RAJA::View<element_t, RAJA::StaticLayout<RAJA::PERM_IJ,N,N>> data4_h(data4_vec.data());

  element_t *data4_ptr = tensor_malloc<policy_t>(data4_vec);
  RAJA::View<element_t, RAJA::StaticLayout<RAJA::PERM_IJ,N,N>> data4_d(data4_ptr);


  // alloc data5
  std::vector<element_t> data5_vec(N*N);
  RAJA::View<element_t, RAJA::Layout<2>> data5_h(data5_vec.data(),  N, N);

  element_t *data5_ptr = tensor_malloc<policy_t>(data5_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data5_d(data5_ptr,  N, N);



  // Fill data1, data2, data3, and data4
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      data1_h(i,j) = i*matrix_t::s_num_columns+j;
      data2_h(i,j) = 1+i+j;
      data3_h(i,j) = i*matrix_t::s_num_columns+j;
      data4_h(i,j) = 1+i+j;
    }
  }

  tensor_copy_to_device<policy_t>(data1_ptr, data1_vec);
  tensor_copy_to_device<policy_t>(data2_ptr, data2_vec);
  tensor_copy_to_device<policy_t>(data3_ptr, data3_vec);
  tensor_copy_to_device<policy_t>(data4_ptr, data4_vec);


  //
  // Do Operation: Full sum of data1, data2, data3, and data4
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    auto rows = RAJA::expt::RowIndex<int, matrix_t>::all();
    auto cols = RAJA::expt::ColIndex<int, matrix_t>::all();

    auto SArows = RAJA::expt::RowIndex<int, matrix_t>::static_all();
    auto SAcols = RAJA::expt::ColIndex<int, matrix_t>::static_all();

    auto SRrows = RAJA::expt::RowIndex<int, matrix_t>::template static_range<0,N>();
    auto SRcols = RAJA::expt::ColIndex<int, matrix_t>::template static_range<0,N>();

    // Access types:
    // data1_d - Layout with all() and all().
    // data2_d - Layout with all() and static_range(), which should default to normal Layout access.
    // data3_d - StaticLayout with static_all() and static_range().
    // data4_d - StaticLayout with static_all() and all().

    data5_d(cols, rows) = data1_d(rows, cols) + data2_d(cols, SRrows) + data3_d(SArows, SRcols) + data4_d(SAcols, rows);

  });

  tensor_copy_to_host<policy_t>(data5_vec, data5_ptr);


  //
  // Check results
  //
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      ASSERT_SCALAR_EQ(data5_h(j,i), data1_h(i,j)+data2_h(j,i)+data3_h(i,j)+data4_h(j,i));
//      printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1(i,j), data2(i,j));
    }
  }



  //
  // Loop over all possible sub-matrix sizes using the load_*_nm routines
  //
  for(camp::idx_t n_size = 0;n_size <= N; ++ n_size){
    for(camp::idx_t m_size = 0;m_size <= N; ++ m_size){
//      printf("Running %d x %d\n", (int)n_size, (int)m_size);
      //
      // Clear data5
      //
      for(camp::idx_t i = 0;i < N; ++ i){
        for(camp::idx_t j = 0;j < N; ++ j){
          data5_h(j,i) = -1;
        }
      }
      tensor_copy_to_device<policy_t>(data5_ptr, data5_vec);


      //
      // Do Operation: Perform partial sum
      //
      tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
        // Load data using a View
        auto rows = RAJA::expt::RowIndex<int, matrix_t>::range(0, n_size);
        auto cols = RAJA::expt::ColIndex<int, matrix_t>::range(0, m_size);

        // Access types:
        // Layout with range() and range() because loop iterate cannot be determined statically.

        data5_d(cols, rows) = data1_d(rows, cols) + data2_d(cols, rows);
      });

      tensor_copy_to_host<policy_t>(data5_vec, data5_ptr);


      //
      // Check results
      //
      for(camp::idx_t i = 0;i < N; ++ i){
        for(camp::idx_t j = 0;j < N; ++ j){
//          printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1(i,j), data2(i,j));
          if(i < n_size && j < m_size){
            ASSERT_SCALAR_EQ(data5_h(j,i), data1_h(i,j)+data2_h(j,i));
          }
          else{
            ASSERT_SCALAR_EQ(element_t(-1), data5_h(j,i));
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
}



TYPED_TEST_P(TestTensorMatrix, ET_Add)
{
  ET_AddImpl<TypeParam>();
}


#endif
