//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_MATRIX_ET_MatrixMatrixMultiplyAdd_HPP__
#define __TEST_TENSOR_MATRIX_ET_MatrixMatrixMultiplyAdd_HPP__

#include<RAJA/RAJA.hpp>

RAJA_INDEX_VALUE( TX, "TX" );
RAJA_INDEX_VALUE( TY, "TY" );

template <typename MATRIX_TYPE>
void ET_MatrixMatrixMultiplyAddImpl()
{

  using matrix_t = MATRIX_TYPE;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;


  using A_matrix_t = matrix_t;
  using B_matrix_t = typename matrix_t::transpose_type;
  using C_matrix_t = typename matrix_t::product_type;

//  static constexpr camp::idx_t N = 8; //matrix_t::s_num_rows*matrix_t::s_num_columns*2;
  static constexpr camp::idx_t N = RAJA::max<camp::idx_t>(matrix_t::s_num_rows, matrix_t::s_num_columns);
  //
  // Allocate Row-Major Data
  //

  // alloc data1 - The left matrix

  std::vector<element_t> data1_vec(N*N);
  RAJA::TypedView<element_t, RAJA::StaticLayout<RAJA::PERM_IJ,N,N>, TX, TY> data1_h(data1_vec.data());

  element_t *data1_ptr = tensor_malloc<policy_t>(data1_vec);
  RAJA::TypedView<element_t, RAJA::StaticLayout<RAJA::PERM_IJ,N,N>, TX, TY> data1_d(data1_ptr);


  // alloc data2 - The right matrix

  std::vector<element_t> data2_vec(N*N);
  RAJA::TypedView<element_t, RAJA::StaticLayout<RAJA::PERM_IJ,N,N>, TX, TY> data2_h(data2_vec.data());

  element_t *data2_ptr = tensor_malloc<policy_t>(data2_vec);
  RAJA::TypedView<element_t, RAJA::StaticLayout<RAJA::PERM_IJ,N,N>, TX, TY> data2_d(data2_ptr);


  // alloc data3 - The result matrix

  std::vector<element_t> data3_vec(N*N);
  RAJA::TypedView<element_t, RAJA::Layout<2>, TX, TY> data3_h(data3_vec.data(),  N, N);

  element_t *data3_ptr = tensor_malloc<policy_t>(data3_vec);
  RAJA::TypedView<element_t, RAJA::Layout<2>, TX, TY> data3_d(data3_ptr,  N, N);



  // Fill data1 and data2
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      data1_h(i,j) = 1+i*N+j;
      data2_h(i,j) = 3+i*N+j;
      data3_h(i,j) = 5*i+13*j;
    }

  }

//  printf("data1:\n");
//  for(camp::idx_t i = 0;i < N; ++ i){
//    printf("  ");
//    for(camp::idx_t j = 0;j < N; ++ j){
//      printf("%lf  ", (double)data1_h(i,j));
//    }
//    printf("\n");
//  }


//  printf("data2:\n");
//  for(camp::idx_t i = 0;i < N; ++ i){
//    printf("  ");
//    for(camp::idx_t j = 0;j < N; ++ j){
//      printf("%lf  ", (double)data2_h(i,j));
//    }
//    printf("\n");
//  }

  tensor_copy_to_device<policy_t>(data1_ptr, data1_vec);
  tensor_copy_to_device<policy_t>(data2_ptr, data2_vec);
  tensor_copy_to_device<policy_t>(data3_ptr, data3_vec);


  //
  // Do Operation: A*B
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    auto A_rows = RAJA::expt::RowIndex<int, A_matrix_t>::all();
    auto A_cols = RAJA::expt::ColIndex<int, A_matrix_t>::template static_range<0,N>();

    auto B_rows = RAJA::expt::RowIndex<int, B_matrix_t>::template static_range<0,N>();
    auto B_cols = RAJA::expt::ColIndex<int, B_matrix_t>::static_all();

    auto C_rows = RAJA::expt::RowIndex<int, C_matrix_t>::all();
    auto C_cols = RAJA::expt::ColIndex<int, C_matrix_t>::all();

    data3_d(C_rows, C_cols) += data1_d(A_rows, A_cols) * data2_d(B_rows, B_cols);

  });

  tensor_copy_to_host<policy_t>(data3_vec, data3_ptr);

//  printf("data3:\n");
//  for(camp::idx_t i = 0;i < N; ++ i){
//    printf("  ");
//    for(camp::idx_t j = 0;j < N; ++ j){
//      printf("%lf  ", (double)data3_h(i,j));
//    }
//    printf("\n");
//  }

  //
  // Check results
  //
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      element_t expected(5*i+13*j);
      for(camp::idx_t k = 0;k < N; ++ k){
        expected += data1_h(i,k)*data2_h(k,j);
      }
//    printf("i=%d, j=%d, expected=%e, data3=%e\n", (int)i, (int)j, (double)expected, (double)data3_h(i,j));

      ASSERT_SCALAR_EQ(expected, data3_h(i,j));
//      data3_h(i,j) = expected;

    }
  }

//  printf("expected:\n");
//  for(camp::idx_t i = 0;i < N; ++ i){
//    printf("  ");
//    for(camp::idx_t j = 0;j < N; ++ j){
//      printf("%lf  ", (double)data3_h(i,j));
//    }
//    printf("\n");
//  }


  //
  // Loop over all possible sub-matrix sizes for A*x
  //
  for(camp::idx_t n_size = 0;n_size <= N; ++ n_size){
    for(camp::idx_t m_size = 0;m_size <= N; ++ m_size){
//      printf("Running %d x %d\n", (int)n_size, (int)m_size);
      //
      // Clear data3
      //
      for(camp::idx_t i = 0;i < N; ++ i){
        for(camp::idx_t j = 0;j < N; ++ j){
          data3_h(i,j) = 5*i+13*j;
        }
      }

      tensor_copy_to_device<policy_t>(data3_ptr, data3_vec);


      //
      // Do Operation A*B
      //
      tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

        auto A_rows = RAJA::expt::RowIndex<int, A_matrix_t>::range(0, n_size);
        auto A_cols = RAJA::expt::ColIndex<int, A_matrix_t>::range(0, m_size);

        auto B_rows = RAJA::expt::RowIndex<int, B_matrix_t>::range(0, m_size);
        auto B_cols = RAJA::expt::ColIndex<int, B_matrix_t>::range(0, n_size);

        auto C_rows = RAJA::expt::RowIndex<int, C_matrix_t>::range(0, n_size);
        auto C_cols = RAJA::expt::ColIndex<int, C_matrix_t>::range(0, n_size);


        data3_d(C_rows, C_cols) += data1_d(A_rows, A_cols) * data2_d(B_rows, B_cols);
      });

      tensor_copy_to_host<policy_t>(data3_vec, data3_ptr);


      //
      // Check results
      //
      //
      // Check results
      //
      for(camp::idx_t i = 0;i < n_size; ++ i){
        for(camp::idx_t j = 0;j < n_size; ++ j){
          element_t expected(5*i+13*j);
          for(camp::idx_t k = 0;k < m_size; ++ k){
            expected += data1_h(i,k)*data2_h(k,j);
          }
    //    printf("i=%d, j=%d, expected=%e, data3=%e\n", (int)i, (int)j, (double)expected, (double)data3_h(i,j));

          ASSERT_SCALAR_EQ(expected, data3_h(i,j));
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

}



TYPED_TEST_P(TestTensorMatrix, ET_MatrixMatrixMultiplyAdd)
{
  ET_MatrixMatrixMultiplyAddImpl<TypeParam>();
}


#endif
