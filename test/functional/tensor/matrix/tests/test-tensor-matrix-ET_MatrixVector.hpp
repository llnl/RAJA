//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_MATRIX_ET_MatrixVector_HPP__
#define __TEST_TENSOR_MATRIX_ET_MatrixVector_HPP__

#include<RAJA/RAJA.hpp>

template <typename MATRIX_TYPE>
void ET_MatrixVectorImpl()
{

  using matrix_t = MATRIX_TYPE;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;

  using cvector_t = typename matrix_t::column_vector_type;
  using rvector_t = typename matrix_t::row_vector_type;

//  static constexpr camp::idx_t N = 8; //matrix_t::s_num_rows*matrix_t::s_num_columns*2;
  static constexpr camp::idx_t N = RAJA::max<camp::idx_t>(matrix_t::s_num_rows, matrix_t::s_num_columns)*2;
  //
  // Allocate Row-Major Data
  //

  // alloc data1 - The matrix

  std::vector<element_t> data1_vec(N*N);
  RAJA::View<element_t, RAJA::StaticLayout<RAJA::PERM_IJ,N,N>> data1_h(data1_vec.data());

  element_t *data1_ptr = tensor_malloc<policy_t>(data1_vec);
  RAJA::View<element_t, RAJA::StaticLayout<RAJA::PERM_IJ,N,N>> data1_d(data1_ptr);


  // alloc data2 - The input vector

  std::vector<element_t> data2_vec(N);
  RAJA::View<element_t, RAJA::StaticLayout<RAJA::PERM_I,N>> data2_h(data2_vec.data());

  element_t *data2_ptr = tensor_malloc<policy_t>(data2_vec);
  RAJA::View<element_t, RAJA::StaticLayout<RAJA::PERM_I,N>> data2_d(data2_ptr);


  // alloc data3 - The output vector

  std::vector<element_t> data3_vec(N);
  RAJA::View<element_t, RAJA::Layout<1,int,0>> data3_h(data3_vec.data(),  N);

  element_t *data3_ptr = tensor_malloc<policy_t>(data3_vec);
  RAJA::View<element_t, RAJA::Layout<1,int,0>> data3_d(data3_ptr,  N);



  // Fill data1 and data2
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      data1_h(i,j) = 3+i*N+j;
    }
    data2_h(i) = i+1;
  }

//  printf("data1:\n");
//  for(camp::idx_t i = 0;i < N; ++ i){
//    printf("  ");
//    for(camp::idx_t j = 0;j < N; ++ j){
//      printf("%lf  ", (double)data1_h(i,j));
//    }
//    printf("\n");
//  }


//  for(camp::idx_t i = 0;i < N; ++ i){
//    printf("data2[%d]=%lf\n", (int)i, (double)data2_h(i));
//  }

  tensor_copy_to_device<policy_t>(data1_ptr, data1_vec);
  tensor_copy_to_device<policy_t>(data2_ptr, data2_vec);


  //
  // Do Operation: A*x
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    auto rows = RAJA::expt::RowIndex<int, matrix_t>::static_all();
    auto cols = RAJA::expt::ColIndex<int, matrix_t>::template static_range<0,N>();

    auto vrow = RAJA::expt::VectorIndex<int, rvector_t>::static_all();
    auto vcol = RAJA::expt::VectorIndex<int, cvector_t>::all();

    data3_d(vcol) = data1_d(rows, cols) * data2_d(vrow);

  });

  tensor_copy_to_host<policy_t>(data3_vec, data3_ptr);


  //
  // Check results
  //
  for(camp::idx_t i = 0;i < N; ++ i){

    element_t expected(0);
    for(camp::idx_t j = 0;j < N; ++ j){
      expected += data1_h(i,j)*data2_h(j);
    }
//    printf("i=%d, expected=%e, data3=%e\n", (int)i, (double)expected, (double)data3_h(i));

    ASSERT_SCALAR_EQ(expected, data3_h(i));
  }

//return;


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
        data3_h(i) = 0;
      }

      tensor_copy_to_device<policy_t>(data3_ptr, data3_vec);


      //
      // Do Operation (x')*A
      //
      tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
        // Load data using a View
        auto rows = RAJA::expt::RowIndex<int, matrix_t>::range(0, n_size);
        auto cols = RAJA::expt::ColIndex<int, matrix_t>::range(0, m_size);

        auto vrow = RAJA::expt::VectorIndex<int, rvector_t>::range(0, m_size);
        auto vcol = RAJA::expt::VectorIndex<int, cvector_t>::range(0, n_size);

        data3_d(vcol) = data1_d(rows, cols) * data2_d(vrow);
      });

      tensor_copy_to_host<policy_t>(data3_vec, data3_ptr);


      //
      // Check results
      //
      for(camp::idx_t i = 0;i < n_size; ++ i){


        element_t expected(0);
        for(camp::idx_t j = 0;j < m_size; ++ j){
          expected += data1_h(i,j) * data2_h(j);
        }

        if(i >= n_size || m_size == 0){
          expected = 0;
        }

//        printf("i=%d, expected=%e, data3=%e\n", (int)i, (double)expected, (double)data3_h(i));
        ASSERT_SCALAR_EQ(expected, data3_h(i));

      }


    }
  }



  //
  // Do Operation: (x')*A
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    auto rows = RAJA::expt::RowIndex<int, matrix_t>::static_all();
    auto cols = RAJA::expt::ColIndex<int, matrix_t>::static_all();

    auto vrow = RAJA::expt::VectorIndex<int, rvector_t>::static_all();
    auto vcol = RAJA::expt::VectorIndex<int, cvector_t>::static_all();

    data3_d(vrow) =  data2_d(vcol) * data1_d(rows, cols);

  });

  tensor_copy_to_host<policy_t>(data3_vec, data3_ptr);


  //
  // Check results
  //
  for(camp::idx_t j = 0;j < N; ++ j){


    element_t expected(0);
    for(camp::idx_t i = 0;i < N; ++ i){
      expected += data2_h(i)*data1_h(i,j);
    }

    ASSERT_SCALAR_EQ(expected, data3_h(j));
//    printf("i=%d, data3=%lf, expected=%lf\n", (int)j, (double)data3_h(j), (double)expected);
  }




  //
  // Loop over all possible sub-matrix sizes for (x')*A
  //
  for(camp::idx_t n_size = 0;n_size <= N; ++ n_size){
    for(camp::idx_t m_size = 0;m_size <= N; ++ m_size){
//      printf("Running %d x %d\n", (int)n_size, (int)m_size);
      //
      // Clear data3
      //
      for(camp::idx_t j = 0;j < N; ++ j){
        data3_h(j) = 0;
      }

      tensor_copy_to_device<policy_t>(data3_ptr, data3_vec);


      //
      // Do Operation (x')*A
      //
      tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
        // Load data using a View
        auto rows = RAJA::expt::RowIndex<int, matrix_t>::range(0, n_size);
        auto cols = RAJA::expt::ColIndex<int, matrix_t>::range(0, m_size);

        auto vrow = RAJA::expt::VectorIndex<int, rvector_t>::range(0, m_size);
        auto vcol = RAJA::expt::VectorIndex<int, cvector_t>::range(0, n_size);

        data3_d(vrow) =  data2_d(vcol) * data1_d(rows, cols);
      });

      tensor_copy_to_host<policy_t>(data3_vec, data3_ptr);


      //
      // Check results
      //
      for(camp::idx_t j = 0;j < N; ++ j){

        element_t expected(0);

        for(camp::idx_t i = 0;i < n_size; ++ i){
          expected += data2_h(i) * data1_h(i,j);
        }

        if(j >= m_size || n_size == 0){
          expected = 0;
        }

//        printf("j=%d, expected=%e, data3=%e\n", (int)j, (double)expected, (double)data3_h(j));
        ASSERT_SCALAR_EQ(expected, data3_h(j));

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



TYPED_TEST_P(TestTensorMatrix, ET_MatrixVector)
{
  ET_MatrixVectorImpl<TypeParam>();
}


#endif
