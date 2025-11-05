//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <cassert>

#include "camp/camp.hpp"
#include "camp/concepts.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

using namespace RAJA;
using namespace RAJA::statement;

//Define tile size ( TILE_DIM x TILE_DIM )
//Matrix transpose and matrix multiplication
//are carried out via tiling algorithms
RAJA_INDEX_VALUE(TX, "TX");
RAJA_INDEX_VALUE(TY, "TY");

const int TILE_DIM = 16;

template <typename NestedPolicy>
class TypedLocalMem : public ::testing::Test
{

  virtual void SetUp() {}
  virtual void TearDown() {}
};
TYPED_TEST_SUITE_P(TypedLocalMem);

GPU_TYPED_TEST_P(TypedLocalMem, Basic)
{
  using Pol = at_v<TypeParam, 0>;

  const int DIM = 2;
  const int N_rows = 144;
  const int N_cols = 255;

  const int inner_Dim0 = TILE_DIM;
  const int inner_Dim1 = TILE_DIM;

  const int outer_Dim0 = (N_cols-1)/TILE_DIM+1;
  const int outer_Dim1 = (N_rows-1)/TILE_DIM+1;

  double *A, *B;
#if defined(RAJA_ENABLE_CUDA)
  size_t Arr_sz = N_rows * N_cols;
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, &A,  sizeof(double) * Arr_sz);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, &B, sizeof(double)  * Arr_sz);
#else
  A  = new double[N_rows * N_cols];
  B  = new double[N_rows * N_cols];
#endif

  RAJA::TypedView<double, RAJA::Layout<DIM>, TY, TX> Aview(A, N_rows, N_cols);
  RAJA::TypedView<double, RAJA::Layout<DIM>, TY, TX> Bview(B, N_rows, N_cols);

  for (int row = 0; row < N_rows; ++row) {
    for (int col= 0 ; col < N_cols; ++col) {
      A[col + N_cols*row] = col;
    }
  }

  using SharedTile = AtomicTypedLocalArray<RAJA::auto_atomic, double, RAJA::PERM_IJ, RAJA::SizeList<TILE_DIM,TILE_DIM>, TY, TX>;
  SharedTile myTile, myTile2;

  const TX TX_TILE_DIM(16);
  const TY TY_TILE_DIM(16);

  RAJA::kernel_param<Pol>(RAJA::make_tuple(RAJA::TypedRangeSegment<TX>(0, inner_Dim0), RAJA::TypedRangeSegment<TY>(0,inner_Dim1),
                                           RAJA::TypedRangeSegment<TX>(0, outer_Dim0), RAJA::TypedRangeSegment<TY>(0,outer_Dim1)),
                          RAJA::make_tuple(myTile, myTile2),

  //Load data into shared memory
  [=] RAJA_HOST_DEVICE (TX tx, TY ty, TX bx, TY by, SharedTile &_myTile, SharedTile &) {

    TX col = bx * TX_TILE_DIM + tx;  // Matrix column index
    TY row = by * TY_TILE_DIM + ty;  // Matrix row index

    if(row < N_rows && col < N_cols){
      _myTile(ty,tx)   = Aview(row, col);
    }

  },

  //read from shared mem
  [=] RAJA_HOST_DEVICE (TX tx, TY ty, TX bx, TY by, SharedTile &_myTile, SharedTile &) {

    TX col = bx * TX_TILE_DIM + tx;  // Matrix column index
    TY row = by * TY_TILE_DIM + ty;  // Matrix row index

    if(row < N_rows && col < N_cols){
      Bview(row, col) = _myTile(ty, tx);
    }

  });

  //Check result
  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
      ASSERT_FLOAT_EQ((double)B[col + row*N_cols], (double)A[col + row*N_cols]);
    }
  }

#if defined(RAJA_ENABLE_CUDA)
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, A);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, B);
#else
  delete [] A;
  delete [] B;
#endif
}

REGISTER_TYPED_TEST_SUITE_P(TypedLocalMem, Basic);

#if defined(RAJA_ENABLE_HIP)
template <typename NestedPolicy>
class TypedLocalMem_gpu : public ::testing::Test
{

  virtual void SetUp() {}
  virtual void TearDown() {}
};
TYPED_TEST_SUITE_P(TypedLocalMem_gpu);

GPU_TYPED_TEST_P(TypedLocalMem_gpu, Basic)
{
  using Pol = at_v<TypeParam, 0>;

  const int DIM = 2;
  const int N_rows = 144;
  const int N_cols = 255;

  const int inner_Dim0 = TILE_DIM;
  const int inner_Dim1 = TILE_DIM;

  const int outer_Dim0 = (N_cols-1)/TILE_DIM+1;
  const int outer_Dim1 = (N_rows-1)/TILE_DIM+1;

  double *A, *B;
  double *d_A, *d_B;
  size_t Arr_sz = N_rows * N_cols;
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMalloc, &d_A, sizeof(double) * Arr_sz);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMalloc, &d_B, sizeof(double) * Arr_sz);
  A  = new double[N_rows * N_cols];
  B  = new double[N_rows * N_cols];

  RAJA::TypedView<double, RAJA::Layout<DIM>, TY, TX> Aview(A, N_rows, N_cols);
  RAJA::TypedView<double, RAJA::Layout<DIM>, TY, TX> Bview(B, N_rows, N_cols);
  RAJA::TypedView<double, RAJA::Layout<DIM>, TY, TX> d_Aview(d_A, N_rows, N_cols);
  RAJA::TypedView<double, RAJA::Layout<DIM>, TY, TX> d_Bview(d_B, N_rows, N_cols);

  for (int row = 0; row < N_rows; ++row) {
    for (int col= 0 ; col < N_cols; ++col) {
      A[col + N_cols*row] = col;
    }
  }

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_A, A, Arr_sz*sizeof(double), hipMemcpyHostToDevice);

  using SharedTile = TypedLocalArray<double, RAJA::PERM_IJ, RAJA::SizeList<TILE_DIM,TILE_DIM>, TY, TX>;
  SharedTile myTile, myTile2;

  const TX TX_TILE_DIM(16);
  const TY TY_TILE_DIM(16);

  RAJA::kernel_param<Pol>(RAJA::make_tuple(RAJA::TypedRangeSegment<TX>(0, inner_Dim0), RAJA::TypedRangeSegment<TY>(0,inner_Dim1),
                                           RAJA::TypedRangeSegment<TX>(0, outer_Dim0), RAJA::TypedRangeSegment<TY>(0,outer_Dim1)),
                          RAJA::make_tuple(myTile, myTile2),

  //Load data into shared memory
  [=] RAJA_HOST_DEVICE (TX tx, TY ty, TX bx, TY by, SharedTile &myTile, SharedTile &) {

    TX col = bx * TX_TILE_DIM + tx;  // Matrix column index
    TY row = by * TY_TILE_DIM + ty;  // Matrix row index

    if(row < N_rows && col < N_cols){
      myTile(ty,tx)   = d_Aview(row, col);
    }

  },

  //read from shared mem
  [=] RAJA_HOST_DEVICE (TX tx, TY ty, TX bx, TY by, SharedTile &myTile, SharedTile &) {

    TX col = bx * TX_TILE_DIM + tx;  // Matrix column index
    TY row = by * TY_TILE_DIM + ty;  // Matrix row index

    if(row < N_rows && col < N_cols){
      d_Bview(row, col) = myTile(ty, tx);
    }

  });

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, B, d_B, Arr_sz*sizeof(double), hipMemcpyDeviceToHost);

  //Check result
  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
      ASSERT_FLOAT_EQ(B[col + row*N_cols], A[col + row*N_cols]);
    }
  }

  CAMP_HIP_API_INVOKE_AND_CHECK(hipFree, d_A);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipFree, d_B);
  delete [] A;
  delete [] B;
}

REGISTER_TYPED_TEST_SUITE_P(TypedLocalMem_gpu, Basic);
#endif //defined(RAJA_ENABLE_HIP)


//
//Matrix transpose example - test all variants
//
template <typename NestedPolicy>
class MatTranspose : public ::testing::Test
{

  virtual void SetUp() {}
  virtual void TearDown() {}
};
TYPED_TEST_SUITE_P(MatTranspose);

GPU_TYPED_TEST_P(MatTranspose, Basic)
{

  using Pol = at_v<TypeParam, 0>;

  const int DIM = 2;
  const int N_rows = 144;
  const int N_cols = 255;

  const int inner_Dim0 = TILE_DIM;
  const int inner_Dim1 = TILE_DIM;

  const int outer_Dim0 = (N_cols-1)/TILE_DIM+1;
  const int outer_Dim1 = (N_rows-1)/TILE_DIM+1;

  double *A, *At, *B, *Bt;
#if defined(RAJA_ENABLE_CUDA)
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, &A,  sizeof(double) * N_rows * N_cols);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, &At, sizeof(double) * N_rows * N_cols);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, &B,  sizeof(double) * N_rows * N_cols);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, &Bt, sizeof(double) * N_rows * N_cols);
#else
  A  = new double[N_rows * N_cols];
  At = new double[N_rows * N_cols];
  B  = new double[N_rows * N_cols];
  Bt = new double[N_rows * N_cols];
#endif

  RAJA::View<double, RAJA::Layout<DIM>> Aview(A, N_rows, N_cols);
  RAJA::View<double, RAJA::Layout<DIM>> Atview(At, N_cols, N_rows);

  RAJA::View<double, RAJA::Layout<DIM>> Bview(B, N_rows, N_cols);
  RAJA::View<double, RAJA::Layout<DIM>> Btview(Bt, N_cols, N_rows);


  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
      Aview(row, col) = col;
      Bview(row, col) = col;
    }
  }


  using SharedTile = LocalArray<double, RAJA::PERM_IJ, RAJA::SizeList<TILE_DIM,TILE_DIM>>;

  SharedTile myTile, myTile2;

  RAJA::kernel_param<Pol>(RAJA::make_tuple(RAJA::RangeSegment(0, inner_Dim0), RAJA::RangeSegment(0,inner_Dim1),
                                           RAJA::RangeSegment(0, outer_Dim0), RAJA::RangeSegment(0,outer_Dim1)),
                          RAJA::make_tuple(myTile, myTile2),

  //Load data into shared memory
  [=] RAJA_HOST_DEVICE (int tx, int ty, int bx, int by, SharedTile &_myTile, SharedTile &_myTile2) {

    int col = bx * TILE_DIM + tx;  // Matrix column index
    int row = by * TILE_DIM + ty;  // Matrix row index

    if(row < N_rows && col < N_cols){
      _myTile(ty,tx)  = Aview(row, col);
      _myTile2(ty,tx) = Bview(row, col);
    }

  },

  //read from shared mem
  [=] RAJA_HOST_DEVICE (int tx, int ty, int bx, int by, SharedTile &_myTile, SharedTile &_myTile2) {

    int col = by * TILE_DIM + tx;  // Transposed matrix column index
    int row = bx * TILE_DIM + ty;  // Transposed matrix row index

    if(row < N_cols && col < N_rows){
      Atview(row, col) = _myTile(tx,ty);
      Btview(row, col) = _myTile2(tx,ty);
    }

  });

  //Check result
  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
      ASSERT_FLOAT_EQ((double)Atview(col,row), (double)col);
      ASSERT_FLOAT_EQ((double)Btview(col,row), (double)col);
    }
  }


#if defined(RAJA_ENABLE_CUDA)
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, A);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, At);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, B);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, Bt);
#else
  delete [] A;
  delete [] At;
  delete [] B;
  delete [] Bt;
#endif
}

REGISTER_TYPED_TEST_SUITE_P(MatTranspose, Basic);

#if defined(RAJA_ENABLE_HIP)

template <typename NestedPolicy>
class MatTranspose_gpu : public ::testing::Test
{

  virtual void SetUp() {}
  virtual void TearDown() {}
};
TYPED_TEST_SUITE_P(MatTranspose_gpu);

GPU_TYPED_TEST_P(MatTranspose_gpu, Basic)
{

  using Pol = at_v<TypeParam, 0>;

  const int DIM = 2;
  const int N_rows = 144;
  const int N_cols = 255;

  const int inner_Dim0 = TILE_DIM;
  const int inner_Dim1 = TILE_DIM;

  const int outer_Dim0 = (N_cols-1)/TILE_DIM+1;
  const int outer_Dim1 = (N_rows-1)/TILE_DIM+1;

  double *A, *At, *B, *Bt;
  double *d_A, *d_At, *d_B, *d_Bt;
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMalloc, &d_A,  sizeof(double) * N_rows * N_cols);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMalloc, &d_At, sizeof(double) * N_rows * N_cols);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMalloc, &d_B,  sizeof(double) * N_rows * N_cols);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMalloc, &d_Bt, sizeof(double) * N_rows * N_cols);
  A  = new double[N_rows * N_cols];
  At = new double[N_rows * N_cols];
  B  = new double[N_rows * N_cols];
  Bt = new double[N_rows * N_cols];

  RAJA::View<double, RAJA::Layout<DIM>> Aview(A, N_rows, N_cols);
  RAJA::View<double, RAJA::Layout<DIM>> Atview(At, N_cols, N_rows);

  RAJA::View<double, RAJA::Layout<DIM>> Bview(B, N_rows, N_cols);
  RAJA::View<double, RAJA::Layout<DIM>> Btview(Bt, N_cols, N_rows);

  RAJA::View<double, RAJA::Layout<DIM>> d_Aview(d_A, N_rows, N_cols);
  RAJA::View<double, RAJA::Layout<DIM>> d_Atview(d_At, N_cols, N_rows);

  RAJA::View<double, RAJA::Layout<DIM>> d_Bview(d_B, N_rows, N_cols);
  RAJA::View<double, RAJA::Layout<DIM>> d_Btview(d_Bt, N_cols, N_rows);


  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
      Aview(row, col) = col;
      Bview(row, col) = col;
    }
  }

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_A, A, N_rows * N_cols * sizeof(double), hipMemcpyHostToDevice);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_B, B, N_rows * N_cols * sizeof(double), hipMemcpyHostToDevice);


  using SharedTile = LocalArray<double, RAJA::PERM_IJ, RAJA::SizeList<TILE_DIM,TILE_DIM>>;

  SharedTile myTile, myTile2;

  RAJA::kernel_param<Pol>(RAJA::make_tuple(RAJA::RangeSegment(0, inner_Dim0), RAJA::RangeSegment(0,inner_Dim1),
                                           RAJA::RangeSegment(0, outer_Dim0), RAJA::RangeSegment(0,outer_Dim1)),
                          RAJA::make_tuple(myTile, myTile2),

  //Load data into shared memory
  [=] RAJA_HOST_DEVICE (int tx, int ty, int bx, int by, SharedTile &myTile, SharedTile &myTile2) {

    int col = bx * TILE_DIM + tx;  // Matrix column index
    int row = by * TILE_DIM + ty;  // Matrix row index

    if(row < N_rows && col < N_cols){
      myTile(ty,tx)  = d_Aview(row, col);
      myTile2(ty,tx) = d_Bview(row, col);
    }

  },

  //read from shared mem
  [=] RAJA_HOST_DEVICE (int tx, int ty, int bx, int by, SharedTile &myTile, SharedTile &myTile2) {

    int col = by * TILE_DIM + tx;  // Transposed matrix column index
    int row = bx * TILE_DIM + ty;  // Transposed matrix row index

    if(row < N_cols && col < N_rows){
      d_Atview(row, col) = myTile(tx,ty);
      d_Btview(row, col) = myTile2(tx,ty);
    }

  });

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, At, d_At, N_rows * N_cols * sizeof(double), hipMemcpyDeviceToHost);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, Bt, d_Bt, N_rows * N_cols * sizeof(double), hipMemcpyDeviceToHost);

  //Check result
  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
      ASSERT_FLOAT_EQ(Atview(col,row), col);
      ASSERT_FLOAT_EQ(Btview(col,row), col);
    }
  }


  CAMP_HIP_API_INVOKE_AND_CHECK(hipFree, d_A);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipFree, d_At);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipFree, d_B);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipFree, d_Bt);
  delete [] A;
  delete [] At;
  delete [] B;
  delete [] Bt;
}

REGISTER_TYPED_TEST_SUITE_P(MatTranspose_gpu, Basic);

#endif //defined(RAJA_ENABLE_HIP)

using SeqTypes =
  ::testing::Types<
  RAJA::list<
    RAJA::KernelPolicy<
        RAJA::statement::For<3, RAJA::seq_exec,
          RAJA::statement::For<2, RAJA::seq_exec,

          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<0,1>,

              //Load data into shared memory
              RAJA::statement::For<1, RAJA::seq_exec,
                RAJA::statement::For<0, RAJA::seq_exec,
                  RAJA::statement::Lambda<0>
                                   >
                                 >,

                //Read data from shared memory
                RAJA::statement::For<1, RAJA::seq_exec,
                  RAJA::statement::For<0, RAJA::seq_exec,
                    RAJA::statement::Lambda<1> > >

              > //close shared memory scope
            >//for 2
        >//for 3
      > //kernel policy
    > //list
  >; //types
INSTANTIATE_TYPED_TEST_SUITE_P(Seq, MatTranspose, SeqTypes);
INSTANTIATE_TYPED_TEST_SUITE_P(Seq, TypedLocalMem, SeqTypes);


#if defined(RAJA_ENABLE_OPENMP)
using TestTypes =
  ::testing::Types<
  RAJA::list<
    RAJA::KernelPolicy<
      RAJA::statement::For<3, RAJA::seq_exec,
        RAJA::statement::For<2, RAJA::seq_exec,

          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<0,1>,

           //Load data into shared memory
           RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                     RAJA::ArgList<0, 1>,
                                     RAJA::statement::Lambda<0>
                                     >,

           //Read data from shared memory
           RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                     RAJA::ArgList<0, 1>,
                                     RAJA::statement::Lambda<1>
                                     >
                                 >
        >//for 2
       >//for 3
       > //close policy
     > //close list

  ,RAJA::list<
      RAJA::KernelPolicy<
      RAJA::statement::For<3, RAJA::seq_exec,
        RAJA::statement::For<2, RAJA::seq_exec,

          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<0,1>,

           //Load data into shared memory
            RAJA::statement::For<1, RAJA::omp_parallel_for_exec,
              RAJA::statement::For<0, RAJA::seq_exec,
                RAJA::statement::Lambda<0>
              >
             >,

           //Read data from shared memory
            RAJA::statement::For<1, RAJA::seq_exec,
           RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
                                RAJA::statement::Lambda<1>
           >
          >
         > //close shared mem window
        > //2
       >//3
     >//close policy
    > //close list
  ,RAJA::list<
    RAJA::KernelPolicy<
      RAJA::statement::For<3, RAJA::omp_parallel_for_exec,
        RAJA::statement::For<2, RAJA::seq_exec,

          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<0,1>,

           //Load data into shared memory
           RAJA::statement::For<1, RAJA::seq_exec,
              RAJA::statement::For<0, RAJA::seq_exec,
                RAJA::statement::Lambda<0>
              >
             >,

           //Read data from shared memory
            RAJA::statement::For<1, RAJA::seq_exec,
              RAJA::statement::For<0, RAJA::seq_exec,
                RAJA::statement::Lambda<1>
           >
          >
         > //close shared mem window
        > //2
       >//3
      > //close policy list
     > //close list
  ,RAJA::list<
    RAJA::KernelPolicy<
           RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                     RAJA::ArgList<2, 3>,

          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem,RAJA::ParamList<0,1>,

           //Load data into shared memory
           RAJA::statement::For<1, RAJA::seq_exec,
              RAJA::statement::For<0, RAJA::seq_exec,
                RAJA::statement::Lambda<0>
              >
             >,

           //Read data from shared memory
            RAJA::statement::For<1, RAJA::seq_exec,
              RAJA::statement::For<0, RAJA::seq_exec,
                RAJA::statement::Lambda<1>
           >
          >
         > //close shared mem window
       >//outer collapsed
      > //close policy list
     > //close list
   >;


INSTANTIATE_TYPED_TEST_SUITE_P(OpenMP, MatTranspose, TestTypes);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMP, TypedLocalMem, TestTypes);
#endif

#if defined(RAJA_ENABLE_CUDA)

using CUDATypes =
  ::testing::Types<
  RAJA::list<
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
        RAJA::statement::For<3, RAJA::cuda_block_y_direct,
          RAJA::statement::For<2, RAJA::cuda_block_x_direct,

            RAJA::statement::InitLocalMem<RAJA::cuda_shared_mem, RAJA::ParamList<0,1>,

              //Load data into shared memory
              RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                  RAJA::statement::Lambda<0> > >,
              RAJA::statement::CudaSyncThreads,

              //Read data from shared memory
              RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                  RAJA::statement::Lambda<1> > >,
              RAJA::statement::CudaSyncThreads
            > //close shared memory scope
          >//for 2
        >//for 3
      > //CudaKernel
    > //kernel policy
  > //list
  ,
  RAJA::list<
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
        RAJA::statement::For<3, RAJA::cuda_block_y_loop,
          RAJA::statement::For<2, RAJA::cuda_block_x_loop,

            RAJA::statement::InitLocalMem<RAJA::cuda_shared_mem, RAJA::ParamList<0,1>,

              //Load data into shared memory
              RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                  RAJA::statement::Lambda<0> > >,
              RAJA::statement::CudaSyncThreads,

              //Read data from shared memory
              RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                  RAJA::statement::Lambda<1> > >,
              RAJA::statement::CudaSyncThreads
            > //close shared memory scope
          >//for 2
        >//for 3
      > //CudaKernel
    > //kernel policy
  > //list
  >; //types
INSTANTIATE_TYPED_TEST_SUITE_P(CUDA, MatTranspose, CUDATypes);
INSTANTIATE_TYPED_TEST_SUITE_P(CUDA, TypedLocalMem, CUDATypes);

#endif

#if defined(RAJA_ENABLE_HIP)

using HIPTypes =
  ::testing::Types<
  RAJA::list<
    RAJA::KernelPolicy<
      RAJA::statement::HipKernel<
        RAJA::statement::For<3, RAJA::hip_block_y_direct,
          RAJA::statement::For<2, RAJA::hip_block_x_direct,

            RAJA::statement::InitLocalMem<RAJA::hip_shared_mem, RAJA::ParamList<0,1>,

              //Load data into shared memory
              RAJA::statement::For<1, RAJA::hip_thread_y_direct,
                RAJA::statement::For<0, RAJA::hip_thread_x_direct,
                  RAJA::statement::Lambda<0>
                >
              >,
              RAJA::statement::HipSyncThreads,

              //Read data from shared memory
              RAJA::statement::For<1, RAJA::hip_thread_y_direct,
                RAJA::statement::For<0, RAJA::hip_thread_x_direct,
                  RAJA::statement::Lambda<1>
                >
              >,
              RAJA::statement::HipSyncThreads
            > //close shared memory scope
          >//for 2
        >//for 3
      > //HipKernel
    > //kernel policy
  > //list
  ,
  RAJA::list<
    RAJA::KernelPolicy<
      RAJA::statement::HipKernel<
        RAJA::statement::For<3, RAJA::hip_block_y_loop,
          RAJA::statement::For<2, RAJA::hip_block_x_loop,

            RAJA::statement::InitLocalMem<RAJA::hip_shared_mem, RAJA::ParamList<0,1>,

              //Load data into shared memory
              RAJA::statement::For<1, RAJA::hip_thread_y_direct,
                RAJA::statement::For<0, RAJA::hip_thread_x_direct,
                  RAJA::statement::Lambda<0>
                >
              >,
              RAJA::statement::HipSyncThreads,

              //Read data from shared memory
              RAJA::statement::For<1, RAJA::hip_thread_y_direct,
                RAJA::statement::For<0, RAJA::hip_thread_x_direct,
                  RAJA::statement::Lambda<1>
                >
              >,
              RAJA::statement::HipSyncThreads
            > //close shared memory scope
          >//for 2
        >//for 3
      > //HipKernel
    > //kernel policy
  > //list
  >; //types
INSTANTIATE_TYPED_TEST_SUITE_P(HIP, MatTranspose_gpu, HIPTypes);
INSTANTIATE_TYPED_TEST_SUITE_P(HIP, TypedLocalMem_gpu, HIPTypes);

#endif



template <typename NestedPolicy>
class MatMultiply : public ::testing::Test
{
  virtual void SetUp(){}
  virtual void TearDown(){}
};

TYPED_TEST_SUITE_P(MatMultiply);

GPU_TYPED_TEST_P(MatMultiply, shmem)
{

  using Pol = typename TypeParam::exec_policy;

  static constexpr size_t N = TypeParam::N;
  static constexpr size_t M = TypeParam::M;
  static constexpr size_t P = TypeParam::P;

  //Matrix A size: N x M
  //Matrix B size: M x P
  //Result C size: N x P

  // Note: on CPU A==d_A, etc.
  double *A, *d_A;
  TypeParam::alloc_double(N*M, &A, &d_A);

  double *B, *d_B;
  TypeParam::alloc_double(M*P, &B, &d_B);

  double *C, *d_C;
  TypeParam::alloc_double(N*P, &C, &d_C);


  double *C_sol = new double[N*P];

  RAJA::View<double, RAJA::Layout<2>> C_solView(C_sol, N, P);

  {
    // Create solution using CPU bare loops
    RAJA::View<double, RAJA::Layout<2>> Aview(A, N, M);
    RAJA::View<double, RAJA::Layout<2>> Bview(B, M, P);
    RAJA::View<double, RAJA::Layout<2>> Cview(C, N, P);
    for (size_t row = 0; row < N; ++row) {
      for (size_t col = 0; col < M; ++col) {
        Aview(row, col) = ((double)col-row)/(N*M)+1;
      }
    }

    for (size_t row = 0; row < M; ++row) {
      for (size_t col = 0; col < P; ++col) {
        Bview(row, col) = ((double)col+row)/(M*P)+1;
      }
    }

    for(size_t r=0; r<N; ++r){
      for(size_t c=0; c<P; ++c){
        double dot = 0.0;
        for(size_t k=0; k<M; ++k){
          dot += Aview(r,k)*Bview(k,c);
        }
        C_solView(r,c) = dot;
        Cview(r,c) = 0;
      }
    }
  }

  // Copy A, B and C to the device (NOP on CPU)
  TypeParam::copy_d2h(N*M, d_A, A);
  TypeParam::copy_d2h(M*P, d_B, B);
  TypeParam::copy_d2h(N*P, d_C, C);

  // Create device views of data
  RAJA::View<double, RAJA::Layout<2>> Aview(d_A, N, M);
  RAJA::View<double, RAJA::Layout<2>> Bview(d_B, M, P);
  RAJA::View<double, RAJA::Layout<2>> Cview(d_C, N, P);

  using Shmem      = typename TypeParam::Shmem;
  using ThreadPriv = typename TypeParam::ThreadPriv;

  Shmem aShared, bShared; //memory to be shared between threads
  ThreadPriv pVal; //iteration dependent data

  RAJA::kernel_param<Pol>(RAJA::make_tuple(RAJA::RangeSegment(0, N),
                                           RAJA::RangeSegment(0, M),
                                           RAJA::RangeSegment(0, P)),
                          RAJA::make_tuple(aShared, bShared, pVal),

  // Zero out thread local memory for storing dot products
  [=] RAJA_HOST_DEVICE (int tn, int tp, ThreadPriv &_pVal) {

    _pVal(tn,tp) = 0.0;

  },

  // Load tile of A
  [=] RAJA_HOST_DEVICE (int n, int m, int tn, int tm, Shmem &_aShared) {

     _aShared(tn, tm) = Aview(n, m);

  },

  // Load tile of B
  [=] RAJA_HOST_DEVICE (int m, int p, int tm, int tp, Shmem &_bShared) {

    _bShared(tm, tp) = Bview(m, p);

  },

  // Do partial update in shmem
  [=] RAJA_HOST_DEVICE (int tn, int tm, int tp, Shmem &_aShared,  Shmem &_bShared, ThreadPriv & _pVal) {

    _pVal(tn,tp) += _aShared(tn,tm) * _bShared(tm, tp);

  },

  // Write out complete result
  [=] RAJA_HOST_DEVICE (int n, int p, int tn, int tp,  ThreadPriv &_pVal) {

    Cview(n,p) = _pVal(tn,tp);

  });

  // copy result back to host (NOP on CPU)
  TypeParam::copy_d2h(N*P, C, d_C);

  // Check result
  RAJA::View<double, RAJA::Layout<2>> Cresult(C, N, P);
  for (size_t row = 0; row < N; ++row) {
    for (size_t col = 0; col < P; ++col) {
      ASSERT_FLOAT_EQ((double)Cresult(row,col), (double)C_solView(row,col));
    }
  }

  TypeParam::free_double(A, d_A);
  TypeParam::free_double(B, d_B);
  TypeParam::free_double(C, d_C);
  delete [] C_sol;
}

REGISTER_TYPED_TEST_SUITE_P(MatMultiply, shmem);

void alloc_cpu(size_t N, double **host, double **device){
  *host = new double[N];
  *device = *host;
}

void copy_h2d_cpu(size_t , double *, double *){
  // NOP
}

void copy_d2h_cpu(size_t , double *, double *){
  // NOP
}

void free_cpu(double *host, double *){
  delete[] host;
}

struct Policy_MatMultiply_cpu {

    static constexpr size_t N = 150;
    static constexpr size_t M = 25;
    static constexpr size_t P = 95;
    static constexpr size_t tile_size = 16;

    constexpr static void(*alloc_double)(size_t, double**, double**) = alloc_cpu;
    constexpr static void(*copy_h2d)(size_t, double*, double*) = copy_h2d_cpu;
    constexpr static void(*copy_d2h)(size_t, double*, double*) = copy_d2h_cpu;
    constexpr static void(*free_double)(double*, double*) = free_cpu;

    using Shmem      = RAJA::LocalArray<double, RAJA::PERM_IJ, RAJA::SizeList<tile_size, tile_size>>;
    using ThreadPriv = RAJA::LocalArray<double, RAJA::PERM_IJ, RAJA::SizeList<tile_size, tile_size>>;

    using shmem_Lambda0 = RAJA::statement::Lambda<0, RAJA::Offsets<0, 2>, RAJA::Params<2>>;
    using shmem_Lambda1 = RAJA::statement::Lambda<1, RAJA::Segs<0, 1>, RAJA::Offsets<0, 1>, RAJA::Params<0>>;
    using shmem_Lambda2 = RAJA::statement::Lambda<2, RAJA::Segs<1, 2>, RAJA::Offsets<1, 2>, RAJA::Params<1>>;
    using shmem_Lambda3 = RAJA::statement::Lambda<3, RAJA::Offsets<0, 1, 2>, RAJA::Params<0, 1, 2>>;
    using shmem_Lambda4 = RAJA::statement::Lambda<4, RAJA::Segs<0, 2>, RAJA::Offsets<0, 2>, RAJA::Params<2>>;

    // Segments:
    // 0: N
    // 1: M
    // 2: P

    using exec_policy =
        RAJA::KernelPolicy<
          //Initalize thread private value
          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<2,1,0>,

            // Tile of N and P (the result matrix C)
            RAJA::statement::Tile<0, RAJA::tile_fixed<tile_size>, RAJA::seq_exec,
              RAJA::statement::Tile<2, RAJA::tile_fixed<tile_size>, RAJA::seq_exec,

               // zero out shmem tile of C
               RAJA::statement::For<2, RAJA::seq_exec,
                  RAJA::statement::For<0, RAJA::seq_exec,
                  shmem_Lambda0 > >,

                // Slide window across matrix: Tile in M
                RAJA::statement::Tile<1, RAJA::tile_fixed<tile_size>, RAJA::seq_exec,

                   // Load tile of A into shmem
                   RAJA::statement::For<1, RAJA::seq_exec,
                     RAJA::statement::For<0, RAJA::seq_exec,
                     shmem_Lambda1
                    >
                   >,

                   // Load tile of B into shmem
                   RAJA::statement::For<2, RAJA::seq_exec,
                     RAJA::statement::For<1, RAJA::seq_exec,
                     shmem_Lambda2
                    >
                   >,

                   //Partial multiplication
                   RAJA::statement::For<2, RAJA::seq_exec,
                     RAJA::statement::For<1, RAJA::seq_exec,
                       RAJA::statement::For<0, RAJA::seq_exec,
                       shmem_Lambda3
                       >
                     >
                   >
                >, //sliding window

                //Write memory out to global matrix
                RAJA::statement::For<2, RAJA::seq_exec,
                  RAJA::statement::For<0, RAJA::seq_exec,
                  shmem_Lambda4 > >
             >
            >
           > //Create shared memory
          >;
};

using MatMultiplyTypes = ::testing::Types<Policy_MatMultiply_cpu>;

INSTANTIATE_TYPED_TEST_SUITE_P(Seq, MatMultiply, MatMultiplyTypes);
