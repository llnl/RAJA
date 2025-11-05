//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <iostream>
#include <cmath>

#include "RAJA/RAJA.hpp"

#include "memoryManager.hpp"

/*
 * Jacobi Example
 *
 * ----[Details]--------------------
 * This code uses a five point finite difference stencil
 * to discretize the following boundary value problem
 *
 * U_xx + U_yy = f on [0,1] x [0,1].
 *
 * The right-hand side is chosen to be
 * f = 2*x*(y-1)*(y-2*x+x*y+2)*exp(x-y).
 *
 * A structured grid is used to discretize the domain
 * [0,1] x [0,1]. Values inside the domain are computed
 * using the Jacobi method to solve the associated
 * linear system. The scheme is invoked until the l_2
 * difference of subsequent iterations is below a
 * tolerance.
 *
 * The scheme is implemented by allocating two arrays
 * (I, Iold) and initialized to zero. The first set of
 * nested for loops apply an iteration of the Jacobi
 * scheme. The scheme is only applied to the interior
 * nodes. 
 *
 * The second set of nested for loops is used to
 * update Iold and compute the l_2 norm of the
 * difference of the iterates.
 *
 * Computing the l_2 norm requires a reduction operation.
 * To simplify the reduction procedure, the RAJA API
 * introduces thread safe variables.
 *
 * ----[RAJA Concepts]---------------
 * - Forall::nested loop
 * - RAJA Reduction
 * 
 */


/*
 *  ----[Constant Values]-----
 * CUDA_BLOCK_SIZE_X - Number of threads in the
 *                     x-dimension of a cuda thread block
 *
 * CUDA_BLOCK_SIZE_Y - Number of threads in the
 *                     y-dimension of a cuda thread block
 * 
 * CUDA_BLOCK_SIZE   - Number of threads per threads block
*/
#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
#endif

#if defined(RAJA_ENABLE_HIP)
const int HIP_BLOCK_SIZE = 256;
#endif

//
//  Struct to hold grid info
//  o - Origin in a cartesian dimension
//  h - Spacing between grid points
//  n - Number of grid points
//
struct grid_s {
  double o, h;
  int n;
};

// 
// ----[Functions]---------
// solution   - Function for the analytic solution
// computeErr - Displays the maximum error in the solution
//
double solution(double x, double y);
void computeErr(double *I, grid_s grid);

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout<<"Jacobi Example"<<std::endl;

  /*
   * ----[Solver Parameters]------------
   * tol       - Method terminates once the norm is less than tol
   * N         - Number of unknown gridpoints per cartesian dimension
   * NN        - Total number of gridpoints on the grid
   * maxIter   - Maximum number of iterations to be taken
   *
   * resI2     - Residual
   * iteration - Iteration number
   * grid_s    - Struct with grid information for a cartesian dimension
  */
  double tol = 1e-10;

  int N = 50;
  int NN = (N + 2) * (N + 2);
  int maxIter = 100000;

  double resI2;
  int iteration;

  grid_s gridx;
  gridx.o = 0.0;
  gridx.h = 1.0 / (N + 1.0);
  gridx.n = N + 2;

  //
  //I, Iold - Holds iterates of Jacobi method
  //
  double *I = memoryManager::allocate<double>(NN);
  double *Iold = memoryManager::allocate<double>(NN);


  memset(I, 0, NN * sizeof(double));
  memset(Iold, 0, NN * sizeof(double));


  printf("Standard  C++ Loop \n");
  resI2 = 1;
  iteration = 0;

  while (resI2 > tol * tol) {

    //
    // Jacobi Iteration
    //
    for (int n = 1; n <= N; ++n) {
      for (int m = 1; m <= N; ++m) {

        double x = gridx.o + m * gridx.h;
        double y = gridx.o + n * gridx.h;

        double f = gridx.h * gridx.h
                   * (2 * x * (y - 1) * (y - 2 * x + x * y + 2) * exp(x - y));

        int id = n * (N + 2) + m;
        I[id] = 0.25 * (-f + Iold[id - N - 2] + Iold[id + N + 2] + Iold[id - 1]
                           + Iold[id + 1]);
      }
    }

    //
    // Compute residual and update Iold
    //
    resI2 = 0.0;
    for (int k = 0; k < NN; k++) {
      resI2 += (I[k] - Iold[k]) * (I[k] - Iold[k]);
      Iold[k] = I[k];
    }

    if (iteration > maxIter) {
      printf("Standard C++ Loop - Maxed out on iterations \n");
      exit(-1);
    }

    iteration++;
  }
  computeErr(I, gridx);
  printf("No of iterations: %d \n \n", iteration);


  //
  // RAJA loop calls may be shortened by predefining policies
  //
  RAJA::RangeSegment gridRange(0, NN);
  RAJA::RangeSegment jacobiRange(1, (N + 1));

  using jacobiSeqNestedPolicy = RAJA::KernelPolicy<
  RAJA::statement::For<1, RAJA::seq_exec,
    RAJA::statement::For<0, RAJA::seq_exec, RAJA::statement::Lambda<0>> > >;

  printf("RAJA: Sequential Policy - Nested ForallN \n");
  resI2 = 1;
  iteration = 0;
  memset(I, 0, NN * sizeof(double));
  memset(Iold, 0, NN * sizeof(double));

  /*
   *  Sequential Jacobi Iteration. 
   *
   *  Note that a RAJA ReduceSum object is used to accumulate the sum
   *  for the residual. Since the loop is run sequentially, this is 
   *  not strictly necessary. It is done here for consistency and 
   *  comparison with other RAJA variants in this example.
   */  
  while (resI2 > tol * tol) {

    RAJA::kernel<jacobiSeqNestedPolicy>(RAJA::make_tuple(jacobiRange,jacobiRange),
                         [=] (RAJA::Index_type m, RAJA::Index_type n) {
                         
          double x = gridx.o + m * gridx.h;
          double y = gridx.o + n * gridx.h;

          double f = gridx.h * gridx.h
                     * (2 * x * (y - 1) * (y - 2 * x + x * y + 2) * exp(x - y));

          int id = n * (N + 2) + m;
          I[id] =
               0.25 * (-f + Iold[id - N - 2] + Iold[id + N + 2] + Iold[id - 1]
                          + Iold[id + 1]);
        });

    RAJA::ReduceSum<RAJA::seq_reduce, double> RAJA_resI2(0.0);
    RAJA::forall<RAJA::seq_exec>(
      gridRange, [=](RAJA::Index_type k) {
      
        RAJA_resI2 += (I[k] - Iold[k]) * (I[k] - Iold[k]);          
        Iold[k] = I[k];

      });
    
    resI2 = RAJA_resI2;
    if (iteration > maxIter) {
      printf("Jacobi: Sequential - Maxed out on iterations! \n");
      exit(-1);
    }
    iteration++;
  }
  computeErr(I, gridx);
  printf("No of iterations: %d \n \n", iteration);
  
  
#if defined(RAJA_ENABLE_OPENMP)
  printf("RAJA: OpenMP Policy - Nested ForallN \n");
  resI2 = 1;
  iteration = 0;
  memset(I, 0, NN * sizeof(double));
  memset(Iold, 0, NN * sizeof(double));
  
  /*
   *  OpenMP parallel Jacobi Iteration. 
   *
   *  ----[RAJA Policies]-----------
   *  RAJA::omp_collapse_for_exec -
   *  introduced a nested region
   *
   *  Note that OpenMP RAJA ReduceSum object performs the reduction
   *  operation for the residual in a thread-safe manner.
   */
  
  using jacobiOmpNestedPolicy = RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::omp_parallel_for_exec,
        RAJA::statement::For<0, RAJA::seq_exec, RAJA::statement::Lambda<0> > > >;

  while (resI2 > tol * tol) {
    
    RAJA::kernel<jacobiOmpNestedPolicy>(RAJA::make_tuple(jacobiRange,jacobiRange),
                         [=] (RAJA::Index_type m, RAJA::Index_type n) {

                
      double x = gridx.o + m * gridx.h;
      double y = gridx.o + n * gridx.h;

      double f = gridx.h * gridx.h * 
                 (2 * x * (y - 1) * (y - 2 * x + x * y + 2) * exp(x - y));

      int id = n * (N + 2) + m;
      I[id] = 0.25 * (-f + Iold[id - N - 2] + Iold[id + N + 2] + 
                           Iold[id - 1] + Iold[id + 1]);              
    });


    RAJA::ReduceSum<RAJA::omp_reduce, double> RAJA_resI2(0.0);

    RAJA::forall<RAJA::omp_parallel_for_exec>( gridRange, 
      [=](RAJA::Index_type k) {
      
      RAJA_resI2 += (I[k] - Iold[k]) * (I[k] - Iold[k]);                    
      Iold[k] = I[k];
        
    });
    
    resI2 = RAJA_resI2;
    if (iteration > maxIter) {
      printf("Jacobi: OpenMP - Maxed out on iterations! \n");
      exit(-1);
    }
    iteration++;
  }
  computeErr(I, gridx);
  printf("No of iterations: %d \n \n", iteration);
#endif


#if defined(RAJA_ENABLE_CUDA)
  /*
   *  CUDA Jacobi Iteration. 
   *
   *  ----[RAJA Policies]-----------
   *  RAJA::cuda_threadblock_y_exec, RAJA::cuda_threadblock_x_exec -
   *  define the mapping of loop iterations to GPU thread blocks
   *
   *  Note that CUDA RAJA ReduceSum object performs the reduction
   *  operation for the residual in a thread-safe manner on the GPU.
   */

  printf("RAJA: CUDA Policy - Nested ForallN \n");

  using jacobiCUDANestedPolicy = RAJA::KernelPolicy<
    RAJA::statement::CudaKernel<
      RAJA::statement::Tile<1, RAJA::tile_fixed<32>, RAJA::cuda_block_y_loop,
        RAJA::statement::Tile<0, RAJA::tile_fixed<32>, RAJA::cuda_block_x_loop,
          RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
            RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    > >;
  
  resI2 = 1;
  iteration = 0;
  memset(I, 0, NN * sizeof(double));
  memset(Iold, 0, NN * sizeof(double));

  while (resI2 > tol * tol) {

    //
    // Jacobi Iteration 
    //
    RAJA::kernel<jacobiCUDANestedPolicy>(
                         RAJA::make_tuple(jacobiRange,jacobiRange),
                         [=] RAJA_DEVICE  (RAJA::Index_type m, RAJA::Index_type n) {
                           
          double x = gridx.o + m * gridx.h;
          double y = gridx.o + n * gridx.h;

          double f = gridx.h * gridx.h
                     * (2 * x * (y - 1) * (y - 2 * x + x * y + 2) * exp(x - y));

          int id = n * (N + 2) + m;
          I[id] = 0.25 * (-f + Iold[id - N - 2] + Iold[id + N + 2] + Iold[id - 1]
                             + Iold[id + 1]);                            
        });

    //
    // Compute residual and update Iold
    //
    RAJA::ReduceSum<RAJA::cuda_reduce, double> RAJA_resI2(0.0);
    RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(
      gridRange, [=] RAJA_DEVICE (RAJA::Index_type k) {
      
          RAJA_resI2 += (I[k] - Iold[k]) * (I[k] - Iold[k]);
          Iold[k] = I[k];

      });

    resI2 = RAJA_resI2;

    if (iteration > maxIter) {
      printf("RAJA: CUDA - Maxed out on iterations! \n");
      exit(-1);
    }
    iteration++;
  }
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  computeErr(I, gridx);
  printf("No of iterations: %d \n \n", iteration);
#endif

#if defined(RAJA_ENABLE_HIP)
  /*
   *  HIP Jacobi Iteration.
   *
   *  ----[RAJA Policies]-----------
   *  RAJA::cuda_threadblock_y_exec, RAJA::cuda_threadblock_x_exec -
   *  define the mapping of loop iterations to GPU thread blocks
   *
   *  Note that HIP RAJA ReduceSum object performs the reduction
   *  operation for the residual in a thread-safe manner on the GPU.
   */

  printf("RAJA: HIP Policy - Nested ForallN \n");

  using jacobiHIPNestedPolicy = RAJA::KernelPolicy<
    RAJA::statement::HipKernel<
      RAJA::statement::Tile<1, RAJA::tile_fixed<32>, RAJA::hip_block_y_loop,
        RAJA::statement::Tile<0, RAJA::tile_fixed<32>, RAJA::hip_block_x_loop,
          RAJA::statement::For<1, RAJA::hip_thread_y_direct,
            RAJA::statement::For<0, RAJA::hip_thread_x_direct,
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    > >;

  resI2 = 1;
  iteration = 0;
  memset(I, 0, NN * sizeof(double));
  memset(Iold, 0, NN * sizeof(double));

  double *d_I    = memoryManager::allocate_gpu<double>(NN);
  double *d_Iold = memoryManager::allocate_gpu<double>(NN);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_I, I, NN * sizeof(double), hipMemcpyHostToDevice);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_Iold, Iold, NN * sizeof(double), hipMemcpyHostToDevice);

  while (resI2 > tol * tol) {

    //
    // Jacobi Iteration
    //
    RAJA::kernel<jacobiHIPNestedPolicy>(
                         RAJA::make_tuple(jacobiRange,jacobiRange),
                         [=] RAJA_DEVICE  (RAJA::Index_type m, RAJA::Index_type n) {

          double x = gridx.o + m * gridx.h;
          double y = gridx.o + n * gridx.h;

          double f = gridx.h * gridx.h
                     * (2 * x * (y - 1) * (y - 2 * x + x * y + 2) * exp(x - y));

          int id = n * (N + 2) + m;
          d_I[id] = 0.25 * (-f + d_Iold[id - N - 2] + d_Iold[id + N + 2] + d_Iold[id - 1]
                             + d_Iold[id + 1]);
        });

    //
    // Compute residual and update Iold
    //
    RAJA::ReduceSum<RAJA::hip_reduce, double> RAJA_resI2(0.0);
    RAJA::forall<RAJA::hip_exec<HIP_BLOCK_SIZE>>(
      gridRange, [=] RAJA_DEVICE (RAJA::Index_type k) {

          RAJA_resI2 += (d_I[k] - d_Iold[k]) * (d_I[k] - d_Iold[k]);
          d_Iold[k] = d_I[k];

      });

    resI2 = RAJA_resI2;

    if (iteration > maxIter) {
      printf("RAJA: HIP - Maxed out on iterations! \n");
      exit(-1);
    }
    iteration++;
  }
  CAMP_HIP_API_INVOKE_AND_CHECK(hipDeviceSynchronize);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, I, d_I, NN * sizeof(double), hipMemcpyDeviceToHost);
  computeErr(I, gridx);
  printf("No of iterations: %d \n \n", iteration);

  memoryManager::deallocate_gpu(d_I);
  memoryManager::deallocate_gpu(d_Iold);
#endif

  memoryManager::deallocate(I);
  memoryManager::deallocate(Iold);
  

  return 0;
}

//
// Function for the anlytic solution
//
double solution(double x, double y)
{
  return x * y * exp(x - y) * (1 - x) * (1 - y);
}

//
// Error is computed via ||I_{approx}(:) - U_{analytic}(:)||_{inf}
//
void computeErr(double *I, grid_s grid)
{

  RAJA::RangeSegment gridRange(0, grid.n);
  RAJA::ReduceMax<RAJA::seq_reduce, double> tMax(-1.0);

  using jacobiSeqNestedPolicy = RAJA::KernelPolicy<
    RAJA::statement::For<1, RAJA::seq_exec,
      RAJA::statement::For<0, RAJA::seq_exec, RAJA::statement::Lambda<0> > > >;

  RAJA::kernel<jacobiSeqNestedPolicy>(RAJA::make_tuple(gridRange,gridRange),
                       [=] (RAJA::Index_type ty, RAJA::Index_type tx ) {

      int id = tx + grid.n * ty;
      double x = grid.o + tx * grid.h;
      double y = grid.o + ty * grid.h;
      double myErr = std::abs(I[id] - solution(x, y));
      tMax.max(myErr);
    });

  double l2err = tMax;
  printf("Max error = %lg, h = %f \n", l2err, grid.h);
}
