//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cmath>
#include <cstdlib>
#include <iostream>
#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *   Time-Domain Finite Difference 
 *   Acoustic Wave Equation Solver
 *
 * ------[Details]----------------------
 * This example highlights how to construct a single
 * kernel capable of being executed with different RAJA policies.
 *
 * Here we solve the acoustic wave equation
 * P_tt = cc*(P_xx + P_yy) via finite differences.
 *
 * The scheme uses a second order central difference discretization
 * for time and a fourth order central difference discretization for space.
 * Periodic boundary conditions are assumed on the grid [-1,1] x [-1, 1].
 * 
 * NOTE: The x and y dimensions are discretized identically.
 * ----[RAJA Concepts]-------------------
 * - RAJA kernels are portable and a single implemenation can run
 * on various platforms
 *
 * RAJA MaxReduction - RAJA's implementation for computing a maximum value
 *    (MinReduction computes the min)
*/

//
//  ---[Constant Values]-------
//  sr - Radius of the finite difference stencil
//  PI - Value of pi
//

const int sr = 2;
const double PI = 3.14159265359;

//
//  ----[Struct to hold grid info]-----
//  o - Origin in a cartesian dimension
//  h - Spacing between grid points
//  n - Number of grid points
//
struct grid_s {
  double ox, dx;
  int nx;
};


//
//  ----[Functions]------
//  wave       - Templated wave propagator
//  waveSol    - Function for the analytic solution of the equation
//  setIC      - Sets the intial value at two time levels (t0,t1)
//  computeErr - Displays the maximum error in the approximation
//

template <typename T, typename fdNestedPolicy>
void wave(T *P1, T *P2, RAJA::RangeSegment fdBounds, double ct, int nx);
double waveSol(double t, double x, double y);
void setIC(double *P1, double *P2, double t0, double t1, grid_s grid);
void computeErr(double *P, double tf, grid_s grid);

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout<<"Time-Domain Finite Difference Acoustic Wave Equation Solver"<<std::endl;
         
  //
  // Wave speed squared
  //
  double cc = 1. / 2.0;

  //
  //  Multiplier for spatial refinement
  //
  int factor = 8;

  //
  // Discretization of the domain.
  // The same discretization of the x-dimension wil be used for the y-dimension
  //
  grid_s grid;
  grid.ox = -1;
  grid.dx = 0.1250 / factor;
  grid.nx = 16 * factor;
  RAJA::RangeSegment fdBounds(0, grid.nx);

  //
  //Solution is propagated until time T
  //
  double T = 0.82;


  int entries = grid.nx * grid.nx;
  double *P1 = memoryManager::allocate<double>(entries);
  double *P2 = memoryManager::allocate<double>(entries);

  //
  //----[Time stepping parameters]----
  // dt - Step size
  // nt - Total number of time steps
  // ct - Merged coefficents
  //
  double dt, nt, time, ct;
  dt = 0.01 * (grid.dx / sqrt(cc));
  nt = ceil(T / dt);
  dt = T / nt;
  ct = (cc * dt * dt) / (grid.dx * grid.dx);

  //
  // Predefined policies
  //

  // Sequential policy
  using fdPolicy = RAJA::KernelPolicy<
    RAJA::statement::For<1, RAJA::seq_exec,
    RAJA::statement::For<0, RAJA::seq_exec, RAJA::statement::Lambda<0> > > >;

  // OpenMP policy
  //using fdPolicy = RAJA::KernelPolicy<
  //RAJA::statement::For<1, RAJA::omp_parallel_for_exec,
  //  RAJA::statement::For<0, RAJA::seq_exec, RAJA::statement::Lambda<0> > > >;

  // CUDA policy
  //using fdPolicy =
  //RAJA::KernelPolicy<
  //  RAJA::statement::CudaKernel<
  //      RAJA::statement::Tile<1, RAJA::tile_fixed<16>, RAJA::cuda_block_y_direct,
  //        RAJA::statement::Tile<0, RAJA::tile_fixed<16>, RAJA::cuda_block_x_direct,
  //          RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
  //            RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
  //              RAJA::statement::Lambda<0>
  //            >
  //          >
  //        >
  //      >
  //    >
  //  >;


  time = 0;
  setIC(P1, P2, (time - dt), time, grid);
  for (int k = 0; k < nt; ++k) {

    wave<double, fdPolicy>(P1, P2, fdBounds, ct, grid.nx);

    time += dt;

    double *Temp = P2;
    P2 = P1;
    P1 = Temp;
  }
#if defined(RAJA_ENABLE_CUDA)
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
#endif
  computeErr(P2, time, grid);
  printf("Evolved solution to time = %f \n", time);

  memoryManager::deallocate(P1);
  memoryManager::deallocate(P2);

  return 0;
}


//
//  Function for the analytic solution
//
double waveSol(double t, double x, double y)
{
  return cos(2. * PI * t) * sin(2. * PI * x) * sin(2. * PI * y);
}

//
//  Error is computed via ||P_{approx}(:) - P_{analytic}(:)||_{inf}
//
void computeErr(double *P, double tf, grid_s grid)
{

  RAJA::RangeSegment fdBounds(0, grid.nx);
  RAJA::ReduceMax<RAJA::seq_reduce, double> tMax(-1.0);

  using initialPolicy = RAJA::KernelPolicy<
  RAJA::statement::For<1, RAJA::seq_exec ,
    RAJA::statement::For<0, RAJA::seq_exec, RAJA::statement::Lambda<0> > > >;

  RAJA::kernel<initialPolicy>(RAJA::make_tuple(fdBounds,fdBounds),
                       [=] (RAJA::Index_type tx, RAJA::Index_type ty) {

      int id = tx + grid.nx * ty;
      double x = grid.ox + tx * grid.dx;
      double y = grid.ox + ty * grid.dx;
      double myErr = std::abs(P[id] - waveSol(tf, x, y));

      //
      // tMax.max() is used to store the maximum value
      //
      tMax.max(myErr);
    });

  double lInfErr = tMax;
  printf("Max Error = %lg, dx = %f \n", lInfErr, grid.dx);
}


//
// Function to set intial condition
//
void setIC(double *P1, double *P2, double t0, double t1, grid_s grid)
{

  RAJA::RangeSegment fdBounds(0, grid.nx);

  using initialPolicy = RAJA::KernelPolicy<
  RAJA::statement::For<1, RAJA::seq_exec,
    RAJA::statement::For<0, RAJA::seq_exec, RAJA::statement::Lambda<0>> > >;
  
  RAJA::kernel<initialPolicy>(RAJA::make_tuple(fdBounds,fdBounds),
                       [=] (RAJA::Index_type tx, RAJA::Index_type ty) {
                         
      int id = tx + ty * grid.nx;
      double x = grid.ox + tx * grid.dx;
      double y = grid.ox + ty * grid.dx;
      
      P1[id] = waveSol(t0, x, y);
      P2[id] = waveSol(t1, x, y);
    });
}



template <typename T, typename fdNestedPolicy>
void wave(T *P1, T *P2, RAJA::RangeSegment fdBounds, double ct, int nx)
{

  RAJA::kernel<fdNestedPolicy>(RAJA::make_tuple(fdBounds,fdBounds),
                       [=] RAJA_HOST_DEVICE (RAJA::Index_type tx, RAJA::Index_type ty) {
      //                  
      //Coefficients for fourth order stencil
      //
     double coeff[5] = { -1.0/12.0, 4.0/3.0, -5.0/2.0, 4.0/3.0, -1.0/12.0};

     const int id = tx + ty * nx;
     double P_old = P1[id];
     double P_curr = P2[id];

     //
     // Compute Laplacian
     //
     double lap = 0.0;

     for (auto r : RAJA::RangeSegment(-sr, sr + 1)) {
       const int xi = (tx + r + nx) % nx;
       const int idx = xi + nx * ty;
       lap += coeff[r + sr] * P2[idx];
  
       const int yi = (ty + r + nx) % nx;
       const int idy = tx + nx * yi;
       lap += coeff[r + sr] * P2[idy];
     }

     //
     // Store result
     //
     P1[id] = 2 * P_curr - P_old + ct * lap;

  });
}
