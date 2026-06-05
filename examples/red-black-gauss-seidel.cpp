//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <iostream>
#include <cmath>

#include "RAJA/RAJA.hpp"

#include "camp/resource.hpp"

/*
 * Gauss-Seidel with Red-Black Ordering Example
 *
 * ----[Details]--------------------
 * This example is an extension of Example 3.
 * In particular we maintain the five point stencil
 * to discretize the boundary value problem
 *
 * U_xx + U_yy = f on [0,1] x [0,1]
 *
 * on a structured grid. The right-hand side is
 * chosen to be f = 2*x*(y-1)*(y-2*x+x*y+2)*exp(x-y).
 *
 * Rather than computing values inside the domain with
 * the Jacobi method, a Gauss-Seidel method with red-black
 * ordering is now used.
 *
 * The scheme is implemented by treating the grid as
 * a checker board and storing the indices of red and
 * black cells in RAJA list segments. The segments are
 * then stored in a RAJA typed index set.
 *
 * ----[RAJA Concepts]---------------
 * - Forall loop
 * - RAJA Reduction
 * - RAJA::omp_collapse_nowait_exec
 * - RAJA::ListSegment
 * - RAJA::TypedIndexSet
 */

/*
 * Struct to hold grid info
 * o - Origin in a cartesian dimension
 * h - Spacing between grid points
 * n - Number of grid points
 */
struct grid_s {
  double o, h;
  int n;
};

/*
 * ----[Functions]---------
 * solution      - Function for the analytic solution
 * computeErr    - Displays the maximum error in the solution
 * gsColorPolicy - Generates the custom index set for this example
*/
double solution(double x, double y);
void computeErr(double *I, grid_s grid);
RAJA::TypedIndexSet<RAJA::ListSegment> 
  gsColorPolicy(int N, camp::resources::Resource res);

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout<<"Red-Black Gauss-Seidel Example"<<std::endl;

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

  int N = 100;
  int NN = (N + 2) * (N + 2);
  int maxIter = 100000;

  double resI2;
  int iteration;

  grid_s gridx;
  gridx.o = 0.0;
  gridx.h = 1.0 / (N + 1.0);
  gridx.n = N + 2;

  camp::resources::Resource resource{camp::resources::Host()};

  double *I = resource.allocate<double>(NN);

  memset(I, 0, NN * sizeof(double));

  RAJA::TypedIndexSet<RAJA::ListSegment> colorSet = gsColorPolicy(N, resource);

  memset(I, 0, NN * sizeof(double));

#if defined(RAJA_ENABLE_OPENMP)
  using colorPolicy =
      RAJA::ExecPolicy<RAJA::seq_segit, RAJA::omp_parallel_for_exec>;
#else
  using colorPolicy = RAJA::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec>;
#endif

  resI2 = 1;
  iteration = 0;
  while (resI2 > tol * tol) {

#if defined(RAJA_ENABLE_OPENMP)
    RAJA::ReduceSum<RAJA::omp_reduce, double> RAJA_resI2(0.0);
#else
    RAJA::ReduceSum<RAJA::seq_reduce, double> RAJA_resI2(0.0);
#endif

    //
    // Gauss-Seidel Iteration
    //
    RAJA::forall<colorPolicy>(colorSet, 
      [=](RAJA::Index_type id) {
        
      //
      // Compute x,y grid index
      //
      int m = id % (N + 2);
      int n = id / (N + 2);

      double x = gridx.o + m * gridx.h;
      double y = gridx.o + n * gridx.h;

      double f = gridx.h * gridx.h * 
                 (2 * x * (y - 1) * (y - 2 * x + x * y + 2) * exp(x - y));

      double newI = -0.25 * (f - I[id - N - 2] - I[id + N + 2] -
                                 I[id - 1] - I[id + 1]);

      double oldI = I[id];
      RAJA_resI2 += (newI - oldI) * (newI - oldI);
      I[id] = newI;

    });
    resI2 = RAJA_resI2;

    if (iteration > maxIter) {
      std::cout<<"Gauss-Seidel maxed out on iterations"<<std::endl;
      break;
    }

    iteration++;
  }
  computeErr(I, gridx);
  printf("No of iterations: %d \n \n", iteration);

  resource.deallocate(I);

  return 0;
}

//
//  This function will loop over the red and black cells of a grid
//  and store the index in a buffer. The buffers will then be used
//  to generate RAJA ListSegments and populate a RAJA Static Index
//  Set.

RAJA::TypedIndexSet<RAJA::ListSegment> 
  gsColorPolicy(int N, camp::resources::Resource res)
{
  RAJA::TypedIndexSet<RAJA::ListSegment> colorSet;

  int redN = static_cast<int>( std::ceil( static_cast<double>(N * N / 2) ) );
  int blkN = static_cast<int>( std::floor( static_cast<double>(N * N / 2) ) );
  RAJA::Index_type *Red = new RAJA::Index_type[redN];
  RAJA::Index_type *Blk = new RAJA::Index_type[blkN];

  int ib = 0;
  int ir = 0;

  bool isRed = true;

  for (int n = 1; n <= N; ++n) {
    
    for (int m = 1; m <= N; ++m) {
      
      RAJA::Index_type id = n * (N + 2) + m;
      if (isRed) {
        Red[ib] = id;
        ib++;
      } else {
        Blk[ir] = id;
        ir++;
      }
      isRed = !isRed;
    }

  }

  // Create Index
  colorSet.push_back(RAJA::ListSegment(Blk, blkN, res));
  colorSet.push_back(RAJA::ListSegment(Red, redN, res));
  delete[] Blk;
  delete[] Red;

  return colorSet;
}


//
//  Function for the anlytic solution
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

  RAJA::RangeSegment fdBounds(0, grid.n);
  RAJA::ReduceMax<RAJA::seq_reduce, double> tMax(-1.0);

  using errPolicy = RAJA::KernelPolicy<
    RAJA::statement::For<1, RAJA::seq_exec,
    RAJA::statement::For<0, RAJA::seq_exec, RAJA::statement::Lambda<0>> > >;

  RAJA::kernel<errPolicy>(RAJA::make_tuple(fdBounds,fdBounds),
                       [=] (RAJA::Index_type tx, RAJA::Index_type ty) {
    
      int id = tx + grid.n * ty;
      double x = grid.o + tx * grid.h;
      double y = grid.o + ty * grid.h;
      double myErr = std::abs(I[id] - solution(x, y));
      tMax.max(myErr);

    });

  double l2err = tMax;
  printf("Max error = %lg, h = %f \n", l2err, grid.h);
}
