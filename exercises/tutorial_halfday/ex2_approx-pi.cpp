//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>
#include <iomanip>

#include "RAJA/RAJA.hpp"

/*
 *  EXERCISE #2: Approximate pi using a Riemann sum
 *
 *  In this exercise, you will apprimate pi using the formula
 * 
 *    pi/4 = atan(1) = integral (1/1+x^2) dx, where integral is over the
 *    interval [0, 1].
 *
 *  This file contains sequential and OpenMP variants of the vector addition
 *  using C-style for-loops. You will fill in RAJA versions of these variants,
 *  plus a RAJA CUDA version if you have access to an NVIDIA GPU and a CUDA
 *  compiler, in empty code sections indicated by comments.
 *
 *  RAJA features you will use:
 *    - `forall` loop iteration template method
 *    - Index range segment
 *    - Sum reduction
 *    - Execution and reduction policies 
 */

/*
  CUDA_BLOCK_SIZE - specifies the number of threads in a CUDA thread block

                    Uncomment to use when filling in exercises.

#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
#endif
*/

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nExercise #2: Approximate pi using a Riemann sum...\n";

//
// Define number of subintervals (N) and size of each subinterval (dx) used in
// Riemann integral sum to approximate pi.
//
  const int N = 512 * 512;
  const double dx = 1.0 / double(N); 

// Set precision for printing pi
  int prec = 16;


//----------------------------------------------------------------------------//
// C-style sequential variant establishes reference solution to compare with.
//----------------------------------------------------------------------------//

  std::cout << "\n Running C-style sequential pi approximation...\n";
 
  double c_pi = 0.0;

  for (int i = 0; i < N; ++i) {
      double x = (double(i) + 0.5) * dx;
      c_pi += dx / (1.0 + x * x);
  }
  c_pi *= 4.0;

  std::cout << "\tpi = " << std::setprecision(prec) 
            << c_pi << std::endl;


//----------------------------------------------------------------------------//
// RAJA sequential variant.
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA sequential pi approximation...\n";

  ///
  /// TODO...
  ///
  /// EXERCISE: Implement the pi approximation kernel using a RAJA::forall
  ///           method with RAJA::seq_exec execution policy type and a 
  ///           RAJA::ReduceSum object with RAJA::seq_reduce policy type
  ///           to accumulate the sum.
  ///
  /// NOTE: We've done this one for you to help you get started...
  ///

  using EXEC_POL1   = RAJA::seq_exec;
  using REDUCE_POL1 = RAJA::seq_reduce;

  RAJA::ReduceSum< REDUCE_POL1, double > seq_pi(0.0);

  RAJA::forall< EXEC_POL1 >(RAJA::RangeSegment(0, N), [=](int i) {
      double x = (double(i) + 0.5) * dx;
      seq_pi += dx / (1.0 + x * x);
  });
  double seq_pi_val = seq_pi.get() * 4.0; 

  std::cout << "\tpi = " << std::setprecision(prec) 
            << seq_pi_val << std::endl;


//----------------------------------------------------------------------------//
// C-style OpenMP multithreading variant.
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::cout << "\n Running C-style OpenMP vector addition...\n";

  double c_pi_omp = 0.0;

  #pragma omp parallel for reduction(+:c_pi_omp)
  for (int i = 0; i < N; ++i) {
      double x = (double(i) + 0.5) * dx;
      c_pi_omp += dx / (1.0 + x * x);
  }
  c_pi_omp *= 4.0;

  std::cout << "\tpi = " << std::setprecision(prec)
            << c_pi_omp << std::endl;

#endif


//----------------------------------------------------------------------------//
// RAJA OpenMP multithreading variant.
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::cout << "\n Running RAJA OpenMP pi approximation...\n";

  ///
  /// TODO...
  ///
  /// EXERCISE: Implement the pi approximation kernel using a RAJA::forall
  ///           method with RAJA::omp_parallel_for_exec execution policy type 
  ///           and a RAJA::ReduceSum object with RAJA::omp_reduce policy type
  ///           to accumulate the sum.
  /// 


  double omp_pi_val = 0.0;

  std::cout << "\tpi = " << std::setprecision(prec)
            << omp_pi_val << std::endl;

#endif


//----------------------------------------------------------------------------//
// RAJA CUDA variant.
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  std::cout << "\n Running RAJA CUDA pi approximation...\n";

  ///
  /// TODO...
  ///
  /// EXERCISE: Implement the pi approximation kernel using a RAJA::forall
  ///           method with RAJA::cuda_exec execution policy type and a 
  ///           RAJA::ReduceSum object with RAJA::cuda_reduce policy type
  ///           to accumulate the sum.
  /// 


  double cuda_pi_val = 0.0;

  std::cout << "\tpi = " << std::setprecision(prec)
            << cuda_pi_val << std::endl;

#endif

  std::cout << "\n DONE!...\n";

  return 0;
}
