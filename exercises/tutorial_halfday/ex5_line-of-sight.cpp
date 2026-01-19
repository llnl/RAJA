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
#include <cstring>
#include <cmath>
#include <algorithm>

#include "RAJA/RAJA.hpp"

#include "memoryManager.hpp"

/*
 *  EXERCISE #5: The line-of-sight problem.
 *
 *  In this exercise, you will use use RAJA scan operations to solve
 *  the 'line-of-sight'' problem, which is stated as:
 *
 *  Given an observation point X on a terrain map, and a set of points
 *  {Y0, Y1, Y2, ...} along a ray starting at X, find which points on the
 *  terrain at Y0, Y1, etc. are visible from the point at X. A point is 
 *  visible from the point at X if and only if there is no other point on the 
 *  terrain that blocks its view from the point at X. More precisely, 
 *  a point on the terrain at Y is visible from the point at X if and only if 
 *  no other point on the terrain between X and Y has a greater vertical angle 
 *  from the point at X than the point at Y. So although a point at Y may
 *  be at a higher altitude than all other points on the terrain between Y 
 *  and X, the point at Y may not be visible from the point at X.
 *
 *  Let 'altX' be the altidue at point X. Suppose we have a vector 'dist' 
 *  such that dist[i] is the horizontal distance between X and Yi, and a 
 *  vector 'alt' such that alt[i] is the altitude at point Yi. To solve 
 *  the line of sight problem, we compute an angle vector 'ang', where 
 *  ang[i] = arctan( (alt[i] - altX)/(dist[i]). Next, we perform a "max"
 *  scan on the vector 'ang' to form the vector 'ang_max'. Then, the point 
 *  at Yi is visible from the point at X if ang[i] >= ang_max[i]. Otherwise,
 *  the point at Yi is not visible.
 *
 *  This file contains a C-style sequential implementation of the solution to
 *  the line-of-sight problem. Where indicated by comments, you will fill in 
 *  sequential and OpenMP versions of the algorithm using a RAJA scan operation
 *  to compute the 'ang_max' vector and a RAJA forall method to determine which
 *  points are/are not visible. If you have access to an NVIDIA GPU and a CUDA 
 *  compiler, fill in the RAJA CUDA version of the algorithm also. 
 *
 *  RAJA features you will use:
 *    - inclusive scan operations with 'max' operator
 *    - `forall` loop iteration template method
 *    - Index range segment
 *
 *  If CUDA is enabled, CUDA unified memory is used.
 */

/*
  CUDA_BLOCK_SIZE - specifies the number of threads in a CUDA thread block

                    Uncomment to use when filling in exercises.

#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
#endif
*/

//
// Functions to check results and print arrays.
//
// checkResult() returns number of visible points.
//
int checkResult(int* visible, int* visibleref, int len);

template <typename T>
void printArray(T* v, int len);


int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nExercise #5: The line-of-sight problem...\n";

  //
  // Define array bounds and initialize distance and altitude arrays.
  //
  int N = 100;
  double alt_max = 100.0;

  double* dist = memoryManager::allocate<double>(N);
  double* alt = memoryManager::allocate<double>(N);
  double* ang = memoryManager::allocate<double>(N);
  double* ang_max = memoryManager::allocate<double>(N);
  int* visible = memoryManager::allocate<int>(N);
  int* visible_ref = memoryManager::allocate<int>(N);

  for (int i = 0; i < N; ++i) { 
    dist[i] = static_cast<double>(i+1);
    double alt_fact = alt_max * ( (i+1) % 5 == 0 ? i*10 : i+1 );
    alt[i] = alt_fact * 
             static_cast<double>( rand() ) / static_cast<double>( RAND_MAX );
  }

  //
  // Set angle array
  // 
  for (int i = 0; i < N; ++i) { 
    ang[i] = atan2( alt[i], dist[i] );       // set angle in radians
  }


//----------------------------------------------------------------------------//
// C-style sequential variant establishes reference solution to compare with.
//----------------------------------------------------------------------------//

  std::cout << "\n\n Running C-style sequential line-of-sight algorithm...\n";

  std::memset(visible_ref, 0, N * sizeof(int));

  ang_max[0] = ang[0];
  for (int i = 1; i < N; ++i) {
      ang_max[i] = std::max(ang[i], ang_max[i-1]);
  }

  int num_visible = 0;

  for (int i = 0; i < N; ++i) {
     if ( ang[i] >= ang_max[i] ) {
        visible_ref[i] = 1;
        num_visible++;
     } else {
        visible_ref[i] = 0;
     }
  }

  std::cout << "\n\t num visible points = " << num_visible << "\n\n";
//printArray(visible_ref, N);


//----------------------------------------------------------------------------//
// RAJA sequential variant
//----------------------------------------------------------------------------//

  std::cout << "\n\n Running RAJA sequential line-of-sight algorithm...\n";

  std::memset(ang_max, 0, N * sizeof(double));
  std::memset(visible, 0, N * sizeof(int));
  num_visible = 0;

  ///
  /// TODO...
  ///
  /// EXERCISE: Implement the line-of-sight algorithm using RAJA constructs.
  ///           First, use a 'max' RAJA::inclusive_scan on the angle vector 
  ///           with RAJA::seq_exec execution policy. Then, use a RAJA::forall
  ///           template with the same execution policy to determine which
  ///           points are visible.
  ///


  num_visible = checkResult(visible, visible_ref, N);
  std::cout << "\n\t num visible points = " << num_visible << "\n\n";
//printArray(visible, N);


//----------------------------------------------------------------------------//
// RAJA OpenMP multithreading variant
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::cout << "\n\n Running RAJA OpenMP line-of-sight algorithm...\n";

  std::memset(ang_max, 0, N * sizeof(double));
  std::memset(visible, 0, N * sizeof(int));
  num_visible = 0;

  ///
  /// TODO...
  ///
  /// EXERCISE: Implement the line-of-sight algorithm using RAJA constructs.
  ///           First, use a 'max' RAJA::inclusive_scan on the angle vector 
  ///           with RAJA::omp_parallel_for_exec execution policy. Then, use 
  ///           a RAJA::forall template with the same execution policy to 
  ///           determine which points are visible.
  ///


  num_visible = checkResult(visible, visible_ref, N);
  std::cout << "\n\t num visible points = " << num_visible << "\n\n";
//printArray(visible, N);

#endif


//----------------------------------------------------------------------------//
// RAJA CUDA variant
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  std::cout << "\n\n Running RAJA CUDA line-of-sight algorithm...\n";

  std::memset(ang_max, 0, N * sizeof(double));
  std::memset(visible, 0, N * sizeof(int));
  num_visible = 0;

  ///
  /// TODO...
  ///
  /// EXERCISE: Implement the line-of-sight algorithm using RAJA constructs.
  ///           First, use a 'max' RAJA::inclusive_scan on the angle vector 
  ///           with RAJA::cuda_exec execution policy. Then, use a 
  ///           RAJA::forall template with the same execution policy to 
  ///           determine which points are visible.
  ///


  num_visible = checkResult(visible, visible_ref, N);
  std::cout << "\n\t num visible points = " << num_visible << "\n\n";
//printArray(visible, N);

#endif

  //
  // Clean up.
  //

  memoryManager::deallocate(dist);
  memoryManager::deallocate(alt);
  memoryManager::deallocate(ang);
  memoryManager::deallocate(ang_max);
  memoryManager::deallocate(visible);
  memoryManager::deallocate(visible_ref);

  std::cout << "\n DONE!...\n";

  return 0;
}

//
// Function to check result and report P/F.
//
int checkResult(int* visible, int* visible_ref, int len)
{
  int num_visible = 0;

  bool correct = true;
  for (int i = 0; i < len; i++) {
    if ( correct && visible[i] != visible_ref[i] ) { correct = false; }
    num_visible += visible[i];
  }
  if ( correct ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }

  return num_visible;
}

//
// Function to print array.
//
template <typename T>
void printArray(T* v, int len)
{
  std::cout << std::endl;
  for (int i = 0; i < len; i++) {
    std::cout << "v[" << i << "] = " << v[i] << std::endl;
  }
  std::cout << std::endl;
}
