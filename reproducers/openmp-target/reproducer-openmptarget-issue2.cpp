//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "RAJA/RAJA.hpp"

/*
 *  Reproducer for...
 *
 */

//
// Functions for checking and printing arrays used for all cases
//
void checkResult(int* c, int* c_ref, int len); 
void printArray(int* v, int len);


#if defined(RAJA_ENABLE_TARGET_OPENMP)
//
// For OpenMP target variants
//
const size_t threads_per_team = 256;

template <typename T>
void initOpenMPDeviceData(T& dptr, const T hptr, int len, 
                          int did, int hid)
{
  omp_target_memcpy( dptr, hptr, 
                     len * sizeof(typename std::remove_pointer<T>::type),
                     0, 0, did, hid ); 
}

template <typename T>
void allocAndInitOpenMPDeviceData(T& dptr, const T hptr, int len,
                                  int did, int hid)
{
  dptr = static_cast<T>( omp_target_alloc(
                         len * sizeof(typename std::remove_pointer<T>::type), 
                         did) );

  initOpenMPDeviceData(dptr, hptr, len, did, hid);
}

template <typename T>
void getOpenMPDeviceData(T& hptr, const T dptr, int len, int hid, int did)
{
  omp_target_memcpy( hptr, dptr, 
                     len * sizeof(typename std::remove_pointer<T>::type),
                     0, 0, hid, did );
}

template <typename T>
void deallocOpenMPDeviceData(T& dptr, int did)
{
  omp_target_free( dptr, did );
  dptr = 0;
}
#endif


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

#if defined(RAJA_ENABLE_TARGET_OPENMP)
  // Pointers for device-allocated data
  int* dc;

  int hid = omp_get_initial_device();
  int did = omp_get_default_device();
#endif

///////////////////////////////////////////////////////////////////

  std::cout << "\n\n 3D matrix init example...\n";

//
// Define matrix extents
//
  constexpr int N = 10;
  constexpr int M = 100;
  constexpr int L = 100;
  constexpr int N_3D = N * M * L;

  // Host allocations for variant results
  int* c_ref = new int[N_3D];
  int* c_raja_omp = new int[N_3D];
  int* c_raja_omp_collapse2 = new int[N_3D];
  int* c_base_omptarget = new int[N_3D];
  int* c_raja_omptarget = new int[N_3D];
  int* c_raja_omptarget_collapse2 = new int[N_3D];

//----------------------------------------------------------------------------//
// C-style sequential variant establishes reference solution to compare with.
//----------------------------------------------------------------------------//

  std::cout << "\n1: C-style 2D matrix init...\n";

  std::memset(c_ref, 0, N_3D * sizeof(int));

  for (int n = 0; n < N; ++n) {
    for (int m = 0; m < M; ++m) {
      for (int l = 0; l < L; ++l) {
        c_ref[l + L*m + L*M*n] = l + L*m + L*M*n;
      }
    }
  }

//printArray(c_ref, N_3D);


//----------------------------------------------------------------------------//
// RAJA OpenMP variants (for comparing to RAJA OpenMP target)
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::cout << "\n2: RAJA OpenMP collapse 3D matrix init...\n";

  std::memset(c_raja_omp, 0, N_3D * sizeof(int));

  using EXEC_POL5 =
    RAJA::KernelPolicy<
      RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                RAJA::ArgList<2, 1, 0>,  // n, m, l 
        RAJA::statement::Lambda<0>
      >
    >;

  RAJA::kernel<EXEC_POL5>( 
    RAJA::make_tuple(RAJA::RangeSegment{0, L},
                     RAJA::RangeSegment{0, M},
                     RAJA::RangeSegment{0, N}),
    [=](int l, int m, int n) {  
      c_raja_omp[l + L*m + L*M*n] = l + L*m + L*M*n;
  });

  checkResult(c_raja_omp, c_ref, N_3D);
//printArray(c_raja_omp, N_3D);


//////////////////////////////////////////////

  std::cout << "\n3: RAJA OpenMP 3D matrix init (collapse 2)...\n";

  std::memset(c_raja_omp_collapse2, 0, N_3D * sizeof(int));

  using EXEC_POL6 =
    RAJA::KernelPolicy<
      RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                RAJA::ArgList<2, 1>,   // n, m
        RAJA::statement::For<0, RAJA::seq_exec,        // l
          RAJA::statement::Lambda<0>
        >
      >
    >;

  RAJA::kernel<EXEC_POL6>(
    RAJA::make_tuple(RAJA::RangeSegment{0, L},
                     RAJA::RangeSegment{0, M},
                     RAJA::RangeSegment{0, N}),
    [=](int l, int m, int n) {
      c_raja_omp_collapse2[l + L*m + L*M*n] = l + L*m + L*M*n;
  });

  checkResult(c_raja_omp_collapse2, c_ref, N_3D);
//printArray(c_raja_omp_collapse2, N_3D);

#endif

//----------------------------------------------------------------------------//
// OpenMP target variants
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_TARGET_OPENMP)

//----------------------------------------------------------------------------//
// Base OpenMP target variant
//----------------------------------------------------------------------------//

  std::cout << "\n4: Base OpenMP target 3D matrix init...\n";

  std::memset(c_base_omptarget, 0, N_3D * sizeof(int));
  allocAndInitOpenMPDeviceData(dc, c_base_omptarget, N_3D, did, hid);

  #pragma omp target is_device_ptr(dc) device( did )
  #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1) collapse(3)
  for (int n = 0; n < N; ++n) {
    for (int m = 0; m < M; ++m) {
      for (int l = 0; l < L; ++l) {
        dc[l + L*m + L*M*n] = l + L*m + L*M*n;
      }
    }
  }

  getOpenMPDeviceData(c_base_omptarget, dc, N_3D, hid, did);

  checkResult(c_base_omptarget, c_ref, N_3D);
//printArray(c_base_omptarget, N_3D);

//----------------------------------------------------------------------------//
// RAJA OpenMP target variant
//----------------------------------------------------------------------------//

  std::cout << "\n5: RAJA OpenMP target 3D matrix init...\n";

  std::memset(c_raja_omptarget, 0, N_3D * sizeof(int));
  initOpenMPDeviceData(dc, c_raja_omptarget, N_3D, did, hid);

  using KERNEL_POL5 =
    RAJA::KernelPolicy<
      RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                RAJA::ArgList<2, 1, 0>,  // n, m, l
        RAJA::statement::Lambda<0>
      >
    >;

  RAJA::kernel<KERNEL_POL5>(
    RAJA::make_tuple(RAJA::RangeSegment{0, L},
                     RAJA::RangeSegment{0, M},
                     RAJA::RangeSegment{0, N}),
    [=](int l, int m, int n) {
      dc[l + L*m + L*M*n] = l + L*m + L*M*n;
  });

  getOpenMPDeviceData(c_raja_omptarget, dc, N_3D, hid, did);

  checkResult(c_raja_omptarget, c_ref, N_3D);
//printArray(c_raja_omptarget, N_3D);


////////////////////////////////////////////

//----------------------------------------------------------------------------//
// Base OpenMP target variant (collapse2)
//----------------------------------------------------------------------------//

  std::cout << "\n6: Base OpenMP target 3D matrix init (collapse2)...\n";

  std::memset(c_base_omptarget, 0, N_3D * sizeof(int));
  allocAndInitOpenMPDeviceData(dc, c_base_omptarget, N_3D, did, hid);

  #pragma omp target is_device_ptr(dc) device( did )
  #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1) collapse(3)
  for (int n = 0; n < N; ++n) {
    for (int m = 0; m < M; ++m) {
      for (int l = 0; l < L; ++l) {
        dc[l + L*m + L*M*n] = l + L*m + L*M*n;
      }
    }
  }

  getOpenMPDeviceData(c_base_omptarget, dc, N_3D, hid, did);

  checkResult(c_base_omptarget, c_ref, N_3D);
//printArray(c_base_omptarget, N_3D);

std::cout << std::flush;

//----------------------------------------------------------------------------//
// RAJA OpenMP target variant
//----------------------------------------------------------------------------//

////////////////////////////////////////////
////////////////////////////////////////////
//
// ISSUE #2
//
// The following kernel show the issue. It hangs at runtime.
//
// The kernell policy has an OpenMP target collapse on the outer two loops 
// and a sequential innder loop. Kernel that collapses all 3 loops work and
// is included above. Similar kernel using OpenMP (no target) and collapse
// outer two loops works and are included above.
//
////////////////////////////////////////////
////////////////////////////////////////////

  std::cout << "\n7: RAJA OpenMP target 3D matrix init (collapse2)...\n" << std::flush;

  std::memset(c_raja_omptarget_collapse2, 0, N_3D * sizeof(int));
  initOpenMPDeviceData(dc, c_raja_omptarget_collapse2, N_3D, did, hid);

  using KERNEL_POL6 =
    RAJA::KernelPolicy<
      RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                RAJA::ArgList<2, 1>,  // n, m
        RAJA::statement::For<0, RAJA::seq_exec,  // l
          RAJA::statement::Lambda<0>
        >
      >
    >;

  RAJA::kernel<KERNEL_POL6>(
    RAJA::make_tuple(RAJA::RangeSegment{0, L},
                     RAJA::RangeSegment{0, M},
                     RAJA::RangeSegment{0, N}),
    [=](int l, int m, int n) {
      dc[l + L*m + L*M*n] = l + L*m + L*M*n;
  });

  getOpenMPDeviceData(c_raja_omptarget_collapse2, dc, N_3D, hid, did);

  checkResult(c_raja_omptarget_collapse2, c_ref, N_3D);
//printArray(c_raja_omptarget_collapse2, N_3D);

#endif


// cleanup

  delete [] c_ref;
  delete [] c_raja_omp;
  delete [] c_raja_omp_collapse2;
  delete [] c_base_omptarget;
  delete [] c_raja_omptarget;
  delete [] c_raja_omptarget_collapse2;

#if defined(RAJA_ENABLE_TARGET_OPENMP) && 0
  deallocOpenMPDeviceData(dc, did);
#endif

  std::cout << "\n 3D matrix init case DONE!...\n";
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

  return 0;
}

//
// Function to check result and report P/F.
//
void checkResult(int* c, int* c_ref, int len)
{
  bool correct = true;
  for (int i = 0; i < len; i++) {
    if ( correct && c[i] != c_ref[i] ) { correct = false; }
  }
  if ( correct ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
}

//
// Function to print array.
//
void printArray(int* v, int len)
{
  std::cout << std::endl;
  for (int i = 0; i < len; i++) {
    std::cout << "v[" << i << "] = " << v[i] << std::endl;
  }
  std::cout << std::endl;
}

