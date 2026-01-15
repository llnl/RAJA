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

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
  std::cout << "\n\n1D case using RAJA::forall...\n";

//
// Define vector length
//
  constexpr int N = 100;

//
// Allocate and initialize vector data.
//
  int* a = new int[N];
  int* b = new int[N];

  for (int i = 0; i < N; ++i) {
    a[i] = i;  
    b[i] = i+1;  
  }

#if defined(RAJA_ENABLE_TARGET_OPENMP)
  // Pointers for device-allocated data
  int* da;
  int* db;
  int* dc;
#endif

  // Host allocations for variant results
  int* c_ref = new int[N];
  int* c_raja_seq = new int[N];
  int* c_raja_omp = new int[N];
  int* c_base_omptarget = new int[N];
  int* c_raja_omptarget = new int[N];
  int* c_raja_omptarget_kernel = new int[N];

//----------------------------------------------------------------------------//
// C-style sequential variant establishes reference solution to compare with.
//----------------------------------------------------------------------------//
  std::memset(c_ref, 0, N * sizeof(int));

  std::cout << "\n Running C-style sequential vector addition...\n";

  for (int i = 0; i < N; ++i) {
    c_ref[i] = a[i] + b[i];
  }

//printArray(c_ref, N);


//----------------------------------------------------------------------------//
// RAJA::seq_exec variant
//----------------------------------------------------------------------------//
  std::memset(c_raja_seq, 0, N * sizeof(int));

  std::cout << "\n Running RAJA sequential vector addition...\n";

  RAJA::forall< RAJA::seq_exec >(RAJA::RangeSegment(0, N), 
  [=] (int i) { 
    c_raja_seq[i] = a[i] + b[i]; 
  });    

  checkResult(c_raja_seq, c_ref, N);
//printArray(c_raja_seq, N);


//----------------------------------------------------------------------------//
// RAJA::omp_parallel_for_exec variant
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::memset(c_raja_omp, 0, N * sizeof(int));

  std::cout << "\n Running RAJA OpenMP multithreaded vector addition...\n";

  RAJA::forall< RAJA::omp_parallel_for_exec >(RAJA::RangeSegment(0, N), 
  [=] (int i) { 
    c_raja_omp[i] = a[i] + b[i]; 
  });    

  checkResult(c_raja_omp, c_ref, N);
//printArray(c_raja_omp, N);

#endif


//----------------------------------------------------------------------------//
// OpenMP target variants
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_TARGET_OPENMP)

  int hid = omp_get_initial_device();
  int did = omp_get_default_device();

  allocAndInitOpenMPDeviceData(da, a, N, did, hid);
  allocAndInitOpenMPDeviceData(db, b, N, did, hid);


//----------------------------------------------------------------------------//
// Base OpenMP target variant
//----------------------------------------------------------------------------//

  std::memset(c_base_omptarget, 0, N * sizeof(int));
  allocAndInitOpenMPDeviceData(dc, c_base_omptarget, N, did, hid);

  std::cout << "\n Running base OpenMP target vector addition...\n";

  #pragma omp target is_device_ptr(da, db, dc) device( did )
  #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
  for (int i = 0; i < N; ++i ) {
    dc[i] = da[i] + db[i];
  }

  getOpenMPDeviceData(c_base_omptarget, dc, N, hid, did);

  checkResult(c_base_omptarget, c_ref, N);
//printArray(c_base_omptarget, N);


//----------------------------------------------------------------------------//
// RAJA OpenMP target variant (forall)
//----------------------------------------------------------------------------//

  std::memset(c_raja_omptarget, 0, N * sizeof(int));
  initOpenMPDeviceData(dc, c_raja_omptarget, N, did, hid);

  std::cout << "\n Running RAJA OpenMP target vector addition...\n";

  RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
  RAJA::RangeSegment(0, N), [=](int i) {
    dc[i] = da[i] + db[i];
  });

  getOpenMPDeviceData(c_raja_omptarget, dc, N, hid, did);

  checkResult(c_raja_omptarget, c_ref, N);
//printArray(c_raja_omptarget, N);


//----------------------------------------------------------------------------//
// RAJA OpenMP target variant (kernel)
//----------------------------------------------------------------------------//

  std::memset(c_raja_omptarget_kernel, 0, N * sizeof(int));
  initOpenMPDeviceData(dc, c_raja_omptarget_kernel, N, did, hid);

  std::cout << "\n Running RAJA OpenMP target vector addition (kernel)...\n";

  using KERNEL_POL0 = 
    RAJA::KernelPolicy<
      RAJA::statement::For<0, RAJA::omp_target_parallel_for_exec<threads_per_team>,
        RAJA::statement::Lambda<0>
      >
    >;

  RAJA::kernel<KERNEL_POL0>(
    RAJA::make_tuple(RAJA::RangeSegment{0, N}),
    [=](int i) {
      dc[i] = da[i] + db[i];
  });

  getOpenMPDeviceData(c_raja_omptarget_kernel, dc, N, hid, did);

  checkResult(c_raja_omptarget_kernel, c_ref, N);
//printArray(c_raja_omptarget_kernel, N);

#endif

// cleanup

  delete [] a;
  delete [] b;

  delete [] c_ref;
  delete [] c_raja_seq;
  delete [] c_raja_omp;
  delete [] c_base_omptarget;
  delete [] c_raja_omptarget;
  delete [] c_raja_omptarget_kernel;

#if defined(RAJA_ENABLE_TARGET_OPENMP)
  deallocOpenMPDeviceData(da, did);
  deallocOpenMPDeviceData(db, did);
  deallocOpenMPDeviceData(dc, did);
#endif

  std::cout << "\n 1D case DONE!...\n";

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

