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

  std::cout << "\n\n 2D matrix init example...\n";

//
// Define matrix extents
//
  constexpr int Nj = 40;
  constexpr int Ni = 100;
  constexpr int N_2D = Nj * Ni;

  // Host allocations for variant results
  int* c_ref = new int[N_2D];
  int* c_base_omp = new int[N_2D];
  int* c_raja_omp = new int[N_2D];
  int* c_raja_omp_no_collapse = new int[N_2D];
  int* c_base_omptarget = new int[N_2D];
  int* c_raja_omptarget = new int[N_2D];
  int* c_raja_omptarget_no_collapse = new int[N_2D];

//----------------------------------------------------------------------------//
// C-style sequential variant establishes reference solution to compare with.
//----------------------------------------------------------------------------//

  std::cout << "\n C-style 2D matrix init...\n";

  std::memset(c_ref, 0, N_2D * sizeof(int));

  for (int j = 0; j < Nj; ++j) {
    for (int i = 0; i < Ni; ++i) {
      c_ref[i + Ni*j] = i + Ni*j;
    }
  }

//printArray(c_ref, N_2D);


//----------------------------------------------------------------------------//
// RAJA OpenMP variants (for comparing to RAJA OpenMP target)
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::cout << "\n1: RAJA OpenMP collapse 2D matrix init...\n";

  std::memset(c_raja_omp, 0, N_2D * sizeof(int));

  using EXEC_POL1 =
    RAJA::KernelPolicy<
      RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                RAJA::ArgList<0, 1>,
        RAJA::statement::Lambda<0>
      >
    >;

  RAJA::kernel<EXEC_POL1>( 
    RAJA::make_tuple(RAJA::RangeSegment{0, Nj},
                     RAJA::RangeSegment{0, Ni}),
    [=](int j, int i) {  
      c_raja_omp[i + Ni*j] = i + Ni*j;
  });

  checkResult(c_raja_omp, c_ref, N_2D);
//printArray(c_raja_omp, N_2D);


//////////////////////////////////////////////


  std::cout << "\n2: RAJA OpenMP collapse (Segs) 2D matrix init...\n";

  std::memset(c_raja_omp, 0, N_2D * sizeof(int));

  using EXEC_POL2 =
    RAJA::KernelPolicy<
      RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                RAJA::ArgList<0, 1>,
        RAJA::statement::Lambda<0, RAJA::Segs<0, 1>>
      >
    >;

  RAJA::kernel<EXEC_POL2>(
    RAJA::make_tuple(RAJA::RangeSegment{0, Nj},
                     RAJA::RangeSegment{0, Ni}),
    [=](int j, int i) {
      c_raja_omp[i + Ni*j] = i + Ni*j;
  });

  checkResult(c_raja_omp, c_ref, N_2D);
//printArray(c_raja_omp, N_2D);


//////////////////////////////////////////////

  std::cout << "\n3: RAJA OpenMP 2D matrix init (no collapse)...\n";

  std::memset(c_raja_omp_no_collapse, 0, N_2D * sizeof(int));
  
  using EXEC_POL3 =
    RAJA::KernelPolicy<
      RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
        RAJA::statement::For<1, RAJA::seq_exec,
          RAJA::statement::Lambda<0>
        >
      >
    >;

  RAJA::kernel<EXEC_POL3>(     
    RAJA::make_tuple(RAJA::RangeSegment{0, Nj},
                     RAJA::RangeSegment{0, Ni}),
    [=](int j, int i) {
      c_raja_omp_no_collapse[i + Ni*j] = i + Ni*j;
  });

  checkResult(c_raja_omp_no_collapse, c_ref, N_2D);
//printArray(c_raja_omp_no_collapse, N_2D);


//////////////////////////////////////////////

  std::cout << "\n4: RAJA OpenMP 2D matrix init (no collapse, segs)...\n";

  std::memset(c_raja_omp_no_collapse, 0, N_2D * sizeof(int));

  using EXEC_POL4 =
    RAJA::KernelPolicy<
      RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
        RAJA::statement::For<1, RAJA::seq_exec,
          RAJA::statement::Lambda<0, RAJA::Segs<0, 1>>
        >
      >
    >;

  RAJA::kernel<EXEC_POL4>(
    RAJA::make_tuple(RAJA::RangeSegment{0, Nj},
                     RAJA::RangeSegment{0, Ni}),
    [=](int j, int i) {
      c_raja_omp_no_collapse[i + Ni*j] = i + Ni*j;
  });

  checkResult(c_raja_omp_no_collapse, c_ref, N_2D);
//printArray(c_raja_omp_no_collapse, N_2D);

#endif


//----------------------------------------------------------------------------//
// OpenMP target variants
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_TARGET_OPENMP)


//----------------------------------------------------------------------------//
// Base OpenMP target variant
//----------------------------------------------------------------------------//

  std::cout << "\n5: Base OpenMP target 2D matrix init...\n";

  std::memset(c_base_omptarget, 0, N_2D * sizeof(int));
  allocAndInitOpenMPDeviceData(dc, c_base_omptarget, N_2D, did, hid);

  #pragma omp target is_device_ptr(dc) device( did )
  #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1) collapse(2)
  for (int j = 0; j < Nj; ++j) {
    for (int i = 0; i < Ni; ++i) {
      dc[i + Ni*j] = i + Ni*j;
    }
  }

  getOpenMPDeviceData(c_base_omptarget, dc, N_2D, hid, did);

  checkResult(c_base_omptarget, c_ref, N_2D);
//printArray(c_base_omptarget, N_2D);

//----------------------------------------------------------------------------//
// RAJA OpenMP target variants
//----------------------------------------------------------------------------//

  std::cout << "\n6: RAJA OpenMP target 2D matrix init (collapse)...\n";

  std::memset(c_raja_omptarget, 0, N_2D * sizeof(int));
  initOpenMPDeviceData(dc, c_raja_omptarget, N_2D, did, hid);

  using KERNEL_POL1 =
    RAJA::KernelPolicy<
      RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                RAJA::ArgList<0, 1>,
        RAJA::statement::Lambda<0>
      >
    >;

  RAJA::kernel<KERNEL_POL1>(
    RAJA::make_tuple(RAJA::RangeSegment{0, Nj},
                     RAJA::RangeSegment{0, Ni}),
    [=](int j, int i) {
      dc[i + Ni*j] = i + Ni*j;
  });

  getOpenMPDeviceData(c_raja_omptarget, dc, N_2D, hid, did);

  checkResult(c_raja_omptarget, c_ref, N_2D);
//printArray(c_raja_omptarget, N_2D);


////////////////////////////////////////////

  std::cout << "\n7: RAJA OpenMP target 2D matrix init (collapse, segs)...\n";

  std::memset(c_raja_omptarget, 0, N_2D * sizeof(int));
  initOpenMPDeviceData(dc, c_raja_omptarget, N_2D, did, hid);

  using KERNEL_POL2 =
    RAJA::KernelPolicy<
      RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                RAJA::ArgList<0, 1>,
        RAJA::statement::Lambda<0, RAJA::Segs<0, 1>>
      >
    >;

  RAJA::kernel<KERNEL_POL2>(
    RAJA::make_tuple(RAJA::RangeSegment{0, Nj},
                     RAJA::RangeSegment{0, Ni}),
    [=](int j, int i) {
      dc[i + Ni*j] = i + Ni*j;
  });

  getOpenMPDeviceData(c_raja_omptarget, dc, N_2D, hid, did);

  checkResult(c_raja_omptarget, c_ref, N_2D);
//printArray(c_raja_omptarget, N_2D);


////////////////////////////////////////////
////////////////////////////////////////////
//
// ISSUE #1
//
// The following two kernels show the issue. They hang at runtime.
//
// The kernell policy has an OpenMP target outer loop and a sequential
// innder loop. There is no loop collapse, which works as shown by the
// kernels above. Similar kernels using OpenMP (no target) work and
// are included above.
//
// Note that the OpenMP exec policy used here works in RAJA::kernel where
// there is no inner loop (i.e., only one loop). An example of this is
// in the reproducer-openmptarget.cpp file.
//
////////////////////////////////////////////
////////////////////////////////////////////

  std::cout << "\n8: RAJA OpenMP target 2D matrix init (no collapse)...\n";

  std::memset(c_raja_omptarget_no_collapse, 0, N_2D * sizeof(int));
  initOpenMPDeviceData(dc, c_raja_omptarget_no_collapse, N_2D, did, hid);

  using KERNEL_POL3 =
    RAJA::KernelPolicy<
      RAJA::statement::For<0, RAJA::omp_target_parallel_for_exec<threads_per_team>,
        RAJA::statement::For<1, RAJA::seq_exec,
          RAJA::statement::Lambda<0>
        >
      >
    >;

  RAJA::kernel<KERNEL_POL3>(
    RAJA::make_tuple(RAJA::RangeSegment{0, Nj},
                     RAJA::RangeSegment{0, Ni}),
    [=](int j, int i) {
      dc[i + Ni*j] = i + Ni*j;
  });

  getOpenMPDeviceData(c_raja_omptarget_no_collapse, dc, N_2D, hid, did);

  checkResult(c_raja_omptarget_no_collapse, c_ref, N_2D);
//printArray(c_raja_omptarget_no_collapse, N_2D);


////////////////////////////////////////////

  std::cout << "\n9: RAJA OpenMP target 2D matrix init (no collapse, Segs)...\n";

  std::memset(c_raja_omptarget_no_collapse, 0, N_2D * sizeof(int));
  initOpenMPDeviceData(dc, c_raja_omptarget_no_collapse, N_2D, did, hid);

  using KERNEL_POL4 =
    RAJA::KernelPolicy<
      RAJA::statement::For<0, RAJA::omp_target_parallel_for_exec<threads_per_team>,
        RAJA::statement::For<1, RAJA::seq_exec,
          RAJA::statement::Lambda<0, RAJA::Segs<0, 1>>
        >
      >
    >;

  RAJA::kernel<KERNEL_POL4>(
    RAJA::make_tuple(RAJA::RangeSegment{0, Nj},
                     RAJA::RangeSegment{0, Ni}),
    [=](int j, int i) {
      dc[i + Ni*j] = i + Ni*j;
  });

  getOpenMPDeviceData(c_raja_omptarget_no_collapse, dc, N_2D, hid, did);

  checkResult(c_raja_omptarget_no_collapse, c_ref, N_2D);
//printArray(c_raja_omptarget_no_collapse, N_2D);

#endif


// cleanup

  delete [] c_ref;
  delete [] c_base_omp;
  delete [] c_raja_omp;
  delete [] c_raja_omp_no_collapse;
  delete [] c_base_omptarget;
  delete [] c_raja_omptarget;
  delete [] c_raja_omptarget_no_collapse;

#if defined(RAJA_ENABLE_TARGET_OPENMP)
  deallocOpenMPDeviceData(dc, did);
#endif

  std::cout << "\n 2D matrix init case DONE!...\n";

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

