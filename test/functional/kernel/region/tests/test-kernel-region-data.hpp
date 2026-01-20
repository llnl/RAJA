//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_REGION_DATA_HPP__
#define __TEST_KERNEL_REGION_DATA_HPP__

template <typename T>
void allocRegionTestData(int N,
                         camp::resources::Resource work_res,
                         T** work1, T** work2, T** work3,
                         camp::resources::Resource host_res,
                         T** check)
{
  *work1 = work_res.allocate<T>(N);
  *work2 = work_res.allocate<T>(N);
  *work3 = work_res.allocate<T>(N);

  *check = host_res.allocate<T>(N);
}

template <typename T>
void deallocRegionTestData(camp::resources::Resource work_res,
                           T* work1, T* work2, T* work3,
                           camp::resources::Resource host_res,
                           T* check)
{
  work_res.deallocate(work1);
  work_res.deallocate(work2);
  work_res.deallocate(work3);

  host_res.deallocate(check);
}

#endif  // __TEST_KERNEL_REGION_UTILS_HPP__
