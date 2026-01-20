//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Utility routines for allocating/deallocating arrays in for forall tests. 
//

#ifndef __RAJA_test_forall_data_HPP__
#define __RAJA_test_forall_data_HPP__

#include "camp/resource.hpp"

template<typename T>
void allocateForallTestData(size_t N,
                            camp::resources::Resource work_res,
                            T** work_array,
                            T** check_array,
                            T** test_array)
{
  camp::resources::Resource host_res{camp::resources::Host::get_default()};

  *work_array = work_res.allocate<T>(RAJA::stripIndexType(N));

  *check_array = host_res.allocate<T>(RAJA::stripIndexType(N));
  *test_array = host_res.allocate<T>(RAJA::stripIndexType(N));
}

// for RAJA strongly typed indices
template<typename T,
         typename std::enable_if<std::is_base_of<RAJA::IndexValueBase, camp::type::ptr::rem<T>>::value>::type* = nullptr>
void allocateForallTestData(T N,
                            camp::resources::Resource work_res,
                            T** work_array,
                            T** check_array,
                            T** test_array)
{
  camp::resources::Resource host_res{camp::resources::Host::get_default()};

  *work_array = work_res.allocate<T>(RAJA::stripIndexType(N));

  *check_array = host_res.allocate<T>(RAJA::stripIndexType(N));
  *test_array = host_res.allocate<T>(RAJA::stripIndexType(N));
}

template<typename T>
void deallocateForallTestData(camp::resources::Resource work_res,
                              T* work_array,
                              T* check_array,
                              T* test_array)
{
  camp::resources::Resource host_res{camp::resources::Host::get_default()};

  work_res.deallocate(work_array);

  host_res.deallocate(check_array);
  host_res.deallocate(test_array);
}

#endif // __RAJA_test_forall_data_HPP__
