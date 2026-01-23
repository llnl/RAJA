//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#ifndef  RAJA_counter_HPP
#define  RAJA_counter_HPP


struct CounterData
{
  RAJA::Platform capture_platform_active = RAJA::Platform::undefined;
  int            capture_counter_pre     = 0;
  int            capture_counter_post    = 0;
  RAJA::Platform launch_platform_active = RAJA::Platform::undefined;
  int            launch_counter_pre     = 0;
  int            launch_counter_post    = 0;
};

// note the use of a pointer here to allow different types of memory
// to be used
extern CounterData* plugin_test_data;

extern camp::resources::Resource* plugin_test_resource;

#endif  // RAJA_counter_HPP
