//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing basic integration tests for plugins with kernel.
///

#ifndef __TEST_PLUGIN_KERNEL_HPP__
#define __TEST_PLUGIN_KERNEL_HPP__

#include "test-plugin.hpp"


// Check that the plugin is called with the right Platform.
// Check that the plugin is called the correct number of times,
// once before and after each kernel capture for the capture counter,
// once before and after each kernel invocation for the launch counter.

// test with basic kernel
template <typename KernelPolicy,
          typename WORKING_RES,
          RAJA::Platform PLATFORM>
void PluginKernelTestImpl()
{
  SetupPluginVars spv(WORKING_RES::get_default());

  CounterData* data = plugin_test_resource->allocate<CounterData>(10);

  for (int i = 0; i < 10; i++) {

    RAJA::kernel<KernelPolicy>(
      RAJA::make_tuple(RAJA::RangeSegment(i,i+1)),
      PluginTestCallable{data}
    );

    CounterData loop_data;
    plugin_test_resource->memcpy(&loop_data, &data[i], sizeof(CounterData));
    ASSERT_EQ(loop_data.capture_platform_active, PLATFORM);
    ASSERT_EQ(loop_data.capture_counter_pre,     i+1);
    ASSERT_EQ(loop_data.capture_counter_post,    i);
    ASSERT_EQ(loop_data.launch_platform_active, PLATFORM);
    ASSERT_EQ(loop_data.launch_counter_pre,     i+1);
    ASSERT_EQ(loop_data.launch_counter_post,    i);
  }

  CounterData plugin_data;
  plugin_test_resource->memcpy(&plugin_data, plugin_test_data, sizeof(CounterData));
  ASSERT_EQ(plugin_data.capture_platform_active, RAJA::Platform::undefined);
  ASSERT_EQ(plugin_data.capture_counter_pre,     10);
  ASSERT_EQ(plugin_data.capture_counter_post,    10);
  ASSERT_EQ(plugin_data.launch_platform_active, RAJA::Platform::undefined);
  ASSERT_EQ(plugin_data.launch_counter_pre,     10);
  ASSERT_EQ(plugin_data.launch_counter_post,    10);

  plugin_test_resource->deallocate(data);
}


TYPED_TEST_SUITE_P(PluginKernelTest);
template <typename T>
class PluginKernelTest : public ::testing::Test
{
};

TYPED_TEST_P(PluginKernelTest, PluginKernel)
{
  using KernelPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<1>>::type;
  using PlatformHolder = typename camp::at<TypeParam, camp::num<2>>::type;

  PluginKernelTestImpl<KernelPolicy, ResType, PlatformHolder::platform>( );
}

REGISTER_TYPED_TEST_SUITE_P(PluginKernelTest,
                            PluginKernel);

#endif  //__TEST_PLUGIN_KERNEL_HPP__
