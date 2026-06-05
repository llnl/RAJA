//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing basic integration tests for plugins with launch.
///

#ifndef __TEST_PLUGIN_RESOURCE_LAUNCH_HPP__
#define __TEST_PLUGIN_RESOURCE_LAUNCH_HPP__

#include "test-plugin.hpp"


// Check that the plugin is called with the right Platform.
// Check that the plugin is called the correct number of times,
// once before and after each launch capture for the capture counter,
// once before and after each launch invocation for the launch counter.

// test with basic launch
template <typename LaunchPolicy,
          typename WORKING_RES,
          RAJA::Platform PLATFORM>
void PluginResourceLaunchTestImpl()
{
  WORKING_RES res;

  SetupPluginVars spv(res);

  CounterData* data = plugin_test_resource->allocate<CounterData>(10);

  for (int i = 0; i < 10; i++) {

    //Keep PluginTestCallable within a scope to ensure
    //destruction, consistent with other test
    {
      PluginTestCallable p_callable{data};

      RAJA::launch<LaunchPolicy>
        (res, RAJA::LaunchParams(RAJA::Teams(1), RAJA::Threads(1)),
         [=] RAJA_HOST_DEVICE(RAJA::LaunchContext RAJA_UNUSED_ARG(ctx))
         {
           p_callable(i);
         });
    }

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


TYPED_TEST_SUITE_P(PluginResourceLaunchTest);
template <typename T>
class PluginResourceLaunchTest : public ::testing::Test
{
};

TYPED_TEST_P(PluginResourceLaunchTest, PluginResourceLaunch)
{
  using LaunchPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<1>>::type;
  using PlatformHolder = typename camp::at<TypeParam, camp::num<2>>::type;

  PluginResourceLaunchTestImpl<LaunchPolicy, ResType, PlatformHolder::platform>( );
}

REGISTER_TYPED_TEST_SUITE_P(PluginResourceLaunchTest,
                            PluginResourceLaunch);

#endif  //__TEST_PLUGIN_LAUNCH_HPP__
