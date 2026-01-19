//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing basic integration tests for plugins with forall.
///

#ifndef __TEST_PLUGIN_FORALL_HPP__
#define __TEST_PLUGIN_FORALL_HPP__

#include "test-plugin.hpp"


// Check that the plugin is called with the right Platform.
// Check that the plugin is called the correct number of times,
// once before and after each kernel capture for the capture counter,
// once before and after each kernel invocation for the launch counter.

// test with basic forall
template <typename ExecPolicy,
          typename WORKING_RES,
          RAJA::Platform PLATFORM>
void PluginForallTestImpl()
{
  SetupPluginVars spv(WORKING_RES::get_default());

  CounterData* data = plugin_test_resource->allocate<CounterData>(10);

  for (int i = 0; i < 10; i++) {

    RAJA::forall<ExecPolicy>(
      RAJA::RangeSegment(i,i+1),
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

// test with basic forall_Icount
template <typename ExecPolicy,
          typename WORKING_RES,
          RAJA::Platform PLATFORM>
void PluginForAllICountTestImpl()
{
  SetupPluginVars spv(WORKING_RES::get_default());

  CounterData* data = plugin_test_resource->allocate<CounterData>(10);

  for (int i = 0; i < 10; i++) {

    RAJA::forall_Icount<ExecPolicy>(
      RAJA::RangeSegment(i,i+1), i,
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

// test with IndexSet forall
template <typename ExecPolicy,
          typename WORKING_RES,
          RAJA::Platform PLATFORM>
void PluginForAllIdxSetTestImpl()
{
  SetupPluginVars spv(WORKING_RES::get_default());

  CounterData* data = plugin_test_resource->allocate<CounterData>(10);

  for (int i = 0; i < 10; i++) {

    RAJA::TypedIndexSet< RAJA::RangeSegment > iset;

    for (int j = i; j < 10; j++) {
      iset.push_back(RAJA::RangeSegment(j, j+1));
    }

    RAJA::forall<RAJA::ExecPolicy<RAJA::seq_segit, ExecPolicy>>(
      iset,
      PluginTestCallable{data}
    );

    for (int j = i; j < 10; j++) {
      CounterData loop_data;
      plugin_test_resource->memcpy(&loop_data, &data[j], sizeof(CounterData));
      ASSERT_EQ(loop_data.capture_platform_active, PLATFORM);
      ASSERT_EQ(loop_data.capture_counter_pre,     i+1);
      ASSERT_EQ(loop_data.capture_counter_post,    i);
      ASSERT_EQ(loop_data.launch_platform_active, PLATFORM);
      ASSERT_EQ(loop_data.launch_counter_pre,     i+1);
      ASSERT_EQ(loop_data.launch_counter_post,    i);
    }
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

// test with IndexSet forall_Icount
template <typename ExecPolicy,
          typename WORKING_RES,
          RAJA::Platform PLATFORM>
void PluginForAllIcountIdxSetTestImpl()
{
  SetupPluginVars spv(WORKING_RES::get_default());

  CounterData* data = plugin_test_resource->allocate<CounterData>(10);

  for (int i = 0; i < 10; i++) {

    RAJA::TypedIndexSet< RAJA::RangeSegment > iset;

    for (int j = i; j < 10; j++) {
      iset.push_back(RAJA::RangeSegment(j, j+1));
    }

    RAJA::forall_Icount<RAJA::ExecPolicy<RAJA::seq_segit, ExecPolicy>>(
      iset,
      PluginTestCallable{data}
    );

    for (int j = i; j < 10; j++) {
      CounterData loop_data;
      plugin_test_resource->memcpy(&loop_data, &data[j], sizeof(CounterData));
      ASSERT_EQ(loop_data.capture_platform_active, PLATFORM);
      ASSERT_EQ(loop_data.capture_counter_pre,     i+1);
      ASSERT_EQ(loop_data.capture_counter_post,    i);
      ASSERT_EQ(loop_data.launch_platform_active, PLATFORM);
      ASSERT_EQ(loop_data.launch_counter_pre,     i+1);
      ASSERT_EQ(loop_data.launch_counter_post,    i);
    }
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

TYPED_TEST_SUITE_P(PluginForallTest);
template <typename T>
class PluginForallTest : public ::testing::Test
{
};

TYPED_TEST_P(PluginForallTest, PluginForall)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<1>>::type;
  using PlatformHolder = typename camp::at<TypeParam, camp::num<2>>::type;

  PluginForallTestImpl<ExecPolicy, ResType, PlatformHolder::platform>( );
}

TYPED_TEST_P(PluginForallTest, PluginForAllICount)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<1>>::type;
  using PlatformHolder = typename camp::at<TypeParam, camp::num<2>>::type;

  PluginForAllICountTestImpl<ExecPolicy, ResType, PlatformHolder::platform>( );
}

TYPED_TEST_P(PluginForallTest, PluginForAllIdxSet)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<1>>::type;
  using PlatformHolder = typename camp::at<TypeParam, camp::num<2>>::type;

  PluginForAllIdxSetTestImpl<ExecPolicy, ResType, PlatformHolder::platform>( );
}

TYPED_TEST_P(PluginForallTest, PluginForAllIcountIdxSet)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<1>>::type;
  using PlatformHolder = typename camp::at<TypeParam, camp::num<2>>::type;

  PluginForAllIcountIdxSetTestImpl<ExecPolicy, ResType, PlatformHolder::platform>( );
}

REGISTER_TYPED_TEST_SUITE_P(PluginForallTest,
                            PluginForall,
                            PluginForAllICount,
                            PluginForAllIdxSet,
                            PluginForAllIcountIdxSet);

#endif  //__TEST_PLUGIN_FORALL_HPP__
