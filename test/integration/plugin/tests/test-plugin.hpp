//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing basic functional tests for atomic operations with forall and views.
///

#ifndef __TEST_PLUGIN_HPP__
#define __TEST_PLUGIN_HPP__

#include "RAJA/RAJA.hpp"
#include "RAJA/util/macros.hpp"
#include "counter.hpp"

#include "gtest/gtest.h"


CounterData* plugin_test_data = nullptr;

camp::resources::Resource* plugin_test_resource = nullptr;

struct SetupPluginVars
{
  SetupPluginVars(camp::resources::Resource const test_resource)
    : m_test_resource(test_resource)
  {
    // ASSERT_EQ(plugin_test_data, nullptr);
    // ASSERT_EQ(plugin_test_resource, nullptr);

    plugin_test_data = m_test_resource.allocate<CounterData>(1);
    plugin_test_resource = &m_test_resource;

    CounterData data;
    data.capture_platform_active = RAJA::Platform::undefined;
    data.capture_counter_pre     = 0;
    data.capture_counter_post    = 0;
    data.launch_platform_active = RAJA::Platform::undefined;
    data.launch_counter_pre     = 0;
    data.launch_counter_post    = 0;

    m_test_resource.memcpy(plugin_test_data, &data, sizeof(CounterData));
    m_test_resource.wait();
  }

  SetupPluginVars(SetupPluginVars const&) = delete;
  SetupPluginVars(SetupPluginVars &&) = delete;
  SetupPluginVars& operator=(SetupPluginVars const&) = delete;
  SetupPluginVars& operator=(SetupPluginVars &&) = delete;

  ~SetupPluginVars()
  {
    // ASSERT_NE(plugin_test_data, nullptr);
    // ASSERT_NE(plugin_test_resource, nullptr);

    m_test_resource.deallocate(plugin_test_data);
    plugin_test_data = nullptr;
    plugin_test_resource = nullptr;
  }

private:
  camp::resources::Resource m_test_resource;
};


struct PluginTestCallable
{
  PluginTestCallable(CounterData* data_optr)
    : m_data_optr(data_optr)
    , m_data_iptr(plugin_test_data)
  {
    clear_data();
  }

  RAJA_HOST_DEVICE PluginTestCallable(PluginTestCallable const& rhs)
    : m_data_optr(rhs.m_data_optr)
    , m_data_iptr(rhs.m_data_iptr)
    , m_data(rhs.m_data)
  {
#if !defined(RAJA_GPU_DEVICE_COMPILE_PASS_ACTIVE)
#if defined(RAJA_ENABLE_TARGET_OPENMP)
    if (omp_is_initial_device())
#endif
    {
      CounterData i_data;
      plugin_test_resource->memcpy(&i_data, m_data_iptr, sizeof(CounterData));
      plugin_test_resource->wait();

      if (m_data.capture_platform_active == RAJA::Platform::undefined &&
          i_data.capture_platform_active != RAJA::Platform::undefined) {
        m_data = i_data;
      }
    }
#endif
  }

  RAJA_HOST_DEVICE PluginTestCallable(PluginTestCallable && rhs)
    : m_data_optr(rhs.m_data_optr)
    , m_data_iptr(rhs.m_data_iptr)
    , m_data(rhs.m_data)
  {
    rhs.clear();
  }

  RAJA_HOST_DEVICE PluginTestCallable& operator=(PluginTestCallable const& rhs)
  {
    if (this != &rhs) {
      m_data_optr = rhs.m_data_optr;
      m_data_iptr = rhs.m_data_iptr;
      m_data      = rhs.m_data;
    }
    return *this;
  }

  RAJA_HOST_DEVICE PluginTestCallable& operator=(PluginTestCallable && rhs)
  {
    if (this != &rhs) {
      m_data_optr = rhs.m_data_optr;
      m_data_iptr = rhs.m_data_iptr;
      m_data      = rhs.m_data;
      rhs.clear();
    }
    return *this;
  }

  RAJA_HOST_DEVICE void operator()(int i) const
  {
    m_data_optr[i].capture_platform_active = m_data.capture_platform_active;
    m_data_optr[i].capture_counter_pre     = m_data.capture_counter_pre;
    m_data_optr[i].capture_counter_post    = m_data.capture_counter_post;
    m_data_optr[i].launch_platform_active = m_data_iptr->launch_platform_active;
    m_data_optr[i].launch_counter_pre     = m_data_iptr->launch_counter_pre;
    m_data_optr[i].launch_counter_post    = m_data_iptr->launch_counter_post;
  }

  RAJA_HOST_DEVICE void operator()(int count, int i) const
  {
    RAJA_UNUSED_VAR(count);
    operator()(i);
  }

private:
        CounterData* m_data_optr = nullptr;
  const CounterData* m_data_iptr = nullptr;
        CounterData  m_data;


  RAJA_HOST_DEVICE void clear()
  {
    m_data_optr = nullptr;
    m_data_iptr = nullptr;
    clear_data();
  }

  RAJA_HOST_DEVICE void clear_data()
  {
    m_data.capture_platform_active = RAJA::Platform::undefined;
    m_data.capture_counter_pre     = -1;
    m_data.capture_counter_post    = -1;
    m_data.launch_platform_active = RAJA::Platform::undefined;
    m_data.launch_counter_pre     = -1;
    m_data.launch_counter_post    = -1;
  }
};

#endif  //__TEST_PLUGIN_HPP__
