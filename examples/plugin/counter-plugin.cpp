//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// _plugin_example_start
#include "RAJA/util/PluginStrategy.hpp"

#include <iostream>

class CounterPlugin :
  public RAJA::util::PluginStrategy
{
  public:
  void preCapture(const RAJA::util::PluginContext& p) override {
    if (p.platform == RAJA::Platform::host) 
    {
      std::cout << " [CounterPlugin]: Capturing host kernel for the " << ++host_capture_counter << " time!" << std::endl;
    }
    else
    {
      std::cout << " [CounterPlugin]: Capturing device kernel for the " << ++device_capture_counter << " time!" << std::endl;
    }
  }

  void preLaunch(const RAJA::util::PluginContext& p) override {
    if (p.platform == RAJA::Platform::host)
    {
      std::cout << " [CounterPlugin]: Launching host kernel for the " << ++host_launch_counter << " time!" << std::endl;
    }
    else
    {
      std::cout << " [CounterPlugin]: Launching device kernel for the " << ++device_launch_counter << " time!" << std::endl;
    }
  }

  private:
   int host_capture_counter;
   int device_capture_counter;
   int host_launch_counter;
   int device_launch_counter;
};

// Statically loading plugin.
static RAJA::util::PluginRegistry::add<CounterPlugin> P("Counter", "Counts number of kernel launches.");

// Dynamically loading plugin.
extern "C" RAJA::util::PluginStrategy *RAJAGetPlugin ()
{
  return new CounterPlugin;
}
// _plugin_example_end
