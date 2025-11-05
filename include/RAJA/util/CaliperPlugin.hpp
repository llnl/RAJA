//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_CaliperPlugin_HPP
#define RAJA_CaliperPlugin_HPP

#include "RAJA/util/PluginOptions.hpp"
#include "RAJA/util/PluginStrategy.hpp"
#include <iostream>

#if defined(RAJA_ENABLE_CALIPER)

namespace RAJA
{
namespace util
{

// Internal variable to toggle Caliper profiling on and off
// Users have access to a global function.
inline bool RAJA_caliper_profile = false;

// Experimental global function to toggle Caliper
// profiling on and off
inline void SetRAJACaliperProfiling(bool enable)
{
  RAJA_caliper_profile = enable;
}

class CaliperPlugin : public ::RAJA::util::PluginStrategy
{
public:
  CaliperPlugin();

  void preLaunch(const RAJA::util::PluginContext& p) override;

  void postLaunch(const RAJA::util::PluginContext& p) override;
};

void linkCaliperPlugin();

}  // end namespace util
}  // end namespace RAJA

#endif

#endif
